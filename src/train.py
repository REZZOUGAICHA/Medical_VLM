"""Fine-tune a pretrained MarianMT checkpoint on a medical parallel corpus.

Mirrors teammate 2's per-direction MarianMT fine-tuning pattern: one pretrained
Helsinki-NLP/opus-mt-* checkpoint in, one fine-tuned checkpoint out, driven by
a small YAML config so the same script covers every direction (en-zh, zh-en,
en-hi, hi-en, ...) with only the config changed.

Usage:
    python src/train.py --config configs/en-zh.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_parallel_file_pair(src_path: Path, tgt_path: Path) -> Dataset:
    with src_path.open(encoding="utf-8") as f:
        src_lines = [l.rstrip("\n") for l in f]
    with tgt_path.open(encoding="utf-8") as f:
        tgt_lines = [l.rstrip("\n") for l in f]
    assert len(src_lines) == len(tgt_lines), (
        f"{src_path} has {len(src_lines)} lines, {tgt_path} has {len(tgt_lines)}"
    )
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    base_model = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    max_src_len = cfg.get("max_source_length", 128)
    max_tgt_len = cfg.get("max_target_length", 128)

    print(f"Loading base model {base_model} ...")
    tokenizer = MarianTokenizer.from_pretrained(base_model)
    model = MarianMTModel.from_pretrained(base_model)

    train_ds = load_parallel_file_pair(Path(cfg["train_src"]), Path(cfg["train_tgt"]))
    val_ds = load_parallel_file_pair(Path(cfg["val_src"]), Path(cfg["val_tgt"]))
    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["src"], max_length=max_src_len, truncation=True,
        )
        labels = tokenizer(
            text_target=batch["tgt"], max_length=max_tgt_len, truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_preds):
        import sacrebleu

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {"bleu": bleu.score}

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=cfg.get("learning_rate", 2e-5),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        weight_decay=cfg.get("weight_decay", 0.01),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        predict_with_generate=True,
        generation_max_length=max_tgt_len,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", []),
        group_by_length=cfg.get("group_by_length", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
    )
    # Filter against whatever this installed transformers version actually
    # accepts -- API surface (e.g. field renames) drifts across major
    # versions, and this script needs to run unmodified on both a pinned
    # local install and whatever Colab happens to have pulled in.
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    dropped = {k: v for k, v in training_args_kwargs.items() if k not in valid_fields}
    if dropped:
        print(f"NOTE: this transformers install doesn't support {list(dropped)} -- skipping (had no effect anyway if unsupported)")
    training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in valid_fields}

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 3))],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()

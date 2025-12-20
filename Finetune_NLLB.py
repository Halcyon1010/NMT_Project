import argparse
import os
import numpy as np
import sacrebleu

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def parse_args():
    p = argparse.ArgumentParser()

    # Model / data
    p.add_argument("--model_name_or_path", type=str, default=r"/mnt/afs/250010063/AP0004_Midterm/NLLB_600M")
    p.add_argument("--data_root", type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/data')
    p.add_argument("--output_dir", type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/results/nllb_600m_zh2en')
    p.add_argument("--train_file", type=str, default="train_100k.jsonl")
    p.add_argument("--valid_file", type=str, default="valid.jsonl")
    p.add_argument("--test_file", type=str, default="test.jsonl")

    # JSON keys
    p.add_argument("--src_key", type=str, default="zh")
    p.add_argument("--tgt_key", type=str, default="en")

    # Lengths
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=256)

    # NLLB language codes (critical)
    p.add_argument("--src_lang", type=str, default="zho_Hans")
    p.add_argument("--tgt_lang", type=str, default="eng_Latn")

    # Training hyperparams
    p.add_argument("--per_device_train_batch_size", type=int, default=60)
    p.add_argument("--per_device_eval_batch_size", type=int, default=60)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Precision switches (avoid setting both true)
    p.add_argument("--fp16", type=int, default=0)
    p.add_argument("--bf16", type=int, default=1)

    # Generation
    p.add_argument("--num_beams", type=int, default=4)

    return p.parse_args()


def main():
    args = parse_args()

    data_files = {
        "train": os.path.join(args.data_root, args.train_file),
        "validation": os.path.join(args.data_root, args.valid_file),
        "test": os.path.join(args.data_root, args.test_file),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load tokenizer / model
    # For NLLB, fast tokenizer is generally fine, but keeping use_fast=False is conservative.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # IMPORTANT: set language codes on tokenizer (NLLB specific)
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang

    # IMPORTANT: NLLB needs forced BOS token for target language during generation
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    # 2) Load dataset
    ds = load_dataset("json", data_files=data_files)

    # 3) Preprocess
    def preprocess(batch):
        src_texts = batch[args.src_key]
        tgt_texts = batch[args.tgt_key]

        # Encode source; the tokenizer will add the correct src_lang tokens based on tokenizer.src_lang
        model_inputs = tokenizer(
            src_texts,
            max_length=args.max_source_len,
            truncation=True,
            padding=False,
        )

        # Encode targets
        labels = tokenizer(
            text_target=tgt_texts,
            max_length=args.max_target_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # Collator will pad and set label pad tokens to -100 (standard)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 4) Training args
    # NOTE: Your transformers version uses eval_strategy (not evaluation_strategy)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=args.num_beams,
        # Critical for NLLB to generate in the correct target language
        generation_config=None,  # keep None; we pass forced_bos_token_id via trainer below
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
    )

    # 5) Metrics (robust decode to avoid SentencePiece crashes)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        # preds can be tuple in some versions
        if isinstance(preds, tuple):
            preds = preds[0]

        # If preds are logits, take argmax
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        # Replace negatives (e.g. -100) with pad before decoding
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

        pred_str = tokenizer.batch_decode(preds.astype(np.int64), skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels.astype(np.int64), skip_special_tokens=True)

        bleu = sacrebleu.corpus_bleu(pred_str, [label_str]).score
        return {"bleu": bleu}

    # 6) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,  # FutureWarning ok in your version
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Sanity prints
    print("train/val/test:", len(tokenized["train"]), len(tokenized["validation"]), len(tokenized["test"]))
    print("model_type:", getattr(model.config, "model_type", None))
    print("vocab_size:", getattr(model.config, "vocab_size", None), "pad_token_id:", tokenizer.pad_token_id)
    print("src_lang:", args.src_lang, "tgt_lang:", args.tgt_lang, "forced_bos_token_id:", forced_bos_token_id)

    sample = tokenized["train"][0]
    labs = sample["labels"]
    print("labels min/max:", min(labs), max(labs), "num_tokens:", len(labs))

    # 8) Patch generation kwargs: force target language during generate()
    # For older transformers, easiest is to set this on trainer.model.generation_config or pass via kwargs in predict/evaluate.
    # We'll set on model.generation_config if present.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_bos_token_id = forced_bos_token_id

    # 9) Train
    trainer.train()

    # 10) Evaluate on test
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    # 11) Save final
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("[OK] Saved to:", final_dir)


if __name__ == "__main__":
    main()

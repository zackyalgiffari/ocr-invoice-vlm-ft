"""LoRA SFT of a Qwen3.5 VLM on invoice/receipt -> JSON.

Runs on the Colab VM (see scripts/remote_job.sh). Loss is computed only on the
assistant answer (the JSON plus the closing <|im_end|>), never on the prompt or image.

    python -m invoice_vlm.train --config configs/qwen35_2b_lora.yaml --output-dir /content/run/model
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

from invoice_vlm.config import cap_pixels, load_config
from invoice_vlm.data import build_messages, load_split

IGNORE_INDEX = -100


class SFTCollator:
    """Turns rows {image, doc_type, target} into a padded batch with masked labels."""

    def __init__(self, processor: Any, max_pixels: int, min_pixels: int) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.image_token = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self._shown = False

    def __call__(self, rows: list[dict]) -> dict:
        import torch

        images, full_texts, prompt_texts = [], [], []
        for row in rows:
            image = cap_pixels(row["image"], self.max_pixels, self.min_pixels)
            messages = build_messages(image, row["doc_type"], row["target"])
            full_texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
            )
            images.append([image])

        self.tokenizer.padding_side = "right"
        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = IGNORE_INDEX
        for i, prompt in enumerate(prompt_texts):
            # The processor expands one <|image_pad|> into N tokens; mirror that on the
            # prompt so its token length lines up with the prefix of the full sequence.
            n_image_tokens = int((batch["input_ids"][i] == self.image_token_id).sum())
            expanded = prompt.replace(self.image_token, self.image_token * n_image_tokens, 1)
            prompt_len = len(self.tokenizer(expanded, add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = IGNORE_INDEX
        if (labels != IGNORE_INDEX).sum() == 0:
            raise RuntimeError("every label is masked: prompt/answer split is wrong")
        if not self._shown:
            # Once per run: show exactly what the loss sees, so a misaligned mask is obvious
            # in the log (it should start at the JSON's "{" and end with <|im_end|>).
            trained = batch["input_ids"][0][labels[0] != IGNORE_INDEX]
            text = self.tokenizer.decode(trained, skip_special_tokens=False)
            print(f"[collator] supervised text: {text[:80]!r} ... {text[-40:]!r}", flush=True)
            self._shown = True
        batch["labels"] = labels
        return dict(batch)


def language_model_linear_names(model: Any) -> list[str]:
    """Full names of every nn.Linear in the language model (not the vision tower/lm_head)."""
    import torch

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and not name.endswith("lm_head")
    ]
    if not names:
        sample = [n for n, _ in model.named_modules()][:40]
        raise RuntimeError(f"no Linear layers under 'language_model'; module names: {sample}")
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", help="override config model_id")
    p.add_argument("--limit", type=int, help="cap examples per source (smoke test)")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps")
    p.add_argument("--no-push", action="store_true", help="do not push to the HF Hub")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_id = args.model_id or cfg["model_id"]
    push = not args.no_push

    import torch
    from huggingface_hub import whoami
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    if push:
        # Fail before spending GPU time if the Hub token is missing or wrong.
        print(f"[train] HF user: {whoami()['name']}", flush=True)

    tc = cfg["train"]
    torch.manual_seed(tc["seed"])

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings

    targets = language_model_linear_names(model)
    suffixes = sorted({n.rsplit(".", 1)[-1] for n in targets})
    print(f"[train] LoRA on {len(targets)} Linear layers, suffixes: {suffixes}", flush=True)
    lora = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    sources = tuple(cfg["sources"])
    train_ds = load_split("train", sources, limit=args.limit, seed=tc["seed"])
    eval_ds = load_split("validation", sources, limit=args.limit)
    print(f"[train] train={len(train_ds)} validation={len(eval_ds)}", flush=True)

    steps_per_epoch = math.ceil(len(train_ds) / (tc["per_device_batch_size"] * tc["grad_accum"]))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * tc["epochs"]
    warmup_steps = max(1, int(total_steps * tc["warmup_fraction"]))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=tc["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=tc["per_device_batch_size"],
        per_device_eval_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=tc["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tc["logging_steps"],
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.max_steps if args.max_steps > 0 else None,
        save_total_limit=2,
        remove_unused_columns=False,  # the collator needs the raw image/doc_type/target columns
        dataloader_num_workers=2,
        report_to="none",
        seed=tc["seed"],
        push_to_hub=push,
        hub_model_id=cfg["hub"]["model_id"] if push else None,
        hub_private_repo=cfg["hub"]["private"] if push else None,
        hub_strategy="checkpoint",  # each save is uploaded, so a dropped VM loses at most one epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTCollator(processor, cfg["max_pixels"], cfg.get("min_pixels", 0)),
    )
    trainer.train()

    trainer.save_model(args.output_dir)  # LoRA adapter only
    processor.save_pretrained(args.output_dir)
    if push:
        trainer.push_to_hub(commit_message="Final LoRA adapter")
        processor.push_to_hub(cfg["hub"]["model_id"], private=cfg["hub"]["private"])
    print(f"[train] done; adapter in {os.path.abspath(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()

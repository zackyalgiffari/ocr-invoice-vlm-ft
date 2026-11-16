# ocr-invoice-vlm-ft

Fine-tuning a small vision-language model to extract structured fields from invoices and
receipts (image → JSON). Training runs on a Google Colab GPU, driven from the `colab` CLI.

| | |
|---|---|
| Base model | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0), LoRA on the language model |
| Data | [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Indonesian receipts, CC-BY-4.0, 800/100/100) + [`katanaml-org/invoices-donut-data-v1`](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) (invoices, MIT, 425/50/26) |
| Compute | Colab L4 24 GB (bf16) via `colab` CLI |
| Weights | [`zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora`](https://huggingface.co/zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora) (LoRA adapter, Apache-2.0) |
| License | Code and adapter: Apache-2.0 ([LICENSE](LICENSE)) |

## How it works

- **Target**: the dataset's Donut `gt_parse` object, serialized as compact JSON. One model learns
  both schemas; the instruction says whether the image is a receipt or an invoice.
- **Loss** only on the assistant answer (JSON + `<|im_end|>`), never on the prompt or image tokens.
- **Images** are downscaled to at most `max_pixels` (Qwen3.5: one visual token per 32×32 px).
- **Checkpoints** are pushed to the HF repo every epoch (`hub_strategy="checkpoint"`),
  so a dropped Colab VM loses at most one epoch.
- **Metrics** (`src/invoice_vlm/metrics.py`): JSON validity, field-level precision/recall/F1 over
  `(path, value)` leaves with list order ignored, per-document exact match, value-only F1, and a
  lenient F1 that forgives label whitespace.

## Layout

```
configs/qwen35_2b_lora.yaml   model, data, LoRA, hyperparameters, hub repo
src/invoice_vlm/
  data.py        load + convert both datasets, chat messages
  train.py       LoRA SFT with transformers Trainer + PEFT
  evaluate.py    test-split generation -> predictions.jsonl, metrics.json
  infer.py       one image -> JSON
  metrics.py     scoring (strict, value-only and lenient F1); rescore: python -m invoice_vlm.metrics predictions.jsonl
scripts/
  colab_train.sh   local: new VM -> upload HEAD -> install -> run in background -> poll -> download -> stop
  remote_job.sh    on the VM: smoke | baseline | full | demo | hubeval
docs/model_card.md  source of the HF repo's README (upload: hf upload <repo> docs/model_card.md README.md)
tests/            CPU unit tests (no model download)
```

## Usage

Prerequisites: `colab` CLI authenticated (`colab sessions` works) and `hf auth login` done
(answer **n** to "add token as git credential").

```bash
# Local tests (CPU only; no model download).
uvx --with pillow --with pyyaml pytest -q -p no:cacheprovider

# Colab runs. Each one releases its VM at the end, even on error or Ctrl-C.
scripts/colab_train.sh smoke      # 10 steps on 32 examples: versions, VRAM, adapter reload
scripts/colab_train.sh baseline   # zero-shot base model on the test split
scripts/colab_train.sh full       # 3 epochs + test evaluation, adapter pushed to HF
scripts/colab_train.sh demo       # infer.py with the Hub adapter on one test receipt + invoice
scripts/colab_train.sh hubeval    # test split at batch 1: zero-shot and the Hub adapter (published numbers)

# One image, on any machine with a GPU and the package installed:
python -m invoice_vlm.infer --config configs/qwen35_2b_lora.yaml \
  --adapter zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora --type receipt struk.jpg
```

On an L4, the demo takes 52 s for the first image (base model download and load, adapter merge)
and 27 s for the next one, including model load. The demo prints each prediction next to its gold
label in `run.log`.

Results land in `outputs/<mode>-<time>-<commit>/` (gitignored): `run.log`, `metrics.json`,
`predictions.jsonl`, and compute-unit balance before/after.

`FAST_KERNELS=1` installs `flash-linear-attention` and compiles `causal-conv1d`, which removes the
"falling back to reference PyTorch" warnings for Qwen3.5's linear-attention layers. Measured with
smoke on an L4 (2026-11-17), it is not worth it, so it is off by default. Training was 4% faster
(225 s vs 235 s for 10 steps), evaluation was no faster, and scores were identical, but the compile
adds about 17 minutes of setup. Over a full run that is about 6 minutes saved for 17 spent.

The VM runs the **committed** `HEAD` (via `git archive`), so commit before launching. The HF
token is uploaded as a file and read through `HF_TOKEN_PATH`, so it never appears in
`colab exec --env` arguments or in the CLI's session history.

## Results

Test split: 126 documents (100 CORD receipts, 26 invoices), Colab L4, greedy decoding, batch 1
(`scripts/colab_train.sh hubeval`, which evaluates the adapter as published on the Hub). *Value F1*
scores values alone, ignoring keys, so it shows how much a zero-shot model reads even when it
invents its own key names (which makes its Field F1 zero).

| Model | Split | JSON valid | Field F1 | Value F1 | Exact match | Lenient F1 | Lenient EM |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B zero-shot | test | 96.8% | 0.000 | 0.462 | 0.0% | 0.000 | 0.0% |
| Qwen3.5-2B + LoRA | test | 100% | 0.886 | 0.916 | 38.9% | 0.921 | 48.4% |

*Lenient* ignores whitespace inside values and object-vs-one-item-list, two label artifacts rather
than reading errors (the invoice labels write `$ 978,12` / `1 490,55`; the model writes `$978,12` /
`1490,55`). Per source after LoRA:

| Source | n | Field F1 | Exact match | Lenient F1 | Lenient EM |
|---|---|---|---|---|---|
| CORD receipts | 100 | 0.869 | 48.0% | 0.892 | 52.0% |
| Invoices | 26 | 0.910 | 3.8% | 0.964 | 34.6% |

The remaining misses concentrate in CORD item names (`menu[].nm`), unit prices and counts, and in
invoice item descriptions, IBANs (often one dropped or swapped character) and totals.

**Batch size.** Batched generation (the configs use `eval.batch_size: 4`, left-padded) changes
16 of 126 outputs relative to batch 1, but the aggregate scores barely move. For the LoRA model:
Field F1 0.885 vs 0.886, Lenient F1 0.918 vs 0.921. For zero-shot: Value F1 0.470 vs 0.462.
Batch 4 takes 8.5 minutes for the LoRA test set and batch 1 takes 20, so the configs keep 4 and
the published numbers use 1.

Training: 3 epochs, 462 steps in 2.5 h on L4 (~20 s/step), validation loss 0.039 → 0.033 → 0.033,
peak GPU memory 11.7 / 22 GiB. The full run cost about 6 compute units.

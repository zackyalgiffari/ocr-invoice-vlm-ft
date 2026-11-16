# ocr-invoice-vlm-ft

Fine-tuning a small vision-language model to extract structured fields from invoices and
receipts (image → JSON). Training runs on a Google Colab GPU, driven from the `colab` CLI.

| | |
|---|---|
| Base model | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0), LoRA on the language model |
| Data | [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Indonesian receipts, CC-BY-4.0, 800/100/100) + [`katanaml-org/invoices-donut-data-v1`](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) (invoices, MIT, 425/50/26) |
| Compute | Colab L4 24 GB (bf16) via `colab` CLI |
| Weights | Private HF model repo `zackyalgiffari/qwen3.5-2b-invoice-extract` |

## How it works

- **Target**: the dataset's Donut `gt_parse` object, serialized as compact JSON. One model learns
  both schemas; the instruction says whether the image is a receipt or an invoice.
- **Loss** only on the assistant answer (JSON + `<|im_end|>`), never on the prompt or image tokens.
- **Images** are downscaled to at most `max_pixels` (Qwen3.5: one visual token per 32×32 px).
- **Checkpoints** are pushed to the private HF repo every epoch (`hub_strategy="checkpoint"`),
  so a dropped Colab VM loses at most one epoch.
- **Metrics** (`src/invoice_vlm/metrics.py`): JSON validity, field-level precision/recall/F1 over
  `(path, value)` leaves with list order ignored, and per-document exact match.

## Layout

```
configs/qwen35_2b_lora.yaml   model, data, LoRA, hyperparameters, hub repo
src/invoice_vlm/
  data.py        load + convert both datasets, chat messages
  train.py       LoRA SFT with transformers Trainer + PEFT
  evaluate.py    test-split generation -> predictions.jsonl, metrics.json
  infer.py       one image -> JSON
  metrics.py     scoring
scripts/
  colab_train.sh   local: new VM -> upload HEAD -> install -> run in background -> poll -> download -> stop
  remote_job.sh    on the VM: smoke | baseline | full
tests/            CPU unit tests (no model download)
```

## Usage

Prerequisites: `colab` CLI authenticated (`colab sessions` works) and `hf auth login` done
(answer **n** to "add token as git credential").

```bash
# Local tests. Runs from the uv cache, so no .venv is created on the slow /mnt/e drive.
uvx --with pillow --with pyyaml pytest -q -p no:cacheprovider

# Colab runs. Each one releases its VM at the end, even on error or Ctrl-C.
scripts/colab_train.sh smoke      # 10 steps on 32 examples: versions, VRAM, adapter reload
scripts/colab_train.sh baseline   # zero-shot base model on the test split
scripts/colab_train.sh full       # 3 epochs + test evaluation, adapter pushed to HF
```

Results land in `outputs/<mode>-<time>-<commit>/` (gitignored): `run.log`, `metrics.json`,
`predictions.jsonl`, and compute-unit balance before/after.

The VM runs the **committed** `HEAD` (via `git archive`), so commit before launching. The HF
token is uploaded as a file and read through `HF_TOKEN_PATH`, so it never appears in
`colab exec --env` arguments or in the CLI's session history.

## Results

Test split: 126 documents (100 CORD receipts, 26 invoices), Colab L4. *Value F1* scores values
alone, ignoring keys, so it shows how much a zero-shot model reads even when it invents its own
key names (which makes its Field F1 zero).

| Model | Split | JSON valid | Field F1 | Value F1 | Exact match |
|---|---|---|---|---|---|
| Qwen3.5-2B zero-shot | test | 97.6% | 0.000 | 0.470 | 0.0% |
| Qwen3.5-2B + LoRA | test | – | – | – | – |

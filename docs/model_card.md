---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: image-text-to-text
datasets:
- naver-clova-ix/cord-v2
- katanaml-org/invoices-donut-data-v1
language:
- en
- id
tags:
- lora
- document-understanding
- information-extraction
- receipts
- invoices
---

# qwen3.5-2b-receipt-invoice-extract-lora

A LoRA adapter for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) that reads an image of
a receipt or an invoice and returns its fields as one JSON object. The output follows the Donut label
schema of the two training datasets.

Training code, evaluation and the run logs are in the GitHub repo `zackyalgiffari/ocr-invoice-vlm-ft`.

## Results

Test split: 126 documents (100 CORD receipts, 26 invoices). Greedy decoding, images capped at
1280×1280 px.

| Model | JSON valid | Field F1 | Value F1 | Exact match | Lenient F1 | Lenient EM |
|---|---|---|---|---|---|---|
| Qwen3.5-2B zero-shot | 97.6% | 0.000 | 0.470 | 0.0% | 0.000 | 0.0% |
| **+ this adapter** | **100%** | **0.885** | **0.916** | **38.9%** | **0.918** | **46.8%** |

| Source | n | Field F1 | Exact match | Lenient F1 | Lenient EM |
|---|---|---|---|---|---|
| CORD receipts | 100 | 0.864 | 47.0% | 0.887 | 51.0% |
| Invoices | 26 | 0.916 | 7.7% | 0.964 | 30.8% |

- **Field F1**: precision/recall over `(path, value)` leaves, with list order ignored, after
  lower-casing and collapsing whitespace.
- **Value F1**: the same comparison on values alone. The zero-shot model reads the text but invents
  its own keys, so it scores 0 on Field F1.
- **Lenient**: also ignores whitespace inside values and treats an object the same as a one-item list.
  These are label artifacts, not reading errors: the invoice labels write `$ 978,12`, while the model
  writes `$978,12`.
- **Exact match**: the whole document is correct.

## Usage

The prompt must match training. The only difference between the two document types is one word
(`receipt` / `invoice`).

```python
import json, torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = "Qwen/Qwen3.5-2B"
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora").merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

image = Image.open("receipt.jpg").convert("RGB")
max_pixels = 1280 * 1280  # the training cap; larger images were downscaled, aspect kept
if image.width * image.height > max_pixels:
    s = (max_pixels / (image.width * image.height)) ** 0.5
    image = image.resize((int(image.width * s), int(image.height * s)), Image.BICUBIC)

doc_type = "receipt"  # or "invoice"
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": f"Extract all fields from this {doc_type} image. Respond with only a JSON object."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[[image]], return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
print(json.loads(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
```

With the GitHub repo installed, the equivalent command is
`python -m invoice_vlm.infer --config configs/qwen35_2b_lora.yaml --adapter zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora --type receipt receipt.jpg`.

## Training

- **Data**: CORD v2 train (800 receipts, Indonesian) plus invoices-donut-data-v1 train (425
  invoices), interleaved. The target is each dataset's `gt_parse` serialized as compact JSON.
- **Method**: LoRA (r=16, alpha=32, dropout 0.05) on every linear layer of the language model. The
  vision tower is frozen. Loss is computed only on the answer tokens.
- **Hyperparameters**: 3 epochs, learning rate 1e-4 with cosine schedule and 3% warmup, effective
  batch size 8 (1 × 8 accumulation), bf16, gradient checkpointing, seed 42.
- **Compute**: one Colab L4 (24 GB). Training took 2.5 h (462 steps) with a peak of 11.7 GiB.

| Epoch | Step | Training loss | Validation loss |
|---|---|---|---|
| 1 | 154 | 0.0514 | 0.0388 |
| 2 | 308 | 0.0324 | 0.0332 |
| 3 | 462 | 0.0134 | 0.0328 |

## Limitations

- **Two layouts only.** The model learned two schemas: CORD's receipt keys (`menu[].nm`,
  `total.total_price`, ...) and the invoice dataset's keys (`header`, `items[]`, `summary`). Expect
  those keys, not a general invoice schema. Other layouts and languages are untested.
- **Small invoice set.** The invoice test split has only 26 documents, so its scores are noisy.
- **Single characters slip.** Remaining errors include dropped or swapped characters in long codes
  (IBANs), item names, and totals. Check amounts before using them for anything that matters.

## Licenses

The adapter is Apache-2.0, like the base model. The training data is
[CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC-BY-4.0) and
[invoices-donut-data-v1](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) (MIT).

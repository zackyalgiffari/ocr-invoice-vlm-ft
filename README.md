# ocr-invoice-vlm-ft

Fine-tuning a small vision-language model to extract structured fields from invoices and
receipts (image → JSON). Training runs on a Google Colab GPU, driven from the `colab` CLI.

| | |
|---|---|
| Base model | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0), LoRA |
| Smoke-test model | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Data | [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Indonesian receipts, CC-BY-4.0), [`katanaml-org/invoices-donut-data-v1`](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) (invoices, MIT) |
| Compute | Colab L4 24 GB (bf16) via `colab` CLI |
| Weights | Private Hugging Face model repo `zackyalgiffari/qwen3.5-2b-invoice-extract` |

## Status

Work in progress. Repository scaffolding only.

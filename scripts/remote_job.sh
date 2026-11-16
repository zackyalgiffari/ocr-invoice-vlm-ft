#!/usr/bin/env bash
# Runs ON the Colab VM, from /content/app, launched in the background by colab_train.sh.
# Usage: remote_job.sh smoke|baseline|full RUN_DIR CONFIG
set -euo pipefail

MODE=$1
RUN=$2
CONFIG=$3
mkdir -p "$RUN"

echo "[setup] installing dependencies"
pip install -q -r requirements-colab.txt
# Colab ships torchao 0.10; peft >= 0.21 refuses to wrap a model while an old torchao is
# importable. Nothing here quantizes, so remove it rather than chase a torch-matched build.
pip uninstall -y -q torchao
pip install -q --no-deps -e .
echo "[setup] done"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch, transformers, peft; print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)'

case "$MODE" in
  smoke)
    # Tiny end-to-end pass: data -> LoRA train -> save -> reload adapter -> generate -> score.
    python -c 'from huggingface_hub import whoami; print("HF user:", whoami()["name"])'
    python -m invoice_vlm.train --config "$CONFIG" --output-dir "$RUN/model" --limit 16 --max-steps 10 --no-push
    python -m invoice_vlm.evaluate --config "$CONFIG" --adapter "$RUN/model" --limit 4 --out "$RUN"
    ;;
  baseline)
    # Zero-shot base model on the full test split: the number fine-tuning has to beat.
    python -m invoice_vlm.evaluate --config "$CONFIG" --out "$RUN"
    ;;
  full)
    python -m invoice_vlm.train --config "$CONFIG" --output-dir "$RUN/model"
    python -m invoice_vlm.evaluate --config "$CONFIG" --adapter "$RUN/model" --out "$RUN"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
nvidia-smi --query-gpu=memory.used --format=csv,noheader

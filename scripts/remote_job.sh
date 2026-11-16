#!/usr/bin/env bash
# Runs ON the Colab VM, from /content/app, launched in the background by colab_train.sh.
# Usage: remote_job.sh smoke|baseline|full|demo RUN_DIR CONFIG
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
if [[ "${FAST_KERNELS:-0}" == 1 ]]; then
  # Qwen3.5's linear-attention layers otherwise fall back to slow reference PyTorch code.
  # No [cuda] extra, so pip leaves Colab's torch alone. causal-conv1d has no wheel for this
  # torch/python, so it compiles; a failed or slow build only costs that one kernel.
  pip install -q flash-linear-attention
  timeout 1200 pip install -q --no-build-isolation causal-conv1d \
    || echo "[setup] causal-conv1d build failed or timed out; continuing without it"
  python -c 'import fla; print("[setup] fla", fla.__version__)'
  python -c 'import causal_conv1d; print("[setup] causal_conv1d", causal_conv1d.__version__)' || true
fi
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
  demo)
    # The published adapter used the way a reader of the README would: Hub repo id + image file.
    ADAPTER=$(python -c 'import sys; from invoice_vlm.config import load_config; print(load_config(sys.argv[1])["hub"]["model_id"])' "$CONFIG")
    python - "$RUN" <<'PY'
import json, sys
from invoice_vlm.data import load_split
run = sys.argv[1]
for row in load_split("test", limit=1):  # first test document of each source
    row["image"].convert("RGB").save(f"{run}/{row['doc_type']}.png")
    with open(f"{run}/{row['doc_type']}.gold.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(row["target"]), f, ensure_ascii=False, indent=2)
PY
    for t in receipt invoice; do
      echo "[demo] $t: gold label"
      cat "$RUN/$t.gold.json"
      echo
      echo "[demo] $t: model output (adapter $ADAPTER)"
      start=$SECONDS
      python -m invoice_vlm.infer --config "$CONFIG" --adapter "$ADAPTER" --type "$t" "$RUN/$t.png" \
        | tee "$RUN/$t.pred.json"
      echo "[demo] $t: $((SECONDS - start)) s including model load"
    done
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
nvidia-smi --query-gpu=memory.used --format=csv,noheader

#!/usr/bin/env bash
# Run a job on a fresh Colab GPU VM via the `colab` CLI, then always release the VM.
#
#   scripts/colab_train.sh smoke      # ~10 steps on 32 examples: checks versions, VRAM, adapter reload
#   scripts/colab_train.sh baseline   # zero-shot base model on the test split
#   scripts/colab_train.sh full       # full LoRA training + test evaluation, adapter pushed to the HF Hub
#
# Env: GPU (default L4), CONFIG, POLL (seconds between status checks, default 60).
# The VM runs the committed HEAD (git archive), so commit before running.
set -euo pipefail

MODE=${1:?usage: colab_train.sh smoke|baseline|full}
case "$MODE" in smoke | baseline | full) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac
GPU=${GPU:-L4}
CONFIG=${CONFIG:-configs/qwen35_2b_lora.yaml}
POLL=${POLL:-60}

cd "$(git rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is dirty; commit first (the VM runs HEAD)" >&2
  exit 1
fi

TOKEN_FILE=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
[[ -s "$TOKEN_FILE" ]] || { echo "no HF token at $TOKEN_FILE; run: hf auth login" >&2; exit 1; }

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_ID="$MODE-$STAMP-$(git rev-parse --short HEAD)"
SESSION="ft-$MODE-${STAMP#*-}"
OUT="outputs/$RUN_ID"
mkdir -p "$OUT" dist
git archive --format=tar.gz -o dist/src.tar.gz HEAD

log() { printf '[local %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
# Pipe Python into the session's kernel. --timeout 0: setup/pip can be silent for minutes.
remote() { colab exec -s "$SESSION" --timeout 0; }

cleanup() {
  log "stopping session $SESSION"
  colab stop -s "$SESSION" </dev/null || true
  colab usage </dev/null >"$OUT/usage-after.txt" 2>&1 || true
}

colab usage </dev/null >"$OUT/usage-before.txt" 2>&1 || true
trap cleanup EXIT # set before `new`, so a half-created session is also released
log "creating session $SESSION on $GPU (run $RUN_ID)"
colab new -s "$SESSION" --gpu "$GPU" </dev/null

colab upload -s "$SESSION" dist/src.tar.gz /content/src.tar.gz </dev/null
# Uploaded as a file and read via HF_TOKEN_PATH, so the token never appears in
# `colab exec --env` arguments or the CLI's local session history.
colab upload -s "$SESSION" "$TOKEN_FILE" /content/.hf_token </dev/null

log "installing dependencies"
remote <<'PY' | tee "$OUT/setup.log"
import subprocess
r = subprocess.run(
    "set -e; mkdir -p /content/app && tar -xzf /content/src.tar.gz -C /content/app"
    " && chmod 600 /content/.hf_token"
    " && pip install -q -r /content/app/requirements-colab.txt"
    # Colab ships torchao 0.10; peft >= 0.21 refuses to wrap a model while an old torchao is
    # importable. Nothing here quantizes, so remove it rather than chase a torch-matched build.
    " && pip uninstall -y -q torchao"
    " && pip install -q --no-deps -e /content/app",
    shell=True, capture_output=True, text=True,
)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
print("SETUP_OK" if r.returncode == 0 else f"SETUP_FAILED rc={r.returncode}")
PY
grep -q SETUP_OK "$OUT/setup.log" || { log "setup failed, see $OUT/setup.log"; exit 1; }

log "launching $MODE job in the background on the VM"
remote <<PY
import os, subprocess
env = dict(os.environ, HF_TOKEN_PATH="/content/.hf_token", PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false")
cmd = "cd /content/app && bash scripts/remote_job.sh $MODE /content/run $CONFIG > /content/run.log 2>&1; echo \$? > /content/run.exit"
subprocess.Popen(["bash", "-c", cmd], env=env, start_new_session=True)
print("LAUNCHED")
PY

STATUS_PY='
import os
lines = open("/content/run.log", errors="replace").read().splitlines() if os.path.exists("/content/run.log") else []
print("\n".join(l[-200:] for l in lines[-3:] if l.strip()))
print("EXIT=" + open("/content/run.exit").read().strip() if os.path.exists("/content/run.exit") else "RUNNING")
'
EXIT_CODE=""
while [[ -z "$EXIT_CODE" ]]; do
  sleep "$POLL"
  # Hard time limit: a single hung `colab exec` once froze this loop for 50 minutes while the
  # finished VM sat idle and billing. A timed-out check is simply retried on the next poll.
  if ! status=$(printf '%s\n' "$STATUS_PY" \
      | timeout 120 colab exec -s "$SESSION" --timeout 60 2>&1); then
    log "status check failed (will retry): ${status: -200}"
    continue
  fi
  printf '%s\n' "$status" | grep -v '^EXIT=\|^RUNNING$' | sed 's/^/  vm| /' >&2 || true
  EXIT_CODE=$(printf '%s\n' "$status" | sed -n 's/^EXIT=//p' | tail -1)
done
log "remote job finished with exit code $EXIT_CODE"

for f in run.log run/metrics.json run/predictions.jsonl; do
  colab download -s "$SESSION" "/content/$f" "$OUT/$(basename "$f")" </dev/null >/dev/null 2>&1 \
    || log "not downloaded: /content/$f"
done
[[ -f "$OUT/metrics.json" ]] && cat "$OUT/metrics.json"
log "artifacts in $OUT"
exit "$EXIT_CODE"

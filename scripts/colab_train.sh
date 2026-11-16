#!/usr/bin/env bash
# Run a job on a fresh Colab GPU VM via the `colab` CLI, then always release the VM.
#
#   scripts/colab_train.sh smoke      # ~10 steps on 32 examples: checks versions, VRAM, adapter reload
#   scripts/colab_train.sh baseline   # zero-shot base model on the test split
#   scripts/colab_train.sh full       # full LoRA training + test evaluation, adapter pushed to the HF Hub
#
# Env: GPU (default L4), CONFIG, POLL (seconds between status checks, default 60),
#      MAX_HOURS (default 4).
# The VM runs the committed HEAD (git archive), so commit before running.
set -euo pipefail

MODE=${1:?usage: colab_train.sh smoke|baseline|full}
case "$MODE" in smoke | baseline | full) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac
GPU=${GPU:-L4}
CONFIG=${CONFIG:-configs/qwen35_2b_lora.yaml}
POLL=${POLL:-60}
MAX_HOURS=${MAX_HOURS:-4} # hard wall-clock cap for the remote job; the VM is stopped after it
NEW_RETRIES=${NEW_RETRIES:-5} # attempts to get a GPU when Colab has none free (503)

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
# Every colab call is wrapped in `timeout`: the CLI can hang when its connection drops.
# Pipe short Python snippets into the session's kernel (stdin = the snippet).
remote() { timeout 120 colab exec -s "$SESSION" --timeout 60; }

cleanup() {
  log "stopping session $SESSION"
  for _ in 1 2 3; do
    timeout 180 colab stop -s "$SESSION" </dev/null && break
    log "colab stop failed or timed out; retrying"
  done
  if timeout 60 colab sessions </dev/null 2>&1 | grep -q "\[$SESSION\]"; then
    log "WARNING: $SESSION is STILL RUNNING and billing. Stop it: colab stop -s $SESSION"
  fi
  timeout 60 colab usage </dev/null >"$OUT/usage-after.txt" 2>&1 || true
}

timeout 60 colab usage </dev/null >"$OUT/usage-before.txt" 2>&1 || true
trap cleanup EXIT # set before `new`, so a half-created session is also released
log "creating session $SESSION on $GPU (run $RUN_ID)"
# Colab answers 503 "Service Unavailable" when it has no free GPU of that type; that is
# transient, so retry. Nothing is billed until an assignment succeeds.
for attempt in $(seq 1 "$NEW_RETRIES"); do
  timeout 600 colab new -s "$SESSION" --gpu "$GPU" </dev/null >"$OUT/new.log" 2>&1 && break
  log "colab new failed (attempt $attempt/$NEW_RETRIES): $(grep -o 'Service Unavailable\|[A-Za-z]*Error[^│]*' "$OUT/new.log" | tail -1)"
  ((attempt < NEW_RETRIES)) || { log "no $GPU available; giving up"; exit 1; }
  sleep 120
done
# The CU/hour rate is only visible while the VM is assigned, so record it now.
timeout 60 colab usage </dev/null >"$OUT/usage-during.txt" 2>&1 || true
log "$(grep -i 'rate' "$OUT/usage-during.txt" || echo 'usage rate unknown') on $GPU"

timeout 300 colab upload -s "$SESSION" dist/src.tar.gz /content/src.tar.gz </dev/null
# Uploaded as a file and read via HF_TOKEN_PATH, so the token never appears in
# `colab exec --env` arguments or the CLI's local session history.
timeout 120 colab upload -s "$SESSION" "$TOKEN_FILE" /content/.hf_token </dev/null

# Everything slow (pip install, training, evaluation) runs in ONE background process on the
# VM, started by this short exec. A `colab exec` whose websocket drops can hang forever
# (seen during a 25-minute foreground pip install), so no long work is done in the foreground.
log "launching $MODE job (setup + run) in the background on the VM"
launch=$(remote <<PY 2>&1
import os, subprocess
env = dict(os.environ, HF_TOKEN_PATH="/content/.hf_token", PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false")
cmd = ("mkdir -p /content/app && tar -xzf /content/src.tar.gz -C /content/app"
       " && chmod 600 /content/.hf_token && cd /content/app"
       " && bash scripts/remote_job.sh $MODE /content/run $CONFIG > /content/run.log 2>&1;"
       " echo \$? > /content/run.exit")
subprocess.Popen(["bash", "-c", cmd], env=env, start_new_session=True)
print("LAUNCHED")
PY
) || true
grep -q LAUNCHED <<<"$launch" || { log "launch failed: ${launch: -300}"; exit 1; }

STATUS_PY='
import os
lines = open("/content/run.log", errors="replace").read().splitlines() if os.path.exists("/content/run.log") else []
print("\n".join(l[-200:] for l in lines[-3:] if l.strip()))
print("EXIT=" + open("/content/run.exit").read().strip() if os.path.exists("/content/run.exit") else "RUNNING")
'
EXIT_CODE=""
FAILS=0
DEADLINE=$((SECONDS + MAX_HOURS * 3600))
while [[ -z "$EXIT_CODE" ]]; do
  sleep "$POLL"
  if ((SECONDS > DEADLINE)); then
    log "job exceeded MAX_HOURS=$MAX_HOURS; giving up (checkpoints already on the Hub survive)"
    EXIT_CODE=124
    break
  fi
  # remote() is time-limited, so a hung check is just retried on the next poll.
  if ! status=$(printf '%s\n' "$STATUS_PY" | remote 2>&1); then
    FAILS=$((FAILS + 1))
    log "status check failed ($FAILS in a row): ${status: -200}"
    ((FAILS < 20)) || { log "VM unreachable; giving up"; EXIT_CODE=1; break; }
    continue
  fi
  FAILS=0
  printf '%s\n' "$status" | grep -v '^EXIT=\|^RUNNING$' | sed 's/^/  vm| /' >&2 || true
  EXIT_CODE=$(printf '%s\n' "$status" | sed -n 's/^EXIT=//p' | tail -1)
done
log "remote job finished with exit code $EXIT_CODE"

for f in run.log run/metrics.json run/predictions.jsonl; do
  timeout 300 colab download -s "$SESSION" "/content/$f" "$OUT/$(basename "$f")" \
    </dev/null >/dev/null 2>&1 || log "not downloaded: /content/$f"
done
[[ -f "$OUT/metrics.json" ]] && cat "$OUT/metrics.json"
log "artifacts in $OUT"
exit "$EXIT_CODE"

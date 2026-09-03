#!/bin/bash
# v2 Q1b queue: final two LOCO runs (gdph, then sysucc), strictly sequential.
# Each cohort is an independent && chain: a gdph failure does NOT block
# sysucc, but stages never interleave. One caffeinate wraps the whole queue
# (launched as: caffeinate -dims bash scripts/v2_loco_queue.sh).
set -u
cd "$(dirname "$0")/.."

run_cohort() {
  local c="$1"
  echo "=== $c: train started $(date '+%F %T') ==="
  .venv/bin/python src/v2_loco.py --hold-out "$c" --train \
      > "logs/v2_loco_${c}.log" 2>&1 \
  && .venv/bin/python src/v2_loco.py --hold-out "$c" --finalize \
      > "logs/v2_loco_${c}_finalize.log" 2>&1 \
  && .venv/bin/python src/v2_loco.py --hold-out "$c" --evaluate --confirm \
      > "logs/v2_loco_${c}_evaluate.log" 2>&1
  echo "=== $c: finished exit=$? $(date '+%F %T') ==="
}

run_cohort gdph
run_cohort sysucc
echo "=== queue done $(date '+%F %T') ==="

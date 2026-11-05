set -u
PY=/Users/yingwei/Documents/code/lora/.venv/bin/python
ROOT="/Users/yingwei/Documents/code/agent/test-RSI"
cd "$ROOT"
log(){ echo "[$(date +%Y-%m-%d_%H:%M:%S)] $*"; }
run_arm(){
  local ARM=$1
  log "start $ARM"
  $PY -m rsi.orchestrator --arm "$ARM" --gens 5 --gen-size small --eval-n 100 --eval-attr-n 100 --reps 1 --task-bank "$ROOT/data/frozen_academic_v1" --arms-root "$ROOT/arms_fixed" --probe-n 20 --call-timeout 90 --task-timeout 180 --phase-timeout 10800
  log "end $ARM exit=$?"
}
run_arm H  >> logs/run_arm_H.log 2>&1
run_arm W  >> logs/run_arm_W.log 2>&1
run_arm WH >> logs/run_arm_WH.log 2>&1
log ALL_DONE

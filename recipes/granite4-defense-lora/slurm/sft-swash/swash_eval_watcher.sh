#!/bin/bash
# Start the defense evals for each swash checkpoint as soon as its save completes.
# Log: ~/swash_eval_watcher.log. Builds started: ~/swash_eval_builds.txt (step build_id).
B=${SWASH_BUILD:-ade7bec1-f9bf-4022-b757-1d295b176def}
EXP=${SWASH_EXP:-sft_swash_3b_v2_odm_90_10_entropy_xml}; TAG=${SWASH_TAG:-swash_3b_ade7bec1}
R=/proj/granite-build/g4os/defense/runs/$B/$EXP
BUILDS=$HOME/${TAG}_eval_builds.txt
# Where the evals run: gpu-mid (default) or gpu-low (QOS=burst, preemptible, no GPU cap).
QUEUE=${SWASH_QUEUE:-gpu-mid}
case "$QUEUE" in gpu-low) QOS=${SWASH_QOS-burst}; TLIMIT=${SWASH_TIME_LIMIT:-1-00:00:00};;
                 *)       QOS=${SWASH_QOS-};      TLIMIT=${SWASH_TIME_LIMIT:-7-00:00:00};; esac
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -i $HOME/granite_build_id_ed25519 granitebuild@login1.bluevela.rmf.ibm.com"
GB="env GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH $HOME/granite.build/.venv/bin/gb"
log() { echo "$(date '+%F %T') $*"; }
cd "$HOME/granite.build"
for N in ${SWASH_STEPS:-1000 2000 3000 4000 5000}; do
  grep -q "^$N " "$BUILDS" 2>/dev/null && { log "step $N: already started, skipping"; continue; }
  log "step $N: waiting for checkpoint-$N"
  while true; do
    # saved = the trainer logged the save (any attempt's log) and the weights + state are on disk
    if $SSH "tr '\r' '\n' < <(cat $R/train_rank0*.log) | grep -aqE 'checkpoint saved in $R/checkpoint-$N\$' && { test -s $R/checkpoint-$N/model.safetensors || test -s $R/checkpoint-$N/model.safetensors.index.json; } && test -s $R/checkpoint-$N/trainer_state.json" 2>/dev/null; then
      break
    fi
    st=$($GB build status $B --skip-version-check 2>/dev/null | grep -oE "Status: [^ ]+ [A-Z]+" | head -1 | awk '{print $NF}')
    case "$st" in FAILED|CANCELLED) log "training build $B is $st; stopping (checkpoint-$N never landed)"; exit 1;; esac
    sleep 300
  done
  log "step $N: checkpoint-$N saved; starting evals"
  id=$($GB build start -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
      --param MODEL_PATH=$R/checkpoint-$N \
      --param EVAL_IMAGE=docker:/proj/granite-build/g4os/enroot/vllm-swash-nightly-defense-eval.sqsh \
      --param NAME=eval_${TAG}_$N \
      --param QUEUE=$QUEUE --param QOS=$QOS --param TIME_LIMIT=$TLIMIT 2>&1 \
      | grep -oE "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" | head -1)
  if [ -n "$id" ]; then echo "$N $id" >> "$BUILDS"; log "step $N: eval build $id"
  else log "step $N: FAILED to start eval build"; fi
done
log "all checkpoints handled"

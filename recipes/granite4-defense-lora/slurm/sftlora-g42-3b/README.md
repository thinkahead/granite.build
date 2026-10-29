# sftlora-g42-3b (BlueVela SLURM)

granite-4.2-3b LoRA SFT with ODM entropy sampling on the 12 defense XML/tool datasets,
on BlueVela SLURM. A port of the Vela build
`defence/granite_dot_build_enhanced/sftlora_g42_3b_v2_odm_90_10_entropy_xml_enhanced.yaml`
(build `224c09b0`) using the [`tuning-slurm`](../../../../configurations/assets/environments/skypilot/steps/tuning-slurm/README.md)
step. The build.yaml header lists what changed against the Vela build.

## Run

```bash
cd ~/granite.build
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/sftlora-g42-3b/build.yaml --skip-version-check
```

Defaults: 1 × 8 H200 on `gpu-high` (QOS `grp_granite_dot_build_high_1n`, 2-day limit).
For gpu-mid: `--param QUEUE=gpu-mid --param QOS="" --param TIME_LIMIT=7-00:00:00`.
Quick plumbing check: `--param MAX_STEPS=50 --param EVAL_STEPS=50 --param ODM_UPDATE_INTERVAL=50 --param SAVE_STEPS=50`.

### granite-4.2-30b (2 nodes, gpu-mid)

Same recipe, other model; 2 × 8 H200 on gpu-mid (gpu-high's QOS caps at 8 GPUs = 1 node):

```bash
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/sftlora-g42-3b/build.yaml --skip-version-check \
  --param MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm \
  --param EXPERIMENT=sftlora_g42_30b_v2_odm_90_10_entropy_xml \
  --param NUM_NODES=2 --param QUEUE=gpu-mid --param QOS= --param TIME_LIMIT=7-00:00:00 \
  --param PER_DEVICE_TRAIN_BATCH_SIZE=5 --param GRADIENT_ACCUMULATION_STEPS=1 \
  --param EVAL_STEPS=1000 --param ODM_UPDATE_INTERVAL=1000 \
  --param SAVE_STEPS=250 --param MAX_RETRIES=3
```

- **Batch.** Effective = per-device × grad-accum × nodes × 8: 5 × 1 × 2 × 8 = 80 (Vela's 30b was
  1 × 1 × 4 × 8 = 32). Batch 5 at 8192 tokens peaks at ~97–121 GB of 141 GB per H200.
- **ODM cost dominates on 30b.** One eval + ODM entropy pass is ~73 min (5 min eval, ~68 min reward
  pass), vs ~26 min on 3b, and doesn't shrink with train batch (eval batch stays 1). Keep
  `EVAL_STEPS` = `ODM_UPDATE_INTERVAL` (they must stay aligned) and raise both: 1000 → 5 passes
  (~6 h) over 5000 steps, vs 20 passes (~24 h) at 250.
- **Speed.** ~9.4 s/step at batch 5 on 2 nodes (7.7 s/step at batch 3).
- **Model dir.** `teacher_overlays/granite-4.2-30b-slurm` holds real bf16 weights on `/proj`; its
  `chat_template.jinja` defaults thinking off (original kept as `.bak_thinking`).

### granite-4.2-8b (2 nodes, gpu-mid)

```bash
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/sftlora-g42-3b/build.yaml --skip-version-check \
  --param MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.2-8b \
  --param EXPERIMENT=sftlora_g42_8b_v2_odm_90_10_entropy_xml \
  --param NUM_NODES=2 --param QUEUE=gpu-mid --param QOS= --param TIME_LIMIT=7-00:00:00 \
  --param PER_DEVICE_TRAIN_BATCH_SIZE=5 --param GRADIENT_ACCUMULATION_STEPS=1 --param MAX_STEPS=5000 \
  --param EVAL_STEPS=500 --param ODM_UPDATE_INTERVAL=500 \
  --param SAVE_STEPS=250 --param MAX_RETRIES=3
```

- **Batch.** 5 × 1 × 2 × 8 = 80, same as the 30b.
- **Speed.** ~3.6 s/step averaged over the first 2650 steps, including eval + ODM passes every 500 steps.
- **Preemption.** `SAVE_STEPS=250` + `MAX_RETRIES=3`: a preempted run resumes from the last checkpoint
  (same for the 30b).

### Logs

Everything is on `/proj` (readable from any BlueVela login node). With `B=<build_id>`,
`EXP=<EXPERIMENT>` (e.g. `sftlora_g42_30b_v2_odm_90_10_entropy_xml`) and `N` = node rank
(0 … NUM_NODES-1):

| Log | Path | What |
|---|---|---|
| **Training, per node** | `/proj/granite-build/g4os/defense/runs/$B/$EXP/train_rank<N>.log` | Full accelerate stdout+stderr of node N (all 8 of its ranks) for the **current attempt**: loss lines, eval, ODM, tracebacks. Mostly tqdm redraws, so use `tail -F` / `grep` rather than `less` |
| Earlier attempts, per node | `…/$EXP/train_rank<N>.attempt<k>.log` | The same log from attempt k (k = 1, 2, … in order) of a retried run |
| Loss / eval history | `…/$EXP/training_logs.jsonl` | One JSON row per logged step (`training_loss`, step, timestamp). Rank 0 only |
| Trainer config + exact launch command | `/proj/granite-build/g4os/gbtest/builds/builds/$B/runs/<targetrun_id>/tuning-config-rank<N>/` | `launch.sh`, `tuning_data_config.yaml`, `trainer_controller_config.yaml`, `step_config.json` |
| Whole-step output (all nodes, SkyPilot) | `/proj/granite-build/g4os/.sky_clusters/<cluster>/sky_logs/1-<task>/run.log` | Setup + both nodes interleaved, each line prefixed `(head, rank=0 …)` / `(worker1, rank=1 …)` |
| sbatch script + its stdout | `/proj/granite-build/g4os/.sky_provision/<cluster>.sh`, `slurm-<jobid>.out` | Container start-up; look here if the job never reaches the trainer |
| gbserver (build orchestration) | `~/gbserver-standalone/gbserver.log` on the gbserver host | Launches, retries, artifact registration, build status |

The cluster name is `gb-<build name>-<target>-<launch id>-…`; find it, and the SLURM job id, with
`squeue --me -o "%i %T %j"` on a login node.

```bash
B=e8ff9ccf-33e4-4ddf-980f-a2369ebbd0cb; EXP=sftlora_g42_30b_v2_odm_90_10_entropy_xml
O=/proj/granite-build/g4os/defense/runs/$B/$EXP
tail -F $O/train_rank0.log                                  # follow node 0 (survives retries)
grep -a "Epoch: .*Step:" $O/train_rank0.log | tail          # latest losses
grep -a -E "Traceback|OutOfMemory" $O/train_rank*.log       # errors, any node and attempt
```

**Retries (preemption).** gpu-mid is preemptible (`PreemptMode=REQUEUE`). With `MAX_RETRIES` > 0 a
preempted run is relaunched into the same `$O`. Before writing, each node renames its previous
`train_rank<N>.log` to the next free `train_rank<N>.attempt<k>.log`, so every attempt's log is kept
and `train_rank<N>.log` is always the current one. The trainer resumes from the newest
`checkpoint-<N>` (look for `fast-forwarding your dataset by <N> steps` near the top of the new
log). Follow with `tail -F` (capital F), which reopens the file when an attempt starts a new one.

Outputs land in `/proj/granite-build/g4os/defense/runs/<build_id>/sftlora_g42_3b_v2_odm_90_10_entropy_xml/`:
`final/` (the LoRA adapter; merge with the base before vLLM eval), `checkpoint-<N>/`
(adapter + optimizer state, every `SAVE_STEPS`), `training_logs.jsonl`, `train_rank0.log`.

## Requirements that are easy to miss

- **64 CPUs reaching the trainer.** `GPU_CPUS: 64` sets the allocation, but SkyPilot's
  SLURM run step defaults to `--cpus-per-task=1`, which pins all 8 ranks and the ODM
  reward pass to one core: 41 s/step instead of 2.32 s/step, with no error. Needs the
  SkyPilot patch described in the step README. Verify on a running job:
  `scontrol show step <jobid> | grep -A1 Name=sky-` must show `CPUs=64`.
- **FastAPI 0.140.0** in the gbserver venv when using `thinkahead/skypilot@docker-host-ami-override`;
  newer FastAPI crashes that branch's SkyPilot API server. See the step README.

**ODM `vocab_size`:** the entropy reward's normalisation is patched from 32000 to 100352, which is the output-layer size of every model these recipes train (4.2 3b/8b/30b, swash-3b, swash-2b, granite-5.1-20b). Per-model table and how to check a new model: [tuning-slurm step README](../../../../configurations/assets/environments/skypilot/steps/tuning-slurm/README.md#odm-entropy-reward-vocab_size).

## Validated runs

| Build | Steps | Result |
|---|---|---|
| job 299620 (direct `sbatch`, no granite.build; see [`smoke/`](smoke/README.md)) | 20 | adapter written, 2.3 s/step |
| `26ae041d` | 50 | **SUCCESS**: 2.32 s/step, `train_runtime` 1695 s, 3 outputs registered |
| `cd2bfbc2` | 5000 | **SUCCESS**: 2.08–2.13 s/step, `train_runtime` 41321 s (11h29m), checkpoints 1000–5000 + `final/` |
| `f2324fdf` (30b, 2 nodes, batch 3) | 50 | **SUCCESS**: 7.7 s/step, `train_runtime` 5183 s (incl. one ~73 min eval+ODM pass) |

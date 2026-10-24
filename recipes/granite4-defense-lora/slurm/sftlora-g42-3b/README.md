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
  --param EVAL_STEPS=1000 --param ODM_UPDATE_INTERVAL=1000 --param SAVE_STEPS=1000
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

### Logs (one set per node)

In the run's output dir, for each node rank N: `train_rank<N>.log` (raw accelerate output of that
node) and `train_rank<N>.clean.log` (the same with progress-bar redraws collapsed, written live,
~2% of the raw size). `tail -f train_rank0.clean.log` to follow a run.

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

## Validated runs

| Build | Steps | Result |
|---|---|---|
| job 299620 (direct `sbatch`, no granite.build; see [`smoke/`](smoke/README.md)) | 20 | adapter written, 2.3 s/step |
| `26ae041d` | 50 | **SUCCESS**: 2.32 s/step, `train_runtime` 1695 s, 3 outputs registered |
| `cd2bfbc2` | 5000 | **SUCCESS**: 2.08–2.13 s/step, `train_runtime` 41321 s (11h29m), checkpoints 1000–5000 + `final/` |
| `f2324fdf` (30b, 2 nodes, batch 3) | 50 | **SUCCESS**: 7.7 s/step, `train_runtime` 5183 s (incl. one ~73 min eval+ODM pass) |

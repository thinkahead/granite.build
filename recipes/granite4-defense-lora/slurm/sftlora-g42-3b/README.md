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
| `625d7bcb` | 50 | cancelled: run step had 1 CPU, 41 s/step |
| `26ae041d` | 50 | **SUCCESS**: 2.32 s/step, `train_runtime` 1695 s, 3 outputs registered |

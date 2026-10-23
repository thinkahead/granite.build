# Direct SLURM smoke test (no granite.build, no SkyPilot)

A hand-submitted `sbatch` job that runs the **exact trainer command** the
`tuning-slurm` step renders for this recipe, cut to 20 steps. It validates the
image, the in-container patches, the `/proj` data paths and LoRA + FSDP + ODM on
H200, independent of gbserver / SkyPilot provisioning.

It mirrors what SkyPilot's SLURM backend does for the step: one pyxis/enroot
container per node created with `--container-remap-root --container-writable`,
with the SkyPilot workdir (`/proj/granite-build/g4os`) bind-mounted.

First run: job **299620**, 2026-10-22, 1 × 8 H200 (`p1-r25-n4`).

## Files

| File | What it is |
|---|---|
| `job.sh` | The sbatch script (below). |
| `setup.sh` | The recipe's `setup_commands` (filelock + ODM patches), one per line under `set -eu`. |
| `launch.sh` | The `accelerate launch -m tuning.sft_trainer ...` command, rendered by `../../../../../configurations/assets/environments/skypilot/steps/tuning-slurm/src/render_tuning_cmd.py` from the recipe config with `max_steps=20`, `eval_steps=10`, `save_steps=20`, ODM `update_interval=10`. |
| `tuning_data_config.yaml` | The recipe's 12 datasets on `/proj` (90/10 split), ODM entropy. |
| `trainer_controller_config.yaml` | The step's default controller (prints loss and the checkpoint-saved lines). |

`launch.sh`, `tuning_data_config.yaml` and `trainer_controller_config.yaml` hardcode
`D=/proj/granite-build/g4os/defense/smoke-tuning-slurm` as the run directory.

## The sbatch script

```bash
#!/bin/bash
#SBATCH -J smoke-tuning-slurm
#SBATCH -p gpu-mid
#SBATCH --account=grp_granite_dot_build
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=256G
#SBATCH --gres=gpu:8 --constraint=h200
#SBATCH --exclude=p1-r18-n1,p1-r10-n1
#SBATCH -t 1:30:00
#SBATCH -o /proj/granite-build/g4os/defense/smoke-tuning-slurm/slurm-%j.out
srun --container-image=/proj/granite-build/g4os/enroot/us.icr.io-cil15-shared-registry-fms-hf-tuning-6mar2026-a66092d5c01.sqsh \
  --container-mounts=/proj/granite-build/g4os:/proj/granite-build/g4os \
  --container-remap-root --container-writable --no-container-mount-home \
  bash -c 'export HF_HOME=/tmp/hf_home NCCL_TIMEOUT=10800000; cd /proj/granite-build/g4os/defense/smoke-tuning-slurm && bash setup.sh && bash launch.sh'
```

Why each flag:
- `--cpus-per-task=64` — BlueVela H200 nodes are `gpu:h200:8(S:0-1)`; a task pinned
  to one socket sees only 4 GPUs and an 8-rank launch dies (`cuda:7 out of range`).
- `--constraint=h200` + `--exclude=...` — same as the SLURM environment
  (`configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml`).
- `--container-writable` — the squashfs is read-only otherwise, and `setup.sh`
  patches `site-packages` in place (`sed -i` fails with "Read-only file system").
- `--container-remap-root` — same as SkyPilot; the patches write root-owned files.
- `HF_HOME=/tmp/hf_home` — node-local datasets cache, so no GPFS file locking.
- The scheduler moved this job from `gpu-mid` to `gpu-high` with QOS
  `grp_granite_dot_build_high_1n` (visible in `scontrol show job`). That's account
  routing, not something the script asks for.

## Run it

From a machine with the BlueVela key:

```bash
D=/proj/granite-build/g4os/defense/smoke-tuning-slurm
BV="granitebuild@login1.bluevela.rmf.ibm.com"
ssh -i ~/granite_build_id_ed25519 $BV "rm -rf $D; mkdir -p $D"
scp -i ~/granite_build_id_ed25519 job.sh setup.sh launch.sh tuning_data_config.yaml trainer_controller_config.yaml $BV:$D/
ssh -i ~/granite_build_id_ed25519 $BV "cd $D && sbatch job.sh"
```

Watch:

```bash
ssh -i ~/granite_build_id_ed25519 $BV 'squeue --me -n smoke-tuning-slurm'
ssh -i ~/granite_build_id_ed25519 $BV "grep -a -E 'FILELOCK_PATCH|Epoch: .*Step:|checkpoint saved|train_runtime|Traceback' $D/slurm-*.out | tail"
```

Success looks like `FILELOCK_PATCH_APPLIED lockf=5 unlink=2`, `Epoch: 0, Step: N ... loss`
lines, `Final checkpoint saved in .../out/exp/final`, and `out/exp/final/` holding
`adapter_config.json` + `adapter_model.safetensors` (a LoRA adapter).

## Regenerating `launch.sh` for other settings

```bash
python3 configurations/assets/environments/skypilot/steps/tuning-slurm/src/render_tuning_cmd.py \
  --config-json step_config.json --python /usr/bin/python3 \
  --nodes 1 --gpus-per-node 8 --node-rank 0 --master-ip 127.0.0.1 --master-port 29500 \
  --output-dir $D/out/exp --save-model-dir $D/out/exp/final --experiment smoke \
  --model /proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.2-3b \
  --data-config $D/tuning_data_config.yaml --controller-config $D/trainer_controller_config.yaml \
  --out launch.sh
```

`step_config.json` is `{"tuning_config": ..., "fsdp_config": ..., "accelerate_config": ...}`
from the rendered recipe (`gb build start ... --dry-run --save-build-file`) merged
over the step defaults.

## Observed timing (job 299620)

- Model + 12 datasets load, then ~2.3 s/step for training (batch 5/GPU, 8192 ctx).
- **Eval dominates a smoke run.** Every `eval_steps` evaluates all 12 validation splits
  (~12 min), and every ODM `update_interval` re-scores the reward over the eval data
  (minutes per category; the router split alone is ~1,100 rows). With `eval_steps=10`
  and `update_interval=10` over 20 steps, that is two full passes of each. That was
  still in progress past 50 minutes, so allow the full 1:30 wall time, or raise
  `eval_steps`/`update_interval` to `max_steps` for a faster plumbing check.

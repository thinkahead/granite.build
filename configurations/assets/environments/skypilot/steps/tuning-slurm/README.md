# tuning-slurm (SkyPilot / SLURM)

fms-hf-tuning SFT / LoRA (`accelerate launch -m tuning.sft_trainer`) on BlueVela SLURM,
through the SkyPilot SLURM cloud. A port of the k8s/LSF `tuning` step: the config
surface is the same (`tuning_config`, `tuning_data_config`, `fsdp_config`,
`accelerate_config`, `compute_config`, `trainer_controller_config`), and flags are
rendered with the same conventions as that step's helm helpers, so a Vela build's step
config carries over unchanged. See the header of [`step.yaml`](step.yaml) for what differs
(topology, `setup_commands`, outputs on `/proj`, inline NCCL pins).

Recipe using it: `recipes/granite4-defense-lora/slurm/sftlora-g42-3b/`.

## CPU cores — required for ODM and padding-free

**Give each node 64 CPUs and make sure the trainer actually gets them.** On an 8-GPU
node fms-hf-tuning is CPU-hungry in a way GPU-bound trainers are not: eight ranks each
drive their GPU's kernel launches, padding-free packing assembles every batch on the
CPU, and ODM (`dataprocessor.type: odm`) adds a per-category reward pass over the
validation data at every `update_interval` plus per-sample category sampling. All of
it is Python on the host.

| Setting | Value | Why |
|---|---|---|
| `compute_config.num_cpus_per_node` | `64` | sbatch `--cpus-per-task`. 64 of 96 spans both sockets of the `gpu:h200:8(S:0-1)` nodes; a one-socket allocation sees only 4 GPUs. Matches the k8s step's default of 8 CPUs per GPU. |
| SkyPilot run-step CPUs | must equal the job's | See below. Check with `scontrol show step <jobid>`: the `Name=sky-N` step must show `CPUs=64`, not `CPUs=1`. |
| `HF_HOME` | node-local `/tmp/hf_home` (step default) | Keeps the datasets cache off GPFS. |

### The one-CPU trap (SkyPilot run step)

The sbatch allocation having 64 CPUs is **not enough**. SkyPilot starts the step's `run`
script with a second `srun` inside the allocation, and its SLURM codegen passes
`--cpus-per-task=<task CPU demand>`, SkyPilot's scheduling default of 0.5 rounded up to
**1**. SLURM pins that step's processes to exactly one core, so all 8 ranks, their
dataloaders and the ODM reward pass share core 0. Nothing fails; everything is slow.

Measured on the same recipe, same 1 × 8 H200 gpu-high node type:

| | run step `CPUs=1` (build 625d7bcb) | run step `CPUs=64` (build 26ae041d) |
|---|---|---|
| Trainer `Cpus_allowed_list` | `0` | `0-31,48-79` |
| Core 0 / other cores busy | 100% / 0% | spread |
| Train speed | 41 s/step | **2.32 s/step** |
| Eval speed | ~11.8 s/batch | ~0.4 s/batch (~2.5 it/s) |
| GPU utilisation | 26–67% | 83–90% |

This is a SkyPilot bug, not a step setting; a step cannot widen its own SLURM CPU
binding from inside. It is present in both the pinned `cmadam/skypilot@gb-sky-v1-stable`
and `thinkahead/skypilot@docker-host-ami-override` (`sky/backends/task_codegen.py`,
`SlurmCodeGen.add_task`, `run_flags = ... --cpus-per-task={task_cpu_demand} ...`).

**Fix used for these builds** (local, uncommitted, on `thinkahead/skypilot@docker-host-ami-override`
`4a084382`): the generated driver asks SLURM for the job's per-node CPU count and uses
that for the run step, falling back to the old value if `squeue` fails.

```python
                def _job_cpus_per_node():
                    # task_cpu_demand is SkyPilot's per-task scheduling demand (0.5 -> 1),
                    # not the job's CPU request, and slurm pins the run step's tasks to
                    # exactly --cpus-per-task cores. With the default every rank of a
                    # multi-GPU task shares a single core. Give the run step the CPUs
                    # the sbatch job was allocated per node instead (squeue %c).
                    try:
                        out = subprocess.run(
                            ['squeue', '-h', '-j', '{self._slurm_job_id}', '-o', '%c'],
                            capture_output=True, text=True, timeout=30, check=False)
                        return max({task_cpu_demand}, int(out.stdout.split()[0]))
                    except Exception:  # pylint: disable=broad-except
                        return {task_cpu_demand}

                def run_thread_func():
                    ...
                    run_flags = f'--nodes={num_nodes} --cpus-per-task={{_job_cpus_per_node()}} --mem=0 {{gpu_arg}} --exclusive'
```

After changing SkyPilot, restart both servers so the driver is regenerated:
`sky api stop`, then restart `gbserver standalone`.

### Does Vela do this?

Vela needs no fix. The k8s `tuning` step runs the trainer directly as the pod's
command, and the pod's CPU request is `compute_config.num_cpus_per_node`, defaulting to
8 × GPUs (64 for 8 GPUs) when unset. Build `224c09b0` leaves it unset, so it got 64.
There is no inner launcher step that re-binds the processes, so the trainer always had
all of the pod's CPUs. On SLURM the extra SkyPilot `srun` layer is what lost them.

## Gbserver host requirements (standalone, SkyPilot fork)

- **FastAPI pin with `thinkahead/skypilot@docker-host-ami-override`.** That branch's
  API server constructs `fastapi.FastAPI(prefix='/api/v1', ...)`, a keyword newer FastAPI
  rejects. With the FastAPI granite.build resolves (0.161.0) the SkyPilot API server dies
  at start (`TypeError: FastAPI.__init__() got an unexpected keyword argument 'prefix'`)
  and every launch fails with `SkyPilot API server process exited unexpectedly`.
  **0.140.0 works; 0.150.0 does not** (bisected by constructing the app). Install:

  ```bash
  cd ~/granite.build
  python3.13 -m venv .venv
  export PIP_USER=0            # the sandbox sets PIP_USER=1, which breaks venv installs
  .venv/bin/pip install -e '.[standalone]'
  .venv/bin/pip install -e '/home/karve/skypilot[kubernetes,slurm]'
  .venv/bin/pip install 'fastapi==0.140.0'
  .venv/bin/sky api start && curl -s http://127.0.0.1:46580/api/health && .venv/bin/sky api stop
  ```

  The pinned `cmadam/skypilot@gb-sky-v1-stable` was not tested against this.
- **rsync** on the gbserver host (SkyPilot uploads `file_mounts` with it). The sandbox
  had none and no root; `rsync` 3.2.5 + `libpopt.so.0` were copied from BlueVela login1
  into `~/.local/lib/rsync/` behind a `~/.local/bin/rsync` wrapper that sets
  `LD_LIBRARY_PATH`.
- **SSH** to `login0/1/2.bluevela.rmf.ibm.com` with `~/granite_build_id_ed25519`, host
  keys accepted (the environment picks one of the three at random per launch).

## Image

`us.icr.io/cil15-shared-registry/fms-hf-tuning:6mar2026-a66092d5c01` (the k8s step's
default), pre-staged as
`/proj/granite-build/g4os/enroot/us.icr.io-cil15-shared-registry-fms-hf-tuning-6mar2026-a66092d5c01.sqsh`
(re-stage with `srun -p cpu --qos=cpu --container-image=<ref> --container-save=<sqsh> true`).
Python 3.12 system install, fms-hf-tuning `ea8c6c737`, torch 2.7.0a0 (NGC 25.02),
transformers 4.55.4, peft 0.18.1, accelerate 1.13.0, fms-acceleration-odm 0.1.5. Its root
filesystem is read-only unless the container is created `--container-writable` (SkyPilot
does), which `setup_commands` that `sed -i` site-packages rely on.

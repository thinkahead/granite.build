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

## `setup_commands` abort the job when they fail

Each entry in `config.setup_commands` runs under `set -e` with an `ERR` trap, so any failing
command stops the job before training, logging `tuning-slurm: setup_commands[N] FAILED; aborting`
(each successful one logs `... OK`). The recipes patch site-packages and then `test`/`grep -q` the
result; without this, a patch that stopped matching after an image or package upgrade was skipped
silently, because the run script itself is not `set -e`.

## ODM entropy reward: `vocab_size`

fms-acceleration-odm's entropy reward divides each token's entropy by `log(vocab_size)` (the
maximum possible entropy) so the reward lands in [0, 1]. It hardcodes `vocab_size=32000` at every
`compute_reward` call site in `fms_acceleration_odm/odm/dataset.py` (2 in 0.1.5, 3 in 0.1.6). The
recipes `sed` it to **100352** and check it (`setup_commands`), then log
`ODM_VOCAB_PATCH_APPLIED <count> vocab_size=100352;` to the training log.

The right value is the size of the model's **output layer**: `config.vocab_size`, the number of rows
of `lm_head.weight` (or of `embed_tokens.weight` when embeddings are tied). It does not depend on
the chat template, and depends on the tokenizer only through the model: a tokenizer change without
resizing the model doesn't change it, and a padded output layer can be larger than the tokenizer.
Verified from each model's weights (2026-10-29):

| Model | `config.vocab_size` | Output-layer rows | Tokenizer tokens | Tied embeddings |
|---|---|---|---|---|
| granite-4.2-3b | 100352 | 100352 (`lm_head`) | 100352 | no |
| granite-4.2-8b | 100352 | 100352 (`lm_head`) | 100352 | no |
| granite-4.2-30b | 100352 | 100352 (`lm_head`) | 100352 | no |
| granite-swash-3b-a600m (`-g42tok`) | 100352 | 100352 (`embed_tokens`) | 100352 | yes |
| granite-swash-2b (`-g42tok`) | 100352 | 100352 (`embed_tokens`) | 100352 | yes |
| granite-5.1-20b-sft-preview-r261020a | 100352 | 100352 (`lm_head`) | 100352 | no |

So 100352 is correct for every model these recipes train. **A model with a different output size
needs a different value**: check `lm_head.weight`'s shape in its safetensors header (not just
`config.json`) and change both the `sed` and the `test` in the recipe.

## Batched ODM reward pass (opt-in)

With `padding_free`, fms-hf-tuning builds ODM's reward-pass collators unpadded too
(`DataCollatorForSeq2Seq(padding=False)`), so `per_device_eval_batch_size` must stay 1 and the
reward pass runs one example per forward. On granite-5.1-20b (8 × H200, FSDP) each ODM update
took ~70 min, ~23 h of a ~30 h run.

`src/odm_batched_reward.py` pads **only the reward-pass batches** (right padding, which leaves
every real token's logits and the masked entropy unchanged) and batches them; training stays
padding-free. It also takes `vocab_size` from the model's `config.vocab_size`. Turn it on with
`launcher_config.envs.ODM_REWARD_BATCH_SIZE: "<N>"` (N > 1); the step then loads it in every
rank via a `sitecustomize` hook and logs `tuning-slurm: batched ODM reward pass enabled` plus one
`ODM_BATCHED_REWARD rank=... eval_batch_size=N` line per rank. Unset or 1 = upstream behaviour.

Verified 2026-10-29 with `tests/odm_reward_fsdp.py` (2 × H200, FSDP FULL_SHARD per decoder
layer, bf16, flex attention, granite-5.1-20b `-eager`, 120 validation examples per category):

| Category | reward/ex bs1 | bs8 | \|diff\| | time bs1 | bs8 | speedup |
|---|---|---|---|---|---|---|
| inventory | 0.12936 | 0.12952 | 1.6e-4 | 30.6 s | 6.4 s | 4.8× |
| news | 0.08575 | 0.08595 | 2.1e-4 | 33.8 s | 14.3 s | 2.4× |
| analysis | 0.08713 | 0.08737 | 2.4e-4 | 33.3 s | 13.0 s | 2.6× |
| **total** | | | | 97.7 s | 33.6 s | **2.9×** |

Differences are bf16 rounding (~0.2% of the reward). Peak memory at bs8: 54 GB per GPU. The
speedup varies with how much example lengths differ within a batch (padding). On 8 GPUs each
forward also gathers more shards, which batching amortizes, so expect at least this much there.
**Memory: the entropy is reduced in chunks.** Upstream `compute_reward` builds several
`[batch, seq, vocab]` fp32 tensors at once (log_softmax, probs, p·log p, p·(log p)²). Next to the
granite-5.1-20b training state, a padded batch of 8 long examples OOM'd at step 500 (build
`32dae5df`, 16 GB allocation with 5 GB free). `odm_batched_reward.py` runs the model once per batch
and reduces the entropy terms over the vocabulary `ODM_REWARD_ENTROPY_CHUNK_TOKENS` positions at a
time (default 2048), in fp32. Single GPU, swash-2b, 8 × 3736-token examples: identical to upstream
to 1.6e-4 per example (upstream rounds its result to bf16), peak 27.5 GB vs 33.3 GB upstream and
59.9 GB unchunked. On the 20b (build `6b75f47f`, 25 steps, reward pass at step 20) all 12
categories fit and finished in ~38 min, ~25 of them in the one long category. A first attempt
that instead split long batches into token-capped sub-batches gave ranks different numbers of
forwards for the same category and hung FSDP at the end of the pass (build `67ba3cdd`); the
chunked version always runs one forward per batch on every rank.

`tests/odm_pad_equiv.py` is the single-GPU exactness check (swash-2b, 96 examples, bs1 vs right-
and left-padded bs8). `tests/odm_reward_fsdp.sbatch` reruns the FSDP test; it needs
`--container-writable` so the stale-torch-2.7 cleanup can delete files.

# sft-swash-3b (BlueVela SLURM)

`ibm-granite/granite-swash-3b-a600m` **full fine-tune** with ODM entropy sampling on the 12 defense
XML/tool datasets, on BlueVela SLURM. Same data, ODM settings and patches as
[`../sftlora-g42-3b`](../sftlora-g42-3b/README.md) (granite-4.2 LoRA), so the scores compare
directly. The build.yaml header lists what differs; the reasons are under
[What swash needed](#what-swash-needed).

Swash is `GraniteMoeSWAForCausalLM`: 3.02B parameters, about 600M active (48 experts, top-4, plus a
shared expert), sliding-window attention with learnable attention sinks, 8192-token context. It is
a **base** model, not an instruct model.

## Model directory

`/proj/granite-build/g4os/models/granite-swash-3b-a600m-g42tok` (the recipe's `MODEL_URI`):

| File | From |
|---|---|
| `model.safetensors` | symlink to `../granite-swash-3b-a600m/model.safetensors` (HF snapshot `568a94fc`, copied from `/proj/data-eng/tmp/hf_cache/.../models--ibm-granite--granite-swash-3b-a600m`; sha256 matches) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, `chat_template.jinja` | granite-4.2-3b (`kd-sandbox/model_copies/granite-4.2-3b`) |
| `config.json` | swash's, with `_experts_implementation: eager` and `eos_token_id`/`pad_token_id` = 100257 (`<\|im_end\|>`) |

Swash and granite-4.2 share the BPE vocab and merges, and 4.2's special tokens (`<|im_start|>`,
`<|im_end|>`, `<tool_call>`, `<think>`, …) sit in swash's unused special-token ids. With the 4.2
tokenizer the defense data tokenizes exactly as for the 4.2 runs (identical ids on 50/50 samples);
with swash's own tokenizer each marker would be several tokens (32 vs 77 on a sample). Same approach
as `kd-sandbox/README_SWASH.md` "Workaround F".

## Run the training

```bash
cd ~/granite.build
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/sft-swash-3b/build.yaml --skip-version-check \
  --param QUEUE=gpu-high --param QOS=grp_granite_dot_build_high_1n \
  --param NUM_NODES=1 --param TIME_LIMIT=2-00:00:00
```

That is the command for build `ade7bec1` (1 × 8 H200 on gpu-high). Settings from `parameters.yaml`:

| Setting | Value | Note |
|---|---|---|
| Trainer image | `fms-hf-tuning:6mar2026` (Ubuntu) | the image the 4.2 runs use; see below for why not v3.2.2 |
| `transformers` | 5.18.0, from `/proj/granite-build/g4os/pylibs/swash-transformers-5.18-6mar` | swash needs ≥ 5.15; the image has 4.55 |
| fms-hf-tuning | patched copy, `/proj/granite-build/g4os/pylibs/swash-fms-tuning-overlay-6mar` | 3 transformers-5 fixes, originals kept as `*.orig` |
| Attention | `flex_attention` | swash has no sdpa / FlashAttention-2 (sinks) |
| Fine-tune | full (no LoRA) | PEFT can't target the fused 3-D expert tensors |
| `LEARNING_RATE` | 1e-5 | 10× below the LoRA runs' 1e-4, usual for a full fine-tune |
| Batch | 5 per GPU × 8 = 40 | as the 3b LoRA run |
| `MAX_STEPS` / `EVAL_STEPS` = `ODM_UPDATE_INTERVAL` | 5000 / 250 | as the 3b LoRA run |
| `SAVE_STEPS` / `MAX_RETRIES` | 500 / 3 | a retry resumes from the latest checkpoint |

`fast_kernels` is off (fms-acceleration-foak imports a transformers-4 symbol).

**Before relaunching a failed build, cancel it** (`gb build cancel <id> --skip-version-check`), or its
retries keep submitting 8-GPU jobs; then check `squeue --me` for leftover `gb-sft-swash-*` jobs.

### granite-swash-2b (same recipe, other model)

`ibm-granite/granite-swash-2b` is **dense** (`GraniteSWAForCausalLM`, 2.14B, no experts; 24 layers, 7
full-attention + 17 sliding-window, sinks, 8192 context) and ships the same tokenizer as swash-3b.
Its model dir is built the same way: `/proj/granite-build/g4os/models/granite-swash-2b` (copy of HF
snapshot `af1e3227`, sha256 matches) and `granite-swash-2b-g42tok` (weights symlinked + the 4.2
tokenizer/chat template; no `_experts_implementation`, since there are no experts). Build `d6c461b2`:

```bash
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/sft-swash-3b/build.yaml --skip-version-check \
  --param MODEL_URI=env:///proj/granite-build/g4os/models/granite-swash-2b-g42tok \
  --param EXPERIMENT=sft_swash_2b_v2_odm_90_10_entropy_xml \
  --param QUEUE=gpu-high --param QOS=grp_granite_dot_build_high_1n \
  --param NUM_NODES=1 --param TIME_LIMIT=2-00:00:00
```

~1.0 s/step (vs 2.3 s for swash-3b). The swash eval image serves it too (vLLM's `granite.py` handles
`GraniteSWAForCausalLM` with per-layer sliding windows and sinks).

### Evaluating each checkpoint as it lands

[`swash_eval_watcher.sh`](swash_eval_watcher.sh) (copy it to the sandbox home) polls a run every 5 min and starts the 14-category
evals for checkpoints 1000…5000 once each save has completed (the trainer logged it and
`model.safetensors` + `trainer_state.json` are on disk; a save lands only after that step's eval +
ODM pass). It records started builds in `~/<TAG>_eval_builds.txt`, skips steps already there on a
restart, and exits if the training build fails.

```bash
SWASH_BUILD=<training build id> SWASH_EXP=<EXPERIMENT> SWASH_TAG=<label> \
  setsid nohup ~/swash_eval_watcher.sh >> ~/<label>_eval_watcher.log 2>&1 < /dev/null &
```

Evals are named `eval_<TAG>_<step>`. Defaults are the swash-3b run (`ade7bec1`, tag `swash_3b_ade7bec1`).

### Timing (build `ade7bec1`)

| Phase | Time |
|---|---|
| Data preparation (tokenize + mask, 96 procs) | ~45 min, once per attempt (each retry repeats it) |
| Training step | ~2.3–2.4 s (5000 steps ≈ 3.3 h) |
| Eval + ODM reward pass, every 250 steps | ~35 min (20 passes ≈ 11.5 h) |
| Total | ~15 h |

The eval + ODM pass is most of the wall time, as on the 30b. `EVAL_STEPS=ODM_UPDATE_INTERVAL=500`
would save ~6 h; 250 was kept to match the 3b run.

### Outputs and logs

`/proj/granite-build/g4os/defense/runs/<build_id>/sft_swash_3b_v2_odm_90_10_entropy_xml/`:
`checkpoint-<N>/` every 500 steps and `final/` (full models), `train_rank0.log`,
`training_logs.jsonl`. Logs work as in the [3b README](../sftlora-g42-3b/README.md#logs).

Each checkpoint is **46 GB**: `model.safetensors` 12.6 GB (what the evals load), plus
`pytorch_model_fsdp.bin` 12.6 GB and `optimizer.bin` 24.2 GB (only for resuming; deletable once
the run is done).

**Loss scale.** `enable_reduce_loss_sum` sums the loss over the trained tokens of a batch, so the
numbers are large. Compare with the 4.2 runs on the same recipe, not with a per-token loss:

| Step | swash (full FT) | 3b LoRA | 8b LoRA |
|---|---|---|---|
| 10 | 526.5 | 101.0 | 92.2 |
| 30 | 43.0 | 29.6 | 25.3 |
| 250 | 20.7 | 18.7 | |

Swash starts higher (a base model that has never seen the 4.2 chat format) and is level with the
3b by step 250.

## Run the evals on a checkpoint

Swash checkpoints are full models, so there is no merge step: point `MODEL_PATH` at the checkpoint.
Swash needs a newer vLLM than the 4.2 eval image, so set `EVAL_IMAGE` too. On gpu-mid:

```bash
cd ~/granite.build
RUN=/proj/granite-build/g4os/defense/runs/ade7bec1-f9bf-4022-b757-1d295b176def/sft_swash_3b_v2_odm_90_10_entropy_xml
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param MODEL_PATH=$RUN/checkpoint-500 \
  --param EVAL_IMAGE=docker:/proj/granite-build/g4os/enroot/vllm-swash-nightly-defense-eval.sqsh \
  --param NAME=eval_swash_3b_ade7bec1_500 \
  --param QUEUE=gpu-mid --param QOS= --param TIME_LIMIT=7-00:00:00
```

Each category's tool definitions come from the eval recipe (`tools:` per target: `janes_tool_calling_v6`
for inventory, `military_10092025_tools_v1` for military, …). They must match the category's
reference answers; a category run with another category's tools scores 0 (it can only call the
wrong tool).

Results: `/proj/granite-build/g4os/defense/evals/<NAME>/<category>/result.jsonl` (+
`instance_scores.jsonl`, `eval.log`). Summarize with the eval recipe's `summarize.py`. Set
`DUMP_PREDICTIONS=1` in a target's `extra_env` to also write `predictions_dump.jsonl` (prompt, raw
output, parsed prediction, references).

### The swash eval image

`/proj/granite-build/g4os/enroot/vllm-swash-nightly-defense-eval.sqsh`, built by
[`build_eval_sqsh_swash.sh`](../../../../configurations/assets/environments/skypilot/steps/defense-eval-slurm/image/build_eval_sqsh_swash.sh):

```bash
sbatch -p gpu-mid --qos=grp_granite_dot_build_medium build_eval_sqsh_swash.sh
```

It starts from `vllm-swash-nightly.sqsh` (vLLM 0.30.1rc1, torch 2.13.0+cu130, transformers 5.18.0;
serves `GraniteMoeSWAForCausalLM` via `granitemoeshared.py`) and adds the same eval packages as the
4.2 image, with vllm/torch/transformers pinned as constraints and `datasets<4.8`. The build checks
the imports, the pins, `datasets.DatasetBuilder.as_dataset(run_post_process=...)` and that vLLM
registers the swash architecture, and only then saves.

## Scores

Scores are `subsets_mean` (mean of the trainingforeval and test subsets).

### Checkpoints 500–5000 (`eval_swash_3b_ade7bec1_<step>`)

All 14 categories, gpu-mid, swash eval image. The 3b LoRA final (`cd2bfbc2`) is the reference.
**Bold** = best swash checkpoint.

| Category | 500 | 1000 | 2000 | 3000 | 4000 | 5000 (= final) | 3b LoRA final |
|---|---|---|---|---|---|---|---|
| router | 0.802 | 0.887 | 0.893 | 0.924 | 0.936 | **0.940** | 0.931 |
| inventory | 0.679 | 0.722 | 0.861 | 0.898 | 0.930 | **0.934** | 0.938 |
| nonstategroups | 0.770 | 0.811 | 0.846 | 0.883 | **0.906** | 0.903 | 0.884 |
| news | 0.548 | 0.613 | 0.751 | 0.819 | 0.833 | **0.837** | 0.832 |
| profiles | 0.664 | 0.732 | 0.732 | 0.738 | 0.780 | **0.811** | 0.802 |
| military | 0.601 | 0.592 | 0.678 | 0.754 | **0.808** | 0.805 | 0.755 |
| analysis | 0.630 | 0.688 | 0.730 | 0.757 | **0.763** | 0.756 | 0.773 |
| organization | 0.638 | 0.630 | 0.691 | 0.685 | **0.714** | 0.711 | 0.708 |
| installations | 0.587 | 0.591 | 0.621 | 0.673 | 0.675 | **0.686** | 0.691 |
| ships | **0.660** | 0.640 | 0.659 | **0.660** | 0.654 | 0.638 | 0.656 |
| equipment | 0.521 | 0.486 | 0.530 | 0.542 | **0.568** | 0.548 | 0.575 |
| events | 0.216 | 0.187 | 0.204 | 0.208 | **0.218** | 0.204 | 0.230 |
| **Mean, 12 tool-calling** | 0.610 | 0.632 | 0.683 | 0.712 | **0.732** | 0.731 | 0.731 |
| knowledge | 0.438 | 0.419 | **0.454** | 0.451 | 0.406 | 0.416 | 0.633 |
| knowledge_with_tools | 0.405 | 0.427 | 0.408 | **0.458** | 0.435 | 0.435 | 0.452 |
| **Mean, 2 knowledge** | 0.421 | 0.423 | 0.431 | **0.454** | 0.421 | 0.425 | 0.542 |

**Reading it:**

- **Tool calling matches the 4.2 3b LoRA.** swash reaches 0.732 at 4000 and 0.731 at 5000, the same as
  the 3b LoRA final (0.731). It is ahead on military (+0.05), nonstategroups (+0.02) and profiles
  (+0.01), and behind on equipment, events and analysis (−0.01 to −0.03). Plateau from 4000, as on the
  30b: 4000 → 5000 is −0.001.
- **Fast early gains.** +0.12 on the tool-calling mean from 500 to 4000, most of it by 3000. A full
  fine-tune of a base model gets to the 4.2 3b LoRA (instruct start) in the same 5000 steps.
- **Knowledge stays flat, ~0.41–0.45 at every checkpoint, 0.2 below the 4.2 3b.** swash started from a
  **base** model and the SFT data has no knowledge examples, so it never gains the general knowledge
  the 4.2 instruct models bring. knowledge_with_tools (which rewards picking the right tool) is level
  with the 4.2 3b.
- **Weak spots are the same as every other model:** events ~0.20, equipment ~0.55.
- **Recommendation:** checkpoint-4000 or `final/` (tied on tool calling). For knowledge-heavy use, the
  4.2 instruct models are better; swash would need knowledge data in the mix.

Results: `/proj/granite-build/g4os/defense/evals/eval_swash_3b_ade7bec1_<step>/<category>/`.
Builds: 500 `5df38202` + `8565a6d5` (knowledge), 1000 `3173d686`, 2000 `7d68fc45`, 3000 `913701d7` +
`d4c58beb` (events rerun), 4000 `64dc4ef7`, 5000 `fbdf40e0`. The 1000–5000 builds were started by
`~/swash_eval_watcher.sh` as each checkpoint was saved.

## What swash needed

Every item below failed a run before it was fixed; they are listed so they aren't rediscovered.

**Training stack (transformers 5 vs an image built for 4.55):**

| Failure | Fix |
|---|---|
| `granitemoe_swa` unknown (transformers 4.55) | transformers 5.18 on `PYTHONPATH` ahead of the image's site-packages |
| `cannot import AutoModelForVision2Seq` | renamed to `AutoModelForImageTextToText` (fms-hf-tuning `sft_trainer.py`) |
| `No module named 'aim'` at start-up | transformers 5 `_is_package_available` returns a non-empty `(bool, version)` tuple, so 4.x callers took every optional tracker as installed. Local patch in the pinned transformers: the tuple is falsy when the package is missing (`import_utils.py.orig` kept) |
| `add_special_tokens() got an unexpected keyword argument 'replace_additional_special_tokens'` | renamed to `replace_extra_special_tokens` (`tokenizer_utils.py`) |
| `BatchEncoding` has no `.clone()` in the masking handler | `apply_chat_template(tokenize=True, return_tensors="pt")` returns a BatchEncoding in transformers 5; unwrap `input_ids` at the 4 call sites (`data_handlers.py`). Verified: only the assistant `<tool_call>…<\|im_end\|>` tokens are labelled |
| `cannot import name 'scaled_mm_configs'` at the first step | the 6mar2026 image upgraded torch 2.7 → 2.8 in place and left 80 torch-2.7-only `.py` files; flex attention's `torch.compile` imports one. A setup command deletes every file in the 2.7 RECORD that 2.8's lacks |
| `'SFTConfig' object has no attribute 'past_index'` at step 250 (first eval) | removed in transformers 5; fms-acceleration-odm's `_evaluate` reads it. Patched to `getattr(args, "past_index", -1)` |

**Attention:** flex attention, measured on one H200 with an 8192-token packed batch and gradient
checkpointing: flex 1.55 s / 17.0 GB, eager 3.69 s / 35.7 GB, same loss. FlashAttention-3 (the hub
kernel `kernels-community/vllm-flash-attn3`) has no build for torch 2.8 that transformers 5.18's
`kernels` 0.17 can load; the cached copy kd-sandbox used needs `kernels` 0.12.3, which 5.18 rejects.

**Image:** `quay.io/modh/fms-hf-tuning:v3.2.2` was tried first. It is RHEL-based with no rsync and no
apt, so SkyPilot's file sync failed (fixed in the SkyPilot fork: bind-mount the host's rsync,
commit `dc18a650`) and then its container setup, which apt-installs its SSH server, failed too.
Hence the Ubuntu 6mar2026 image.

**Eval (vLLM 0.30 vs fm-eval/unitxt written for vLLM 0.11):**

| Failure | Fix |
|---|---|
| vLLM 0.11.2 can't load `GraniteMoeSWAForCausalLM` | swash eval image on vLLM 0.30.1rc1 |
| no `git` in the nightly image (unitxt fork install) | install the fork from GitHub's archive zip of the pinned commit |
| `DatasetBuilder.as_dataset() takes from 1 to 3 positional arguments but 5 were given` | `datasets` 4.8 dropped `run_post_process`; pin `datasets<4.8` (4.7.0) |
| `Unexpected keyword argument 'best_of'` | unitxt passes every engine field to `SamplingParams`; `src/patch_unitxt_vllm.py` keeps only fields vLLM declares |
| `User-specified max_model_len (32768) is greater than the derived max_model_len (8192)` | same patch caps unitxt's hardcoded 32768 at the model's `max_position_embeddings` (prompts are ~1k tokens) |
| knowledge scripts abort: "start a new process before the current process has finished its bootstrapping phase" | they run at module level and vLLM 0.30 forces `spawn` once CUDA is up; every eval script now runs via `src/run_as_main.py`, so the spawn child doesn't re-run it |
| eval failures leave no traceback in `eval.log` | the scripts' `finally` hard-exits before the exception prints; reproduce by hand with an `except: traceback.print_exc()` before the `finally` |

The two unitxt fixes and the wrapper are no-ops on the vLLM 0.11 image the 4.2 evals use (all
`SamplingParams` fields accepted; 4.2 models allow ≥ 32768; scripts with a `__main__` guard run as
before).

## Validated runs

| Build | What | Result |
|---|---|---|
| `ade7bec1` | 5000-step full fine-tune, gpu-high, 1 node | **SUCCESS**: `train_runtime` 57030 s (15h50m), checkpoints 500–5000 + `final/` |
| evals 500–5000 | all 14 categories per checkpoint, gpu-mid | **SUCCESS** (see the score table); a few categories needed the automatic retry or a single-category rerun |

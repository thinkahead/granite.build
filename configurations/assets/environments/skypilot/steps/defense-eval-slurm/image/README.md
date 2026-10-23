# Defense-eval squashfs image

How to build `/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh`,
the enroot image the `defense-eval-slurm` step runs in (its `image_id`). It's used to
run the defense evals from `github.ibm.com/karve/fm-eval@unitxt_benchmarks_local`.

There's no Dockerfile. The image is built on a BlueVela compute node by starting the
public vLLM image with pyxis, pip-installing into it, and saving the container with
`--container-save`. [`build_eval_sqsh.sh`](build_eval_sqsh.sh) does all of it.

## What's in it, and what isn't

| Layer | Contents | Source |
|---|---|---|
| Base | `vllm/vllm-openai:v0.11.2`: vLLM 0.11.2, torch 2.9.0+cu129, transformers 4.57.1, Python 3.12 (`/usr/local/lib/python3.12/dist-packages`) | Docker Hub |
| Vela pip line | `pytz ibm-watsonx-ai==1.3.34 jsonschema-rs rouge-score==0.1.2 tabulate==0.9.0 unitxt==1.26.5 json_repair protobuf litellm==1.77.2.post1 vllm==0.11.2` | Vela `defence/granite_dot_build_enhanced/eval_g42nt_local.yaml` → `/tmp/command.sh` |
| Added | `peft` | for merged-model and adapter handling |
| unitxt fork | `unitxt@git+https://github.com/thinkahead/unitxt@c489d8c8` (reports 1.26.8), replacing PyPI unitxt | Vela installs this fork too, but unpinned |

**fm-eval isn't in the image.** Every eval job clones
`github.ibm.com/karve/fm-eval` at `code_config.ref` (default `unitxt_benchmarks_local`,
recipe `--param FM_EVAL_REF=...`) into its own run directory, then patches it with the
step's `src/prepare_fm_eval.sh`. The image only needs rebuilding when the **Python
dependencies** change, not when fm-eval code changes. A branch with the same
dependencies, such as `janes-direct-api`, runs on this image unchanged.

**Left out of the Vela pip line, on purpose:** three packages that are GHE-only and
need `GB_SDK_GIT_PAT` at install time:

| Vela installs | Replaced by |
|---|---|
| `lakehouse-eval-api` (`lh_eval_api`) | the step's `src/lh_shim/lh_eval_api`: writes scores to `OUTPUT_PATH`, reads pulled Lakehouse tables from parquet |
| `dmf-lib` (`lakehouse`) | not needed: no Lakehouse access at eval time |
| `dmf-aspera` | not needed |

Vela's other `command.sh` lines are file patches, not installs. They're applied per run
by `src/prepare_fm_eval.sh`, so they aren't in the image either.

## Build

On a BlueVela login node (login0/1/2), from a copy of this directory:

```bash
sbatch build_eval_sqsh.sh
```

Takes ~7 min: ~1 min to import the base image, ~5 min of pip, ~1 min to save. Needs one
H200 node slot on `gpu-high`, so the GPU is visible while vLLM imports. It writes
`<SQSH>.tmp.<jobid>` and renames it to `SQSH` only when the save finishes, so a failed
or killed build never replaces a working image. Log:
`/proj/granite-build/g4os/enroot/build-defense-eval-sqsh-<jobid>.out`.

Knobs (environment variables, passed with `sbatch --export=ALL,NAME=value`):

| Variable | Default | Use |
|---|---|---|
| `SQSH` | `/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh` | Output path. Build to a new name to test, then switch the step's `image_id` |
| `BASE_IMAGE` | `vllm/vllm-openai:v0.11.2` | Base image. Changing the vLLM version also needs `vllm==` in the pip line |
| `UNITXT_REF` | `c489d8c83eae090dbc1d4dd042f3eba1f57152a0` | Commit of `thinkahead/unitxt` to install |

Example, building a test image without touching the one in use:

```bash
sbatch --export=ALL,SQSH=/proj/granite-build/g4os/enroot/defense-eval-test.sqsh build_eval_sqsh.sh
```

The `.sqsh` is about 15 GB. A fresh pull of the base image from Docker Hub happens once
per build, on the compute node; no registry login is needed.

## Verify

The build checks itself and fails before saving if any of these is wrong:

- **Imports:** every package the eval scripts import loads, and its version is
  printed (`OK <module> <version>` lines in the log).
- **Patch targets:** the files `prepare_fm_eval.sh` edits exist, including unitxt's
  `self.llm.chat(..., tools=tools)` (the nothink patch), litellm's `http_handler.py` and
  transformers' `tokenization_utils_base.py`. That catches a dependency bump that would
  otherwise make a patch silently not apply.
- **unitxt fork:** it honours `VLLM_TP_SIZE`. That's what gives TP=2 on 2 GPUs; PyPI
  unitxt ignores it.

The log ends with `build: done -> <path>`. Then run one category against the new image
before switching all evals to it, e.g. with the step's `image_id` pointing at the
test path:

```bash
gb build start -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml \
  merge inventory --param NAME=image-test --skip-version-check
```

Inventory on the `cd2bfbc2` final adapter scored **0.938** with the current image
(build `70fd7a88`).

## Current image

| | |
|---|---|
| Path | `/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh` |
| Built | 2026-10-23, SLURM job 324830 (same steps as the script, before it existed) |
| Script verified | 2026-10-23, job 331804: `build_eval_sqsh.sh` rebuilt it to a test path in 7m46s, all checks passed, identical package versions |
| Size | 15 GB (15,304,929,280 bytes) |
| sha256 prefix | `4dca9ae9a1eb822d` |
| Versions | vllm 0.11.2, unitxt 1.26.8 (fork `c489d8c8`), peft 0.21.2, torch 2.9.0+cu129, transformers 4.57.1, ibm-watsonx-ai 1.3.34, litellm 1.77.2.post1 |
| Used by | builds `70fd7a88` (12 tool-calling), `0ec74008` (2 knowledge) |

## Why not the Vela image

Vela ran `icr.io/rl4gb/custom-step:2` and pip-installed this whole stack, plus the three
GHE packages, at the start of every job. BlueVela can't do that as-is:

- **Registry access:** BlueVela's enroot has credentials for `us.icr.io`, not `icr.io/rl4gb`.
- **No GHE from compute nodes:** they reach PyPI and github.com, but GHE installs need a
  token inside the job.
- **Cost:** installing vLLM per job costs ~5 min on every one of the 14 category jobs.

Baking the stack once into a `.sqsh` on `/proj` fixes all three. pyxis mounts it
directly, with no per-job import.

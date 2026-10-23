# defense-eval-slurm (SkyPilot / SLURM)

The defense tool-calling eval with vLLM on BlueVela SLURM. It runs the same code
the same way as the Vela `defence/granite_dot_build_enhanced/eval_g42nt_local.yaml`
targets did through `space://steps/custom_code`:

```
env SYSTEM_PROMPT=none TEMPERATURE=0 TOP_P=1.0 PROVIDER=vllm VLLM_TP_SIZE=2 \
    MODEL_NAME=<merged model> BATCH_SIZE=<n> OUTPUT_PATH=<dir> \
    DATASET_NAMES=<a,b> TOOLS=<tools> \
    python unitxt_benchmarks/defense/defense_unitxt_evaluation_with_profile[_for_router].py
```

The code is `github.ibm.com/karve/fm-eval@unitxt_benchmarks_local`, the same branch,
cloned per run, and the per-run patches are the same (nothink `chat_template_kwargs`,
`max_tokens=1024`, no 8-bit). Recipe: `recipes/granite4-defense-lora/slurm/eval-defense/`
(merge → 12 category targets), with the same per-category `DATASET_NAMES`, `TOOLS` and
`BATCH_SIZE` as Vela.

The companion step **`lora-merge-slurm`** merges a LoRA adapter into its base, writing
bf16 safetensors plus tokenizer and chat template. It's the BlueVela version of Vela's
`/gb-read-write/karve/lora_merge/<run>_ckpt<N>_merged` dirs. vLLM needs a full model dir,
so this step refuses an adapter.

## What differs from Vela

| | Vela (`custom_code`, k8s) | BlueVela (`defense-eval-slurm`) |
|---|---|---|
| Image | `icr.io/rl4gb/custom-step:2` + per-run `pip install ... vllm==0.11.2 ...` | pre-staged `vllm/vllm-openai:v0.11.2` with that pip line baked in (below) |
| vLLM / torch / transformers | 0.11.2 / (custom-step) / (custom-step) | 0.11.2 / 2.9.0+cu129 / 4.57.1 |
| unitxt | `thinkahead/unitxt` (HEAD at run time) | `thinkahead/unitxt@c489d8c8` (pinned), reports 1.26.8 |
| Eval data (`DATA_PATH`) | `/gb-read-write/karve/evaldata` | `/proj/granite-build/g4os/defense/defense_evaldata/20260914T194533` (all 36 tool-calling tables present) |
| Scores | uploaded to the Lakehouse by `lh_eval_api` | written to `OUTPUT_PATH/result.jsonl` + `instance_scores.jsonl` by `src/lh_shim` |
| GPUs | 2 × A100/H100, TP=2 | 2 × H200, TP=2 |
| Knowledge targets (2) | run (Lakehouse datasets + watsonx judge) | **not ported** |

**Scores without the Lakehouse.** BlueVela compute nodes have no `LAKEHOUSE_TOKEN`
and no route to the Lakehouse. The eval scripts import only `upload_unitxt_results_to_lh`
from `lh_eval_api` and then call `.save_url_to_json(<OUTPUT_PATH>/result.jsonl)`.
`src/lh_shim/lh_eval_api` provides that function with the same signature: it writes
`result.jsonl` (model, `global_scores`, `subsets_scores`) and `instance_scores.jsonl`
(prediction, references and score per instance). Inference, the
`DefenseToolCallingMetric` and the printed `Global Results` / `Subsets Results` are the
unmodified fm-eval path.

**Vela patches not ported**, all knowledge-only or A100-only: the litellm `http_handler`
retry, the `defense_knowledge` format/metric seds, the `base_model_inference.py` copy,
and the fused-MoE A100 config copy.

## Requirements

- **GHE token.** fm-eval is private. The step maps the secret `GITHUB_IBM_PAT` to
  `GB_SDK_GIT_PAT`. With the standalone gbserver, export it as
  `GBSERVER_SECRET_GITHUB_IBM_PAT` (`~/gbserver-standalone/start.sh` takes it from the
  sandbox's `$GH_ENTERPRISE_TOKEN`). The clone uses `GIT_ASKPASS`, so the token isn't
  written into the clone's `.git/config` on `/proj`.
- **CPUs.** Run with the patched SkyPilot from the `tuning-slurm` README, or the run step
  gets 1 core.
- **A merged model**, not an adapter. The step fails fast if `adapter_config.json` exists
  without `config.json`.

## Image

`/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh`
(15 GB, built 2026-10-23, job 324830). To rebuild it:

```bash
srun -p gpu-high --qos=grp_granite_dot_build_high_1n --account=grp_granite_dot_build \
  -c 16 --mem=64G --gres=gpu:1 --constraint=h200 -t 1:30:00 \
  --container-image=vllm/vllm-openai:v0.11.2 \
  --container-save=/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh \
  --container-remap-root --container-writable --no-container-mount-home bash -c '
python3 -m pip install --timeout=1200 --retries=10 pytz ibm-watsonx-ai==1.3.34 jsonschema-rs \
  rouge-score==0.1.2 tabulate==0.9.0 unitxt==1.26.5 json_repair protobuf litellm==1.77.2.post1 \
  vllm==0.11.2 peft
python3 -m pip uninstall -y unitxt
python3 -m pip install "unitxt@git+https://github.com/thinkahead/unitxt@c489d8c83eae090dbc1d4dd042f3eba1f57152a0"'
```

This is the Vela `command.sh` pip line, minus the GHE-only packages (`lakehouse-eval-api`,
`dmf-lib`, `dmf-aspera`). `vllm==0.11.2` stays pinned, so pip doesn't move it.

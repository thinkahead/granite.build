#!/bin/bash
# Build the enroot squashfs the defense-eval-slurm step runs in:
#   /proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh
#
# vllm/vllm-openai:v0.11.2 + the Vela eval_g42nt_local.yaml command.sh pip line, built
# on a BlueVela compute node with pyxis --container-save. Submit from a login node:
#
#   sbatch build_eval_sqsh.sh                      # default output path
#   SQSH=/proj/.../my-test.sqsh sbatch build_eval_sqsh.sh
#
# The image holds the Python environment ONLY. The fm-eval code is NOT baked in: the
# step clones github.ibm.com/karve/fm-eval@unitxt_benchmarks_local (code_config.ref)
# into each run and patches it there (src/prepare_fm_eval.sh), so the same image serves
# any fm-eval branch with the same dependencies.
#
# See README.md in this directory for what each part is for and how to verify.
#SBATCH -J build-defense-eval-sqsh
#SBATCH -p gpu-high --qos=grp_granite_dot_build_high_1n --account=grp_granite_dot_build
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=64G --gres=gpu:1 --constraint=h200
#SBATCH -t 1:30:00
#SBATCH -o /proj/granite-build/g4os/enroot/build-defense-eval-sqsh-%j.out
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.11.2}"
SQSH="${SQSH:-/proj/granite-build/g4os/enroot/vllm-vllm-openai-v0.11.2-defense-eval.sqsh}"
UNITXT_REF="${UNITXT_REF:-c489d8c83eae090dbc1d4dd042f3eba1f57152a0}"

# Write to a temp name and rename at the end: a job killed mid-save must not leave a
# truncated .sqsh at the path running evals read (pyxis would fail to mount it).
TMP_SQSH="${SQSH}.tmp.${SLURM_JOB_ID:-$$}"
rm -f "$TMP_SQSH"
echo "build: $BASE_IMAGE -> $SQSH (unitxt fork @ $UNITXT_REF)"

srun --container-image="$BASE_IMAGE" --container-save="$TMP_SQSH" \
  --container-remap-root --container-writable --no-container-mount-home \
  --export=ALL,UNITXT_REF="$UNITXT_REF" \
  bash -c '
set -eux
python3 -c "import sys,vllm,torch,transformers;print(sys.version);print(\"base:\",vllm.__version__,torch.__version__,transformers.__version__)"

# 1) The Vela eval_g42nt_local.yaml command.sh pip line, verbatim, minus the three
#    GHE-only packages (lakehouse-eval-api, dmf-lib, dmf-aspera; replaced at run time by
#    the step'"'"'s src/lh_shim). vllm==0.11.2 stays pinned so pip keeps the base image'"'"'s
#    CUDA build. peft is added for the lora-merge path and adapter detection.
python3 -m pip install --timeout=1200 --retries=10 \
  pytz ibm-watsonx-ai==1.3.34 jsonschema-rs rouge-score==0.1.2 tabulate==0.9.0 \
  unitxt==1.26.5 json_repair protobuf litellm==1.77.2.post1 vllm==0.11.2 peft

# 2) Vela then replaces unitxt with the thinkahead fork (VLLMInferenceEngine honours
#    VLLM_TP_SIZE, tool-call chat path). Vela installed the fork HEAD at run time; pin it.
python3 -m pip uninstall -y unitxt
python3 -m pip install "unitxt@git+https://github.com/thinkahead/unitxt@${UNITXT_REF}"

# 3) Verify what the eval scripts import, and the patch targets prepare_fm_eval.sh edits.
python3 - <<PY
import importlib, os
for m in ("vllm", "unitxt", "peft", "litellm", "torch", "transformers", "ibm_watsonx_ai",
          "pyarrow", "jsonschema_rs", "json_repair", "rouge_score", "tabulate"):
    mod = importlib.import_module(m)
    print("OK", m, getattr(mod, "__version__", "?"))
import unitxt, litellm, transformers
import ibm_watsonx_ai.foundation_models.inference.base_model_inference as bmi
inf = open(os.path.join(os.path.dirname(unitxt.__file__), "inference.py")).read()
assert "self.llm.chat(inputs, self.sampling_params, tools=tools)" in inf, "unitxt nothink patch target missing"
assert "VLLM_TP_SIZE" in inf, "unitxt fork without VLLM_TP_SIZE support"
assert os.path.exists(os.path.join(os.path.dirname(litellm.__file__), "llms", "custom_httpx", "http_handler.py"))
assert os.path.exists(os.path.join(os.path.dirname(transformers.__file__), "tokenization_utils_base.py"))
print("OK patch targets present; watsonx base_model_inference at", bmi.__file__)
PY
'

mv -f "$TMP_SQSH" "$SQSH"
ls -la "$SQSH"
echo "build: done -> $SQSH"

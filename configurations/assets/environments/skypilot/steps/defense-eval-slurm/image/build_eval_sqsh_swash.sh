#!/bin/bash
# Build the defense-eval squashfs for granite-swash (GraniteMoeSWAForCausalLM):
#   /proj/granite-build/g4os/enroot/vllm-swash-nightly-defense-eval.sqsh
#
# Same eval layer as build_eval_sqsh.sh (the Vela eval_g42nt_local.yaml pip line + the
# thinkahead unitxt fork), on top of vllm-swash-nightly.sqsh instead of vllm 0.11.2:
# vLLM 0.30.1rc1.dev493 (serves GraniteMoeSWAForCausalLM via granitemoeshared.py),
# torch 2.13.0+cu130, transformers 5.18.0. vLLM 0.11.2 / transformers 4.57 can't load swash.
#
# vllm is NOT reinstalled here (the Vela line pins vllm==0.11.2; dropping that pin keeps
# the nightly). The pip install uses --no-deps for unitxt/peft so pip can't pull a
# different torch or transformers over the nightly's.
#
#   sbatch build_eval_sqsh_swash.sh                              # gpu-high
#   sbatch -p gpu-mid --qos=grp_granite_dot_build_medium build_eval_sqsh_swash.sh   # gpu-mid
#SBATCH -J build-defense-eval-sqsh-swash
#SBATCH -p gpu-high --qos=grp_granite_dot_build_high_1n --account=grp_granite_dot_build
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=64G --gres=gpu:1 --constraint=h200
#SBATCH -t 1:30:00
#SBATCH --exclude=p1-r18-n1,p1-r10-n1,p3-r10-n2,p1-r30-n3,p1-r13-n3
#SBATCH -o /proj/granite-build/g4os/enroot/build-defense-eval-sqsh-swash-%j.out
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-/proj/granite-build/g4os/enroot/vllm-swash-nightly.sqsh}"
SQSH="${SQSH:-/proj/granite-build/g4os/enroot/vllm-swash-nightly-defense-eval.sqsh}"
UNITXT_REF="${UNITXT_REF:-c489d8c83eae090dbc1d4dd042f3eba1f57152a0}"

TMP_SQSH="${SQSH}.tmp.${SLURM_JOB_ID:-$$}"
rm -f "$TMP_SQSH"
echo "build: $BASE_IMAGE -> $SQSH (unitxt fork @ $UNITXT_REF)"
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

srun --container-image="$BASE_IMAGE" --container-save="$TMP_SQSH" \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  --container-remap-root --container-writable --no-container-mount-home \
  --export=ALL,UNITXT_REF="$UNITXT_REF" \
  bash -c '
set -eux
python3 -c "import sys,vllm,torch,transformers;print(sys.version);print(\"base:\",vllm.__version__,torch.__version__,transformers.__version__)"
PINS="$(python3 -c "import vllm,torch,transformers;print(f\"vllm=={vllm.__version__} torch=={torch.__version__} transformers=={transformers.__version__}\")")"
echo "$PINS" | tr " " "\n" > /tmp/pins.txt
# datasets 4.8 dropped DatasetBuilder.as_dataset(run_post_process, verification_mode), which
# unitxt 1.26 still passes positionally ("takes from 1 to 3 positional arguments but 5 were
# given" at unitxt/dataset.py:167, before any model loads). 4.7 keeps them and allows the
# huggingface_hub 1.x that transformers 5.18 needs.
echo "datasets<4.8" >> /tmp/pins.txt

# 1) The Vela pip line minus vllm==0.11.2 (keep the nightly), with the base image pins as
#    constraints so nothing replaces vllm/torch/transformers.
python3 -m pip install --timeout=1200 --retries=10 -c /tmp/pins.txt \
  pytz ibm-watsonx-ai==1.3.34 jsonschema-rs rouge-score==0.1.2 tabulate \
  json_repair protobuf litellm==1.77.2.post1
# unitxt fork (VLLM_TP_SIZE, tool-call chat path) and peft, with their dependencies (datasets,
# evaluate, scipy, diskcache, ...: the nightly image has none of them) under the same pins.
# The nightly image has no git, so install the fork from the GitHub archive zip of the pinned
# commit instead of a git+ URL (same tree).
python3 -m pip install --timeout=1200 --retries=10 -c /tmp/pins.txt \
  "unitxt @ https://github.com/thinkahead/unitxt/archive/${UNITXT_REF}.zip" peft

# 2) Verify: imports, patch targets of prepare_fm_eval.sh, the pins held, and that vLLM
#    resolves the swash architecture.
python3 - <<PY
import importlib, os
for m in ("vllm", "unitxt", "peft", "litellm", "torch", "transformers", "ibm_watsonx_ai",
          "pyarrow", "jsonschema_rs", "json_repair", "rouge_score", "tabulate",
          "datasets", "evaluate", "scipy", "diskcache"):
    mod = importlib.import_module(m)
    print("OK", m, getattr(mod, "__version__", "?"))
import unitxt, litellm, transformers, vllm, torch
pins = dict(l.split("==") for l in open("/tmp/pins.txt").read().split() if "==" in l)
for k, v in pins.items():
    got = importlib.import_module(k).__version__
    assert got == v, f"{k} changed {v} -> {got}"
import datasets, inspect
assert "run_post_process" in inspect.signature(datasets.DatasetBuilder.as_dataset).parameters, \
    f"datasets {datasets.__version__}: as_dataset lacks run_post_process; unitxt 1.26 needs it"
inf = open(os.path.join(os.path.dirname(unitxt.__file__), "inference.py")).read()
assert "self.llm.chat(inputs, self.sampling_params, tools=tools)" in inf, "unitxt nothink patch target missing"
assert "VLLM_TP_SIZE" in inf, "unitxt fork without VLLM_TP_SIZE support"
assert os.path.exists(os.path.join(os.path.dirname(litellm.__file__), "llms", "custom_httpx", "http_handler.py"))
assert os.path.exists(os.path.join(os.path.dirname(transformers.__file__), "tokenization_utils_base.py"))
from vllm.model_executor.models.registry import ModelRegistry
assert "GraniteMoeSWAForCausalLM" in ModelRegistry.get_supported_archs(), "vLLM lacks GraniteMoeSWAForCausalLM"
print("OK pins held:", pins); print("OK GraniteMoeSWAForCausalLM registered")
PY
'

mv -f "$TMP_SQSH" "$SQSH"
ls -la "$SQSH"
echo "build: done -> $SQSH"

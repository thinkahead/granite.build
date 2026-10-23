#!/bin/bash
# Per-run patches to the fm-eval clone and the container's site-packages, ported from
# the Vela eval_g42nt_local.yaml /tmp/command.sh (tool-calling targets). Paths are
# resolved from the imported modules rather than hardcoded, because the BlueVela image
# (vllm/vllm-openai) installs into /usr/local/lib/python3.12/dist-packages, not the
# Vela custom-step image's /step_venv or /app/git_clone.
#   $1 = fm-eval clone dir
set -eu
CODE_DIR="$1"
PY="${PYBIN:-python3}"

# Disable thinking in vLLM chat (granite-4.2 nothink evaluation).
UNITXT_INF="$("$PY" -c 'import unitxt,os;print(os.path.join(os.path.dirname(unitxt.__file__),"inference.py"))')"
sed -i 's/self\.llm\.chat(inputs, self\.sampling_params, tools=tools)/self.llm.chat(inputs, self.sampling_params, tools=tools, chat_template_kwargs={"thinking": False, "enable_thinking": False})/' "$UNITXT_INF"
grep -q 'chat_template_kwargs={"thinking": False' "$UNITXT_INF" || { echo "defense-eval: FATAL: nothink patch did not apply to $UNITXT_INF" >&2; exit 1; }

# Vela: max_tokens=1024 for the tool-calling scripts; 8-bit quantization off.
sed -i "s/max_tokens=[^,]*,/max_tokens=1024,/" "$CODE_DIR"/unitxt_benchmarks/defense/defense_unitxt_evaluation_with_profile*.py
sed -i "s/\"load_in_8bit\": True,/\"load_in_8bit\": False,/" "$CODE_DIR"/unitxt_benchmarks/defense/defense_unitxt_evaluation*.py

# transformers tokenizer config access fix (Vela applied it when the file existed).
TOK_BASE="$("$PY" -c 'import transformers,os;print(os.path.join(os.path.dirname(transformers.__file__),"tokenization_utils_base.py"))')"
if [ -e "$TOK_BASE" ]; then sed -i "s/_config.model_type/_config.get(\"model_type\")/g" "$TOK_BASE"; fi

# ── Knowledge evals (watsonx llm-as-judge), from the Vela command.sh ──
KDIR="$CODE_DIR/unitxt_benchmarks/defense_knowledge"
# Vela: litellm retry-on-HTTP>=400 for the judge calls + chat_api format with the
# instruction placed in user turns. Vela gated both on litellm being installed.
LITELLM_HTTP="$("$PY" -c 'import litellm,os;print(os.path.join(os.path.dirname(litellm.__file__),"llms","custom_httpx","http_handler.py"))' 2>/dev/null || true)"
if [ -n "$LITELLM_HTTP" ] && [ -e "$LITELLM_HTTP" ]; then
  sed -i "s|format=.*|format=\"formats.chat_api[place_instruction_in_user_turns=True]\",|" "$KDIR/defense_knowledge_unitxt_evaluation.py"
  if ! grep -q 'RETRYING {count}' "$LITELLM_HTTP"; then
    sed -i -e '/response = await self.client.send(req, stream=stream)/a\'$'\n''            count=0\n            while response.status_code>=400:\n                time.sleep(1)\n                count+=1\n                print(f"RETRYING {count} {response.status_code}")\n                response = await self.client.send(req, stream=stream)\n                if count>10: break\n' "$LITELLM_HTTP"
  fi
fi
# Vela: shorter reference-template instruction, and the swim llama-3.3-70b judge.
sed -i "s|Answer .* domain|Give a brief reply in the context of military/defense domain|" "$CODE_DIR/fm_eval/catalogs/private/templates/defense/defense_reference_template.json"
sed -i "s/metric = .*/metric = \"metrics.ilab_swim_language.swim.llama_llm_as_a_judge\"/" "$KDIR"/defense_knowledge_unitxt_evaluation*.py
sed -i "s/\"load_in_8bit\": True,/\"load_in_8bit\": False,/" "$KDIR"/defense_knowledge_unitxt_evaluation*.py
# Vela: /gb-read-write/karve/base_model_inference.py over ibm_watsonx_ai's copy. Same
# ibm-watsonx-ai 1.3.34 file plus 429 retry and httpx timeout retries for the judge
# (WML_READ_TIMEOUT / WML_TIMEOUT_RETRIES). Shipped as src/base_model_inference.py.
STEP_SRC="$(cd "$(dirname "$0")" && pwd)"
WX_BMI="$("$PY" -c 'import ibm_watsonx_ai.foundation_models.inference.base_model_inference as m;print(m.__file__)' 2>/dev/null || true)"
if [ -n "$WX_BMI" ] && [ -e "$STEP_SRC/base_model_inference.py" ]; then
  cp "$STEP_SRC/base_model_inference.py" "$WX_BMI"
  echo "defense-eval: replaced $WX_BMI (watsonx judge retries)"
fi

# Vela-only items intentionally NOT ported:
#  * lakehouse-eval-api / dmf-lib / dmf-aspera installs -> replaced by src/lh_shim
#    (scores to OUTPUT_PATH; Lakehouse tables read from pulled parquet under DATA_PATH).
#  * fused_moe A100 config copy -> A100-only, BlueVela is H200.
echo "defense-eval: patches applied (unitxt=$UNITXT_INF)"

#!/usr/bin/env python3
# Two unitxt 1.26 VLLMInferenceEngine fixes for newer vLLM (0.30, granite-swash). Idempotent.
#  1. It passes every VLLMParamsMixin field to vllm.SamplingParams, including best_of / _real_n,
#     which vLLM removed ("Unexpected keyword argument 'best_of'"). Keep only fields this
#     vLLM's SamplingParams declares.
#  2. It hardcodes LLM(max_model_len=32768); vLLM 0.30 rejects a value above the model's own
#     max_position_embeddings (granite-swash: 8192). Cap it at the model's limit. The defense
#     eval prompts are ~1k tokens, so 8192 loses nothing; models with >= 32768 are unchanged.
import os, sys, unitxt
p = os.path.join(os.path.dirname(unitxt.__file__), "inference.py")
s = open(p).read()
done = []

m1 = "# swash-eval: SamplingParams fields filter"
if m1 not in s:
    a = "        self.sampling_params = SamplingParams(**args)\n"
    assert s.count(a) == 1, f"SamplingParams target count {s.count(a)}"
    s = s.replace(a, "        " + m1 + "\n"
        "        import inspect as _inspect\n"
        "        _ok = set(_inspect.signature(SamplingParams).parameters)\n"
        "        _dropped = sorted(k for k in args if k not in _ok)\n"
        "        args = {k: v for k, v in args.items() if k in _ok}\n"
        "        if _dropped:\n"
        "            print(f\"unitxt: SamplingParams ignoring fields unsupported by this vLLM: {_dropped}\")\n"
        + a)
    done.append("SamplingParams filter")

m2 = "# swash-eval: cap max_model_len"
if m2 not in s:
    a = "                    max_model_len=32768,\n"
    assert s.count(a) == 1, f"max_model_len target count {s.count(a)}"
    s = s.replace(a, "                    max_model_len=_swash_max_model_len(self.model, 32768),\n")
    helper = ("\n\ndef _swash_max_model_len(model, default):  " + m2 + "\n"
        "    try:\n"
        "        from transformers import AutoConfig\n"
        "        mpe = getattr(AutoConfig.from_pretrained(model, trust_remote_code=True), \"max_position_embeddings\", None)\n"
        "    except Exception:\n"
        "        mpe = None\n"
        "    n = min(default, int(mpe)) if mpe else default\n"
        "    if n != default:\n"
        "        print(f\"unitxt: max_model_len capped {default} -> {n} (model max_position_embeddings)\")\n"
        "    return n\n")
    anchor = "\nclass VLLMParamsMixin(Artifact):"
    assert s.count(anchor) == 1
    s = s.replace(anchor, helper + anchor)
    done.append("max_model_len cap")

open(p, "w").write(s)
print("unitxt patches:", ", ".join(done) if done else "already applied", "->", p)

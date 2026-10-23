#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model and save a full HF model dir.

vLLM (and the defense eval) loads a full model directory, not an adapter. This
writes base+adapter as merged bf16 safetensors, plus the tokenizer and chat
template from the ADAPTER dir (fms-hf-tuning saves the tokenizer it trained
with there), falling back to the base model's.

Refuses adapters that carry modules_to_save or resized embeddings: those need
the trained embed/lm_head rows too, and a plain merge would silently drop them.
"""

import argparse
import json
import os
import shutil
import sys

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Base model dir")
    p.add_argument("--adapter", required=True, help="LoRA adapter dir (adapter_config.json)")
    p.add_argument("--out", required=True, help="Output dir for the merged model")
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()

    with open(os.path.join(a.adapter, "adapter_config.json"), encoding="utf-8") as f:
        acfg = json.load(f)
    if acfg.get("peft_type") != "LORA":
        print(f"lora-merge: FATAL: peft_type={acfg.get('peft_type')!r}, expected LORA", file=sys.stderr)
        return 2
    if acfg.get("modules_to_save"):
        print(f"lora-merge: FATAL: adapter has modules_to_save={acfg['modules_to_save']}; "
              "a plain merge would drop those trained modules", file=sys.stderr)
        return 2
    info_path = os.path.join(a.adapter, "added_tokens_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if int(info.get("num_new_tokens", 0)) != 0:
            print(f"lora-merge: FATAL: adapter added {info['num_new_tokens']} tokens; "
                  "merge would need the resized embeddings", file=sys.stderr)
            return 2

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, a.dtype)
    print(f"lora-merge: base={a.base}")
    print(f"lora-merge: adapter={a.adapter} (r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}, "
          f"targets={sorted(acfg.get('target_modules') or [])})")
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()

    tmp = a.out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)

    for name in TOKENIZER_FILES:
        src = os.path.join(a.adapter, name)
        if not os.path.exists(src):
            src = os.path.join(a.base, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp, name))
    gen = os.path.join(a.base, "generation_config.json")
    if os.path.exists(gen) and not os.path.exists(os.path.join(tmp, "generation_config.json")):
        shutil.copy2(gen, tmp)

    with open(os.path.join(tmp, "lora_merge_info.json"), "w", encoding="utf-8") as f:
        json.dump({"base": a.base, "adapter": a.adapter, "dtype": a.dtype,
                   "r": acfg.get("r"), "lora_alpha": acfg.get("lora_alpha"),
                   "target_modules": acfg.get("target_modules")}, f, indent=2)

    # Publish atomically so a reader never sees a half-written model dir.
    shutil.rmtree(a.out, ignore_errors=True)
    os.rename(tmp, a.out)
    print(f"lora-merge: wrote {a.out}: {sorted(os.listdir(a.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

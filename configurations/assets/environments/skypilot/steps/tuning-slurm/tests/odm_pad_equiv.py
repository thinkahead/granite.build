"""Equivalence + speed test: ODM entropy reward at eval batch 1 (unpadded) vs batch N (padded).

Uses the real fms-acceleration-odm compute_reward on a real model and real defense eval
examples (tokenized with the training chat-template handler), on one GPU.
  python3 odm_pad_equiv.py <model_dir> <parquet_glob> <n_examples> <batch> <patch_dir>
"""
import glob, sys, time
sys.path.insert(0, sys.argv[5])
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from fms_acceleration_odm.odm.reward import Reward, compute_reward

model_dir, pglob, n, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model_dir)
rows = pq.read_table(sorted(glob.glob(pglob))[0]).slice(0, n).to_pylist()
feats = []
for r in rows:
    ids = tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=True,
                                  enable_thinking=False)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    feats.append({"input_ids": list(ids), "attention_mask": [1] * len(ids), "labels": list(ids)})
lens = [len(f["input_ids"]) for f in feats]
print(f"examples={len(feats)} len min/mean/max={min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                             attn_implementation="eager").cuda().eval()
V = model.config.vocab_size
nopad = DataCollatorForSeq2Seq(tokenizer=tok, padding=False, max_length=8192)
rtok = AutoTokenizer.from_pretrained(model_dir); rtok.padding_side = "right"
pad = DataCollatorForSeq2Seq(tokenizer=rtok, padding=True, max_length=8192, return_tensors="pt")
ltok = AutoTokenizer.from_pretrained(model_dir); ltok.padding_side = "left"
lpad = DataCollatorForSeq2Seq(tokenizer=ltok, padding=True, max_length=8192, return_tensors="pt")

LIB_SUMS=[]; MY_SUMS=[]
def run(coll, b):
    out = []; torch.cuda.synchronize(); t = time.time()
    for i in range(0, len(feats), b):
        batch = coll([dict(f) for f in feats[i:i + b]])
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items() if k != "labels"}
        if batch["input_ids"].dim() == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        for j in range(batch["input_ids"].shape[0]):
            one = {k: v[j:j + 1] for k, v in batch.items()}
            if b == 1:
                out.append(compute_reward(model, one, V, Reward.ENTROPY))
        if b > 1:
            # per-example values from the batched forward (same math as compute_reward, summed)
            with torch.inference_mode():
                lg = model(**batch).logits[:, :-1, :].float()
                lp = torch.log_softmax(lg, -1); p = lp.exp()
                ent = -(p * lp).sum(-1); m = batch["attention_mask"][:, 1:]
                ent = (ent * m).sum(-1) / m.sum(-1)
                ent = (ent / torch.log(torch.tensor(float(V)))).clamp(0, 1)
            out += ent.tolist()
            total = compute_reward(model, batch, V, Reward.ENTROPY)  # what ODM actually sums
            LIB_SUMS.append(total); MY_SUMS.append(sum(ent.tolist()))
    torch.cuda.synchronize(); return out, time.time() - t

r1, t1 = run(nopad, 1)
# compute_reward on each padded row alone (same padded tensors, batch dim 1)
def lib_rows(coll, b):
    out=[]
    for i in range(0, len(feats), b):
        batch = coll([dict(f) for f in feats[i:i + b]])
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items() if k != "labels"}
        out.append(compute_reward(model, batch, V, Reward.ENTROPY))
    return out
rN, tN = run(pad, bs)
rL, tL = run(lpad, bs)
d = [abs(a - b) for a, b in zip(r1, rN)]
dl = [abs(a - b) for a, b in zip(r1, rL)]
print(f"vocab_size={V}")
print(f"bs=1 unpadded : sum={sum(r1):.6f} time={t1:.1f}s")
print(f"bs={bs} right-pad: sum={sum(rN):.6f} time={tN:.1f}s  max|diff|={max(d):.2e} mean|diff|={sum(d)/len(d):.2e}  speedup={t1/tN:.1f}x")
print(f"bs={bs} left-pad : sum={sum(rL):.6f}            max|diff|={max(dl):.2e} mean|diff|={sum(dl)/len(dl):.2e}")
lib_pad = lib_rows(pad, bs)
lib_b1 = [sum(r1[i:i+bs]) for i in range(0, len(r1), bs)]
dd = [abs(a-b) for a,b in zip(lib_b1, lib_pad)]
print(f"library compute_reward, per batch of {bs}: bs=1 sum vs right-padded batch max|diff|={max(dd):.3e} (per-example {max(dd)/bs:.2e})")
print(f"library total: bs=1 {sum(lib_b1):.4f}  right-padded {sum(lib_pad):.4f}")
print(f"float32 recompute vs library on padded batches: max|diff|={max(abs(a-b) for a,b in zip(MY_SUMS[:len(lib_pad)], lib_pad)):.3e}")
print("PASS" if max(dd)/bs < 5e-3 else "FAIL")

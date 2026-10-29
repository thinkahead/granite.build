"""2-GPU FSDP reward-pass timing: ODM entropy reward at eval batch 1 vs padded batch N.

Shards the model the way training does (FSDP FULL_SHARD, one unit per decoder layer, bf16,
flex attention) and runs fms-acceleration-odm's real compute_reward over real validation
examples. Each rank gets its own shard of the examples, as in training's distributed eval.

  torchrun --nproc_per_node=2 odm_reward_fsdp.py <model_dir> <data_root> <n_per_cat> <bs>
"""
import functools, glob, os, sys, time
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from fms_acceleration_odm.odm.reward import Reward, compute_reward

model_dir, data_root, n_per_cat, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
CATS = ["inventory_10102025_training_v2", "news_14102025_training_v3", "analysis_14102025_training_v10"]

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")

tok = AutoTokenizer.from_pretrained(model_dir)
def feats_for(cat):
    rows = pq.read_table(sorted(glob.glob(f"{data_root}/{cat}/data/*.parquet"))[0]).slice(0, n_per_cat).to_pylist()
    out = []
    for r in rows:
        ids = tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=True, enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        out.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids})
    return out[rank::world]  # this rank's shard

model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="flex_attention")
layer_cls = type(model.model.layers[0])
model = FSDP(model, auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}),
             sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=dev, use_orig_params=True,
             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16))
model.eval()
V = model.module.config.vocab_size

nopad = DataCollatorForSeq2Seq(tokenizer=tok, padding=False, max_length=8192)
rtok = AutoTokenizer.from_pretrained(model_dir); rtok.padding_side = "right"
pad = DataCollatorForSeq2Seq(tokenizer=rtok, padding=True, max_length=8192, return_tensors="pt")

def reward_pass(feats, coll, b):
    total = 0.0
    torch.cuda.synchronize(); dist.barrier(); t = time.time()
    # FSDP needs every rank to run the same number of forwards: pad the batch count.
    n_local = (len(feats) + b - 1) // b
    n = torch.tensor(n_local, device=dev); dist.all_reduce(n, op=dist.ReduceOp.MAX)
    for i in range(int(n)):
        chunk = feats[i * b:(i + 1) * b] or feats[:1]
        batch = coll([dict(f) for f in chunk])
        batch = {k: torch.as_tensor(v).to(dev) for k, v in batch.items() if k != "labels"}
        if batch["input_ids"].dim() == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        r = compute_reward(model, batch, V, Reward.ENTROPY)
        if i * b < len(feats):
            total += r
    torch.cuda.synchronize(); dist.barrier()
    return total, time.time() - t

# Warm up (flex attention compiles its kernels on first use).
reward_pass(feats_for(CATS[0])[:2], nopad, 1)
reward_pass(feats_for(CATS[0])[:2 * bs], pad, bs)

res = []
for cat in CATS:
    f = feats_for(cat)
    r1, t1 = reward_pass(f, nopad, 1)
    rN, tN = reward_pass(f, pad, bs)
    s = torch.tensor([r1, rN, float(len(f))], device=dev); dist.all_reduce(s)
    res.append((cat.split("_")[0], int(s[2]), s[0].item(), s[1].item(), t1, tN))

if rank == 0:
    print(f"model={os.path.basename(model_dir)} world={world} vocab_size={V} batch={bs} peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
    T1 = TN = 0
    for cat, n, r1, rN, t1, tN in res:
        T1 += t1; TN += tN
        print(f"{cat:10} n={n:4d}  reward/ex bs1={r1/n:.5f} bs{bs}={rN/n:.5f} |diff|={abs(r1-rN)/n:.2e}  "
              f"time bs1={t1:6.1f}s bs{bs}={tN:6.1f}s  speedup={t1/tN:4.1f}x")
    print(f"TOTAL time bs1={T1:.1f}s bs{bs}={TN:.1f}s speedup={T1/TN:.1f}x")
dist.destroy_process_group()

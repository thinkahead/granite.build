"""Batched ODM reward pass for padding-free training (fms-acceleration-odm 0.1.x).

With `padding_free`, fms-hf-tuning builds every collator, including ODM's reward-pass
(eval) collators, as `DataCollatorForSeq2Seq(padding=False)`. Unpadded examples of
different lengths can't be stacked, so `per_device_eval_batch_size` has to stay 1 and the
reward pass runs one example per forward: ~70 min per ODM update on granite-5.1-20b.

`compute_reward` already takes a masked mean per sequence and sums over the batch, so it
is batch-size independent as long as the batch is padded with an attention mask. This
module, imported via `sitecustomize` in each rank, swaps only the reward-pass collators
for padded ones and batches the reward pass; training stays padding-free.

Padding must be on the RIGHT. The reward masks shifted logits with attention_mask[:, 1:]:
with right padding that keeps exactly the L-1 next-token predictions an unpadded example
has, and causal attention means real tokens never see the pads, so their logits are
unchanged. With left padding (the granite tokenizers' default) it would add one term
computed at a pad position and shift every position id (different RoPE).

  ODM_REWARD_BATCH_SIZE=N   batch the reward pass N examples at a time (off when unset/1)
  ODM_REWARD_ENTROPY_CHUNK_TOKENS=C   reduce the entropy over the vocabulary C token
                            positions at a time (default 2048), bounding the reward's extra
                            memory whatever the batch shape
  ODM_VOCAB_SIZE=V          entropy normalization (default: the model's config.vocab_size)

Also replaces the hardcoded vocab_size=32000 with the model's `config.vocab_size`.
Logs `ODM_BATCHED_REWARD ...` once per rank when active.
"""

import copy
import os
import sys


def _chunked_entropy_reward(model, batch, vocab_size: int, reward_type: str) -> float:
    """Same value as fms_acceleration_odm.odm.reward.compute_reward for the ENTROPY
    rewards, but never materializes a full [batch, seq, vocab] probability tensor.

    Upstream does log_softmax / exp / p*log p / p*(log p)^2 over all positions at once:
    several [B, S, V] tensors at V = 100352 (~0.4 MB per token each in fp32), which a
    padded batch of long examples cannot fit next to the training state (build 32dae5df
    OOM'd on granite-5.1-20b). Here the model runs once on the whole batch and the entropy
    terms are reduced over the vocabulary `chunk` positions at a time, so the extra memory
    is bounded by `chunk` tokens whatever the batch shape. The forward and the number of
    forwards are the same on every rank, so FSDP collectives stay in lockstep.
    """
    # Third Party
    import torch
    import torch.nn.functional as F

    chunk = int(os.environ.get("ODM_REWARD_ENTROPY_CHUNK_TOKENS", "2048") or 2048)
    with torch.inference_mode():
        logits = model(**batch).logits[:, :-1, :]           # [B, S-1, V], model dtype
        bsz, slen, _ = logits.shape
        flat = logits.reshape(bsz * slen, -1)
        ent = torch.empty(bsz * slen, dtype=torch.float32, device=flat.device)
        var = torch.empty_like(ent)
        for i in range(0, flat.shape[0], chunk):
            lp = F.log_softmax(flat[i:i + chunk].float(), dim=-1)
            p = lp.exp()
            e = -(p * lp).sum(-1)
            ent[i:i + chunk] = e
            var[i:i + chunk] = (p * lp.pow(2)).sum(-1) - e.pow(2)
            del lp, p
        ent = ent.view(bsz, slen)
        var = var.view(bsz, slen)
        last = ent[:, -1]
        mask = batch["attention_mask"][:, 1:].to(ent.dtype)
        ent = (ent * mask).sum(-1) / mask.sum(-1)
        var = (var * mask).sum(-1) / mask.sum(-1)
    max_ent = torch.log(torch.tensor(float(vocab_size), dtype=ent.dtype, device=ent.device))
    ent = (ent / max_ent).clamp(0.0, 1.0)
    var = (var / max_ent ** 2).clamp(0.0, 1.0)
    last = (last / max_ent).clamp(0.0, 1.0)
    if reward_type == "entropy3_varent1":
        return 0.75 * ent.sum().item() + 0.25 * var.sum().item()
    if reward_type == "entropy_last_token":
        return last.sum().item()
    return ent.sum().item()


def _install() -> None:
    bs = int(os.environ.get("ODM_REWARD_BATCH_SIZE", "1") or 1)
    if bs <= 1:
        return
    # Third Party
    from fms_acceleration_odm.odm import dataset as ds
    from fms_acceleration_odm.odm import reward as rw
    from transformers import DataCollatorForSeq2Seq

    orig_reset = ds.OnlineMixingDataset._reset_eval_dataloaders
    orig_reward = rw.compute_reward

    def _reset_eval_dataloaders(self):
        if not getattr(self, "_odm_padded", False):
            padded = {}
            for k, c in (self.eval_collators_dict or {}).items():
                tok = getattr(c, "tokenizer", None)
                if tok is None:
                    raise RuntimeError(f"ODM_BATCHED_REWARD: collator for {k} has no tokenizer")
                tok = copy.copy(tok)  # don't change the shared tokenizer's padding side
                tok.padding_side = "right"
                padded[k] = DataCollatorForSeq2Seq(
                    tokenizer=tok, padding=True, max_length=getattr(c, "max_length", None),
                    return_tensors="pt",
                )
            self.eval_collators_dict = padded
            self.eval_batch_size = bs
            self._odm_padded = True
            print(f"ODM_BATCHED_REWARD rank={os.environ.get('RANK', '0')} "
                  f"eval_batch_size={bs} entropy_chunk_tokens="
                  f"{os.environ.get('ODM_REWARD_ENTROPY_CHUNK_TOKENS', '2048')} categories={len(padded)}",
                  file=sys.stderr, flush=True)
        return orig_reset(self)

    def compute_reward(model, batch, vocab_size, reward_type, **kw):
        v = os.environ.get("ODM_VOCAB_SIZE")
        if v:
            vocab_size = int(v)
        else:
            cfg = getattr(getattr(model, "module", model), "config", None)
            vocab_size = getattr(cfg, "vocab_size", None) or vocab_size
        if batch is None or not str(reward_type).lower().startswith("entropy"):
            return orig_reward(model=model, batch=batch, vocab_size=vocab_size,
                               reward_type=reward_type, **kw)
        batch = {k: t for k, t in batch.items() if k != "labels"}
        return _chunked_entropy_reward(model, batch, vocab_size, str(reward_type).lower())

    ds.OnlineMixingDataset._reset_eval_dataloaders = _reset_eval_dataloaders
    rw.compute_reward = compute_reward
    ds.compute_reward = compute_reward  # dataset.py imported it by name


try:
    _install()
except ImportError as e:
    # Every Python process (not just training ranks) loads sitecustomize; one without
    # fms_acceleration_odm (e.g. a setup helper) is fine. Say so, so a training rank that
    # silently didn't get the patch is visible in the log.
    print(f"ODM_BATCHED_REWARD not installed in pid {os.getpid()}: {e}", file=sys.stderr, flush=True)

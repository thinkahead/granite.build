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
  ODM_VOCAB_SIZE=V          entropy normalization (default: the model's config.vocab_size)

Also replaces the hardcoded vocab_size=32000 with the model's `config.vocab_size`.
Logs `ODM_BATCHED_REWARD ...` once per rank when active.
"""

import copy
import os
import sys


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
                  f"eval_batch_size={bs} categories={len(padded)}", file=sys.stderr, flush=True)
        return orig_reset(self)

    def compute_reward(model, batch, vocab_size, reward_type, **kw):
        v = os.environ.get("ODM_VOCAB_SIZE")
        if v:
            vocab_size = int(v)
        else:
            cfg = getattr(getattr(model, "module", model), "config", None)
            vocab_size = getattr(cfg, "vocab_size", None) or vocab_size
        if batch is not None and "labels" in batch:
            batch = {k: t for k, t in batch.items() if k != "labels"}
        return orig_reward(model=model, batch=batch, vocab_size=vocab_size,
                           reward_type=reward_type, **kw)

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

# eval-defense (BlueVela SLURM): evaluate a LoRA checkpoint

Merges a LoRA adapter into its base model, then runs the defense evals with vLLM, the
same way as Vela's `defence/granite_dot_build_enhanced/eval_g42nt_local.yaml`:
12 tool-calling categories plus 2 knowledge categories, one target each.

```
merge (lora-merge-slurm) ─┬─> inventory, equipment, military, organization, installations,
                          │   router, events, ships, nonstategroups, analysis, profiles, news
                          └─> knowledge, knowledge_with_tools      (KNOWLEDGE=true)
```

Steps: [`lora-merge-slurm`](../../../../configurations/assets/environments/skypilot/steps/lora-merge-slurm/step.yaml)
and [`defense-eval-slurm`](../../../../configurations/assets/environments/skypilot/steps/defense-eval-slurm/README.md).

## Run the evals on a checkpoint

From `~/granite.build`, with the standalone gbserver running (`~/gbserver-standalone/start.sh`).
Pass `ADAPTER_PATH` (the checkpoint) and `NAME` (where its scores go):

```bash
RUN=/proj/granite-build/g4os/defense/runs/cd2bfbc2-f7f0-4252-a32d-119ea2c30d58/sftlora_g42_3b_v2_odm_90_10_entropy_xml
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param ADAPTER_PATH=$RUN/checkpoint-3000 \
  --param NAME=eval_g42nt_lora_3b_cd2bfbc2_3000
```

- **Which dir to pass.** Any dir holding `adapter_config.json` + `adapter_model.safetensors`:
  a training run's `final/` or one of its `checkpoint-<N>/`. Use a full path on `/proj`.
- **Base model.** Defaults to `/proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.2-3b`
  (`BASE_MODEL_PATH`). It must be the base the adapter was trained on; that's
  `base_model_name_or_path` in its `adapter_config.json`.
- **A model that's already full** (full fine-tune, not LoRA): there's nothing to merge,
  so run just the eval targets (below). The merge target needs an adapter.

### Choosing what runs

The targets to run go after the options, as positional names. With none, all 15 run.

| Want | Add |
|---|---|
| Everything (merge + 14 categories) | nothing |
| Only the tool-calling categories | `--param KNOWLEDGE=false` |
| Only the knowledge evals | `merge knowledge knowledge_with_tools` |
| One category | `merge events` |

The `merge` target has to be named whenever eval targets are named, because they take
the model from its output. When the merged model already exists it's skipped (next
section), so naming it costs only a short job.

### Results

`/proj/granite-build/g4os/defense/evals/<NAME>/<category>/`:

| File | Contents |
|---|---|
| `result.jsonl` | global score (`subsets_mean`) + score per subset (trainingforeval / test) |
| `instance_scores.jsonl` | every prediction with its reference and per-instance metric |
| `eval.log` | full eval output, including the printed `Global Results` / `Subsets Results` |

Score table for a run (on a login node):

```bash
python3 summarize.py /proj/granite-build/g4os/defense/evals/<NAME>
```

`summarize.py` is next to this README; copy it over, since login nodes can't reach this
repo.

## When the merge with the base is done

**vLLM needs a full model, so every eval build starts with the `merge` target.** It runs
`peft merge_and_unload()` on base + adapter and writes bf16 safetensors with the
adapter's tokenizer and chat template. That's what Vela's `/gb-read-write/karve/lora_merge/<run>_ckpt<N>_merged`
dirs were, made by hand.

| | |
|---|---|
| **Where** | `MERGED_PATH`; by default `merged_<adapter dir name>` next to the adapter, e.g. `$RUN/merged_final/`, `$RUN/merged_checkpoint-3000/` |
| **When it runs** | first target of every eval build, before any category starts |
| **When it's skipped** | the target dir already holds `lora_merge_info.json` naming **this** adapter and base, and it's newer than the adapter's `adapter_model.safetensors`. The job then only logs `lora-merge: ... already the merge of ...; skipping` |
| **Force a redo** | `--param FORCE_MERGE=true`, e.g. after changing the base or the merge code |
| **Cost** | ~1 min on 1 H200 for the 3b (plus queue time); output ~7 GB per checkpoint |
| **Refuses** | adapters that added tokens or used `modules_to_save`, because a plain merge would drop those trained weights. This recipe's training config avoids both (see the training recipe's `tokenizer_name_or_path` note) |

So the first eval of a checkpoint merges it, and later evals of the same checkpoint, such
as re-running one category or the knowledge evals, reuse that merge. Merged dirs aren't
cleaned up automatically. Delete `$RUN/merged_*` when you no longer need them; the next
eval just re-merges.

The merge writes to a temp dir and renames it into place when it finishes, so an
interrupted merge never leaves a half-written model that a later run would skip to.

## Requirements

Details are in the `defense-eval-slurm` README.

- **GHE token** for the private `karve/fm-eval` clone: start gbserver with
  `~/gbserver-standalone/start.sh` (it passes `GITHUB_IBM_PAT`).
- **Knowledge evals only:**
  - The Lakehouse tables `knowledge_validation_set` and `knowledge_test_set`
    (`infusion.swim_language`), pulled to `KNOWLEDGE_DATA_PATH`, currently
    `/proj/granite-build/g4os/defense/<name>/<name>.parquet`.
  - The watsonx judge credentials in `WATSONX_CREDENTIALS_FILE`
    (`/proj/granite-build/g4os/defense/keys`, `NAME: <base64>` lines). They're decoded
    inside the job and never logged.
- **Patched SkyPilot** (CPU fix, see the `tuning-slurm` README). Without it, vLLM
  generation and scoring run on one core.

## Resources and timing

Each category: 2 × H200 on `gpu-high`, vLLM TP=2, as on Vela. The gpu-high QOS allows
8 GPUs at a time, so 4 categories run at once and the rest queue. The 12 tool-calling
categories took ~6–10 min each once running. Change the partition with
`--param QUEUE=gpu-mid --param QOS=""`.

## Runs

| Build | Model | What | Result |
|---|---|---|---|
| `70fd7a88` | `cd2bfbc2` `final/` | merge + 12 tool-calling | SUCCESS, mean 0.731 |
| `0ec74008` | `cd2bfbc2` `final/` | merge (skipped, reused) + 2 knowledge | SUCCESS, knowledge 0.633, knowledge_with_tools 0.452 |

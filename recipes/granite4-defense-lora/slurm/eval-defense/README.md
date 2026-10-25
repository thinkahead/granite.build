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

### granite-4.2-30b checkpoints

Same build, plus the 30b base. gpu-mid shown here (16-GPU cap); see "Which queue" below for
gpu-high and gpu-low:

```bash
RUN=/proj/granite-build/g4os/defense/runs/e8ff9ccf-33e4-4ddf-980f-a2369ebbd0cb/sftlora_g42_30b_v2_odm_90_10_entropy_xml
BASE30=/proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param ADAPTER_PATH=$RUN/checkpoint-3000 \
  --param BASE_MODEL_PATH=$BASE30 \
  --param NAME=eval_g42nt_lora_30b_e8ff9ccf_3000 \
  --param QUEUE=gpu-mid --param QOS=
```

Several checkpoints at once, one build each; they run in parallel. Builds return immediately,
and the IDs are printed:

```bash
for c in 1000 2000 3000 4000 5000; do
  GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
    -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
    --param ADAPTER_PATH=$RUN/checkpoint-$c --param BASE_MODEL_PATH=$BASE30 \
    --param NAME=eval_g42nt_lora_30b_e8ff9ccf_$c \
    --param QUEUE=gpu-mid --param QOS= \
    -m "defense eval: granite-4.2-30b LoRA e8ff9ccf checkpoint-$c"
done
```

- **30b sizing:**
  - **vLLM:** TP=2 on 2 H200 holds the 30b weights (~28 GB per GPU) with ~99 GB per GPU left
    for KV cache, so the default `NUM_GPUS_PER_NODE=2` / `H200:2` is fine.
  - **Merge:** it needs CPU RAM for the bf16 base (~60 GB) plus the saved copy, which is why
    `MERGE_MEMORY` defaults to 256 GB; 128 is enough for 3b.
- **The last checkpoint is `final/`.** `checkpoint-5000/adapter_model.safetensors` is
  byte-identical to `final/adapter_model.safetensors` for a 5000-step run, so evaluate one or
  the other, not both.
- **Disk:** each merged 30b checkpoint is ~55 GB (`$RUN/merged_checkpoint-<N>/`). Delete them
  after the evals if you're short of space; a later eval re-merges.

### Which queue: gpu-high, gpu-low, gpu-mid

Two parameters pick where the jobs run: `QUEUE`, the SLURM **partition** (what `squeue` shows in
`PARTITION`), and `QOS`, a separate scheduling limit. Each partition only accepts certain QOSes,
so set both together:

| Queue | Params | GPU limit for the account | Time limit | Priority / preemption | Use for |
|---|---|---|---|---|---|
| **gpu-high** (recipe default) | nothing (= `QUEUE=gpu-high QOS=grp_granite_dot_build_high_1n`) | **8 GPUs** total, i.e. 4 categories at 2 GPUs each | 2 days | highest tier (32768), `REQUEUE` | one eval build, starts fastest |
| **gpu-low** | `--param QUEUE=gpu-low --param QOS=burst` | **none** (`burst` has no GPU cap) | 30 days (default 1 day; recipe asks 6 h) | lowest tier (5), `REQUEUE`: higher tiers can preempt it | several eval builds in parallel, alongside training |
| **gpu-mid** | `--param QUEUE=gpu-mid --param QOS=` | **16 GPUs** total (`grp_granite_dot_build_medium`), shared with training | 7 days | middle tier (16384), `REQUEUE` | only when gpu-mid is otherwise idle |

**gpu-high** (default, nothing to add):

```bash
RUN=/proj/granite-build/g4os/defense/runs/cd2bfbc2-f7f0-4252-a32d-119ea2c30d58/sftlora_g42_3b_v2_odm_90_10_entropy_xml
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param ADAPTER_PATH=$RUN/checkpoint-4000 --param NAME=eval_g42nt_lora_3b_cd2bfbc2_4000
```

**gpu-low** (`burst` QOS):

```bash
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param ADAPTER_PATH=$RUN/checkpoint-3000 --param NAME=eval_g42nt_lora_3b_cd2bfbc2_3000 \
  --param QUEUE=gpu-low --param QOS=burst
```

Several checkpoints on gpu-low, one build each, all in parallel:

```bash
for c in 1000 2000; do
  GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
    -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
    --param ADAPTER_PATH=$RUN/checkpoint-$c --param NAME=eval_g42nt_lora_3b_cd2bfbc2_$c \
    --param QUEUE=gpu-low --param QOS=burst
done
```

For the 30b, add `--param BASE_MODEL_PATH=/proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm`
to any of these.

**Notes:**

- **gpu-low needs `QOS=burst` explicitly**: it's the only QOS gpu-low allows, and it isn't the
  account default. Verified on build `bb153c64` (2026-10-25): the merge started immediately as
  `gpu-low burst`.
- **gpu-low is preemptible by everything else.** An eval category that gets preempted fails; evals
  don't resume. Rerun just that category: `... merge <category>` with the same `NAME` (see
  "Choosing what runs"). Each category is only minutes, so the exposure is small.
- **`QOS=normal` does not work for this account.** The sbatch line goes through
  (`#SBATCH --qos=normal`), but BlueVela's job-submit plugin rewrites it to the account default,
  `grp_granite_dot_build_medium`, the same 16-GPU cap as `QOS=` (build `2c467ec4`).
- **The 16 gpu-mid GPUs are shared account-wide.** A 2-node training run needs all 16, so it can't
  start while gpu-mid evals hold any, and evals queue (`QOSGrpGRES`) behind it. Run evals on
  gpu-high or gpu-low while training is on gpu-mid.
- **gbserver limits start-ups, not running jobs.** At most 4 categories can be *starting up* at once,
  whatever the queue (`GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY`, default 4). A category holds a slot only
  from `sky.launch` until its SLURM job starts (1–3 min), then frees it for the next. So new eval jobs
  appear in SLURM a few at a time, but **any number can be running at once**; 8+ is normal. How many
  run together is set by the QOS GPU cap above. To start them faster, start gbserver with e.g.
  `GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY=8`; the cost is more parallel SSH logins to the BlueVela
  login nodes, which is why the default is 4.

Check which partition and QOS a job got, and why it's waiting:

```bash
squeue --me -o "%i %P %q %T %R %j" | grep eval-g42nt     # partition, QOS, state, pending reason
```

`(QOSGrpGRES)` means the QOS GPU cap is full; `(Priority)` / `(Resources)` mean the partition is busy.

### Merge only

To get a full merged model, e.g. for vLLM serving outside these evals, without running any
category, name just the `merge` target:

```bash
GB_ENVIRONMENT=STANDALONE PATH=$HOME/.local/bin:$PATH .venv/bin/gb build start \
  -f recipes/granite4-defense-lora/slurm/eval-defense/build.yaml --skip-version-check \
  --param ADAPTER_PATH=$RUN/checkpoint-3000 --param BASE_MODEL_PATH=$BASE30 \
  --param NAME=merge_30b_e8ff9ccf_3000 --param QUEUE=gpu-mid --param QOS= \
  merge
```

That writes `$RUN/merged_checkpoint-3000/`, or `--param MERGED_PATH=/proj/...` to put it
elsewhere. The model is bf16 safetensors with the adapter's tokenizer and chat template, plus
`lora_merge_info.json` recording which adapter and base it came from.

### Checking on an eval

```bash
# build status (from ~/granite.build, on the gbserver host)
GB_ENVIRONMENT=STANDALONE .venv/bin/gb build status <build_id> --skip-version-check
# SLURM jobs for it (on a BlueVela login node); target names are in the job names
squeue --me -o "%i %P %T %M %j" | grep eval-g42nt
# a category's live output
tail -F /proj/granite-build/g4os/defense/evals/<NAME>/<category>/eval.log
```

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

Each category: 2 × H200, vLLM TP=2, as on Vela. How many **run** at once is limited only by the
QOS GPU cap (see "Which queue": 4 categories on gpu-high, 8 on gpu-mid, no cap on gpu-low).
Separately, gbserver lets at most 4 categories **start up** at a time
(`GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY`, default 4). That only staggers how quickly jobs reach
SLURM, not how many run. The 12 tool-calling categories take ~6–10 min each on 3b once running,
longer on 30b.

## Checkpoint comparison: granite-4.2-30b LoRA (`e8ff9ccf`)

Five checkpoints of the 5000-step 30b run (2 nodes, effective batch 80, eval/ODM every 1000),
each evaluated on all 14 categories with this recipe on 2026-10-25. The 3b final (`cd2bfbc2`,
effective batch 40) is shown for reference. Scores are `subsets_mean`, the mean of the
trainingforeval and test subsets. Tool-calling categories are scored by `DefenseToolCallingMetric`;
knowledge categories by the watsonx Llama-3.3-70B judge. **Bold** = best 30b checkpoint.

| Category | 3b final | 30b 1000 | 30b 2000 | 30b 3000 | 30b 4000 | 30b 5000 (= final) |
|---|---|---|---|---|---|---|
| inventory | 0.938 | 0.867 | 0.917 | 0.935 | 0.951 | **0.952** |
| router | 0.931 | 0.885 | 0.932 | 0.929 | 0.938 | **0.941** |
| nonstategroups | 0.884 | 0.820 | 0.865 | 0.886 | **0.909** | 0.906 |
| news | 0.832 | 0.746 | 0.797 | **0.864** | 0.851 | 0.848 |
| profiles | 0.802 | 0.768 | 0.746 | **0.811** | 0.769 | 0.771 |
| analysis | 0.773 | 0.709 | 0.753 | **0.782** | 0.763 | 0.760 |
| military | 0.755 | 0.698 | 0.768 | 0.812 | 0.819 | **0.825** |
| organization | 0.708 | 0.698 | 0.697 | 0.716 | 0.737 | **0.737** |
| installations | 0.691 | 0.655 | 0.663 | **0.705** | 0.689 | 0.697 |
| ships | 0.656 | **0.706** | 0.698 | 0.697 | 0.700 | 0.703 |
| equipment | 0.575 | **0.626** | 0.549 | 0.544 | 0.572 | 0.563 |
| events | 0.230 | **0.240** | 0.227 | 0.219 | 0.225 | 0.213 |
| knowledge | 0.633 | 0.653 | 0.667 | 0.666 | **0.700** | 0.653 |
| knowledge_with_tools | 0.452 | 0.441 | 0.425 | **0.461** | 0.425 | 0.433 |
| **Mean, 12 tool-calling** | 0.731 | 0.702 | 0.718 | 0.742 | **0.744** | 0.743 |
| **Mean, 2 knowledge** | 0.543 | 0.547 | 0.546 | **0.564** | 0.563 | 0.543 |
| **Mean, all 14** | 0.704 | 0.680 | 0.693 | 0.716 | **0.718** | 0.714 |

Held-out **test** subset only (the trainingforeval subset is drawn from the training
distribution, so test is the cleaner generalisation signal):

| Category | 3b final | 30b 1000 | 30b 2000 | 30b 3000 | 30b 4000 | 30b 5000 |
|---|---|---|---|---|---|---|
| inventory | 0.926 | 0.862 | 0.926 | 0.922 | **0.949** | **0.949** |
| router | 0.912 | 0.872 | **0.912** | 0.892 | 0.901 | 0.896 |
| nonstategroups | 0.819 | 0.818 | 0.811 | 0.829 | **0.853** | 0.849 |
| news | 0.737 | 0.761 | 0.707 | **0.805** | 0.776 | 0.771 |
| profiles | 0.771 | **0.812** | 0.688 | 0.792 | 0.708 | 0.708 |
| analysis | 0.731 | 0.709 | 0.716 | **0.754** | 0.709 | 0.701 |
| military | 0.682 | 0.750 | 0.757 | **0.764** | 0.743 | 0.743 |
| organization | 0.612 | **0.674** | 0.620 | 0.636 | 0.667 | 0.667 |
| installations | 0.612 | 0.646 | 0.599 | **0.650** | 0.603 | 0.620 |
| ships | 0.579 | **0.682** | 0.664 | 0.654 | 0.664 | 0.664 |
| equipment | 0.609 | **0.703** | 0.578 | 0.562 | 0.609 | 0.594 |
| events | 0.134 | **0.138** | 0.130 | 0.121 | 0.134 | 0.121 |
| knowledge | 0.653 | 0.656 | 0.651 | 0.638 | **0.683** | 0.679 |
| knowledge_with_tools | 0.424 | **0.519** | 0.426 | 0.416 | 0.434 | 0.442 |

**Reading it:**

- **Peaks at 3000–4000, then flat.** Tool-calling mean: 0.702 (1000) → 0.718 → 0.742 →
  **0.744** (4000) → 0.743 (5000); all-14 mean peaks at 0.718 (4000). The 3000/4000/5000
  differences are within noise: paired across the 12 categories, 4000 vs 3000 is +0.002
  (t = 0.3) and 5000 vs 3000 is +0.001 (t = 0.2). Per-category sampling error is
  ~0.006–0.012, larger for the small sets.
- **No clear overfitting penalty yet.** Training loss fell to 0.004 by step 5000, but held-out
  scores don't drop: 5000 matches 4000 on most categories. A few test splits are best at
  1000 and fall after it (equipment 0.703 → 0.594, ships 0.682 → 0.664, profiles 0.812 →
  0.708, knowledge_with_tools 0.519 → 0.442). Those are the clearest signs of over-training, so
  watch them if training longer.
- **30b vs 3b: small gain.** From 3000 on, the 30b beats the 3b final by about +0.011 on the
  tool-calling mean (up in 7–8 of 12 categories, paired t about 1.4–1.6, so suggestive, not
  conclusive). The 30b gains most on military (+0.06–0.07) and nonstategroups and inventory
  (+0.02); it is lower on profiles and equipment. Neither run was tuned for this comparison:
  different effective batch (80 vs 40) and eval/ODM interval.
- **Events is the weak category for both models** (≈0.22 overall, ≈0.13 test). It doesn't move
  with model size or steps, so it's a data or label-matching issue, not capacity. Spot-checking
  the 3b's `events/instance_scores.jsonl`, every prediction is a well-formed tool call; the
  misses are details the metric scores as wrong: a missing `endDate` on a date range,
  `"Air-launched"` vs `"Air-Launched"`, or a different filter field (`eventCategory` /
  `primaryCountry` where the reference uses `pmesii` / `eventAddress`).
- **Recommendation:** use **checkpoint-4000** (best tool-calling and all-14 mean, tied within
  noise with 3000 and 5000), or `final/` if one model is wanted without a choice. A 3000-step
  run at this batch would likely match 5000 for ~40% less compute.

Results: `/proj/granite-build/g4os/defense/evals/eval_g42nt_lora_30b_e8ff9ccf_<step>/<category>/`.
Builds: `ffdb7ee2` (1000), `cdd03325` (2000), `24a99365` (3000), `71d48c77` (4000),
`ae9f49bb` (5000). Three categories hit bad nodes (`p3-r10-n2`, `p1-r30-n3`) and were rerun
on other nodes: `bece7095` (1000 installations), `b2510fbd` (3000 military), `14329026`
(4000 knowledge_with_tools). Those nodes are now in `EXCLUDE_NODES`.

## Runs

| Build | Model | What | Result |
|---|---|---|---|
| `70fd7a88` | `cd2bfbc2` `final/` | merge + 12 tool-calling | SUCCESS, mean 0.731 |
| `0ec74008` | `cd2bfbc2` `final/` | merge (skipped, reused) + 2 knowledge | SUCCESS, knowledge 0.633, knowledge_with_tools 0.452 |
| `ffdb7ee2`, `cdd03325`, `24a99365`, `71d48c77`, `ae9f49bb` (+ reruns `bece7095`, `b2510fbd`, `14329026`) | 30b `e8ff9ccf` checkpoints 1000–5000 | merge + 14 categories each | SUCCESS; see "Checkpoint comparison" |

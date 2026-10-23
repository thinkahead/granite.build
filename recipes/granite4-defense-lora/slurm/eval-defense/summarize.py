#!/usr/bin/env python3
"""Print a score table for an eval-defense run.

Usage (on a BlueVela login node):
  python3 summarize.py /proj/granite-build/g4os/defense/evals/<NAME>

Reads <dir>/<category>/result.jsonl written by the defense-eval-slurm step and prints,
per category, the subsets_mean score and each subset (trainingforeval / test) score.
"""

import json
import os
import sys


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows, missing = [], []
    for cat in sorted(os.listdir(root)):
        path = os.path.join(root, cat, "result.jsonl")
        if not os.path.isfile(path):
            if os.path.isdir(os.path.join(root, cat)):
                missing.append(cat)
            continue
        with open(path, encoding="utf-8") as f:
            r = json.loads(f.readline())
        g = r.get("global_scores") or {}
        subs = {k: v for k, v in (r.get("subsets_scores") or {}).items()
                if k not in ("score", "score_name", "num_of_instances")}
        sub_scores = {k: (v.get("score") if isinstance(v, dict) else v) for k, v in subs.items()}
        rows.append((cat, g.get("score"), g.get("num_of_instances"), sub_scores))

    print(f"{'category':16s} {'score':>7s} {'n':>6s}  subsets")
    for cat, score, n, subs in rows:
        s = " ".join(f"{k}={v:.3f}" for k, v in subs.items() if isinstance(v, (int, float)))
        print(f"{cat:16s} {score:7.3f} {n:6d}  {s}")
    if rows:
        mean = sum(r[1] for r in rows) / len(rows)
        print(f"{'MEAN':16s} {mean:7.3f}        ({len(rows)} categories)")
    if missing:
        print(f"no result yet: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

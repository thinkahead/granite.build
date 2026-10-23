"""Local stand-in for lh_eval_api (github.ibm.com/karve/lakehouse-eval-api) on BlueVela.

The defense eval scripts import exactly one thing from lh_eval_api,
`upload_unitxt_results_to_lh`, and call `.save_url_to_json(<OUTPUT_PATH>/result.jsonl)`
on what it returns. On Vela that pushed scores to the Lakehouse; BlueVela compute nodes
have no LAKEHOUSE_TOKEN and no route to it. This writes the same scores to disk instead:

  <OUTPUT_PATH>/result.jsonl         one JSON line: model, benchmark, global + subset scores
  <OUTPUT_PATH>/instance_scores.jsonl per-instance scores (prediction, reference, score)

Everything before the upload (inference, metric, printed "Global Results" /
"Subsets Results") is the unmodified fm-eval code path.
"""

import json
import math
import os

_RESULT_KEYS = ("model", "benchmark", "global_scores", "subsets_scores")


def _jsonable(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    try:
        return _jsonable(o.item())  # numpy scalars
    except Exception:  # pylint: disable=broad-except
        return str(o)


class _LocalUploadTrace:
    def __init__(self, record, instances):
        self._record = record
        self._instances = instances

    def save_url_to_json(self, path):
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._record) + "\n")
        inst_path = os.path.join(out_dir, "instance_scores.jsonl")
        with open(inst_path, "w", encoding="utf-8") as f:
            for row in self._instances:
                f.write(json.dumps(row) + "\n")
        print(f"lh_eval_api(local): wrote {path} and {inst_path} ({len(self._instances)} instances)")


def upload_unitxt_results_to_lh(
    unitxt_evaluation_results,
    model,
    benchmark,
    dataset=None,
    experiment_tags=None,
    is_official=False,
    predictions_namespace=None,
    run_params_to_include=None,
):
    """Same signature as lh_eval_api's; returns an object with save_url_to_json()."""
    res = unitxt_evaluation_results
    record = {
        "model": model if isinstance(model, str) else type(model).__name__,
        "benchmark": benchmark,
        "dataset": dataset,
        "run_params": _jsonable(run_params_to_include or {}),
        "global_scores": _jsonable(dict(res.global_scores)),
        "subsets_scores": _jsonable(dict(res.subsets_scores)),
        "num_instances": len(res),
        "uploaded_to_lakehouse": False,
    }
    instances = []
    for i, inst in enumerate(res):
        score = (inst.get("score") or {}).get("instance", {})
        instances.append(_jsonable({
            "idx": i,
            "subset": inst.get("subset"),
            "source": inst.get("source"),
            "prediction": inst.get("prediction"),
            "references": inst.get("references"),
            "processed_prediction": inst.get("processed_prediction"),
            "score": score,
        }))
    return _LocalUploadTrace(record, instances)


def _find_local_table(name, split):
    """Locate a pulled Lakehouse table under the DATA_PATH roots.

    Accepts both layouts seen on BlueVela: <root>/<name>/<split>.parquet (the
    defense_evaldata pulls) and <root>/<name>/<name>.parquet (a direct table export).
    KNOWLEDGE_DATA_PATH is searched first, then DATA_PATH; both may hold ':'-separated
    roots.
    """
    roots = []
    for var in ("KNOWLEDGE_DATA_PATH", "DATA_PATH"):
        roots += [r for r in os.environ.get(var, "").split(":") if r]
    for root in roots:
        for fname in (f"{split}.parquet", f"{name}.parquet"):
            fp = os.path.join(root, name, fname)
            if os.path.isfile(fp):
                return fp
    raise FileNotFoundError(
        f"lh_eval_api(local): table {name!r} not found as <root>/{name}/{split}.parquet or "
        f"<root>/{name}/{name}.parquet under {roots}")


def _make_lakehouse_loader():
    from typing import Optional

    import pyarrow.parquet as pq
    from unitxt.loaders import LoadFromDictionary

    class LakeHouseLoader(LoadFromDictionary):
        """Local LakeHouseLoader: same fields as lh_eval_api's, reads parquet from DATA_PATH."""

        name: str = None
        namespace: str = None
        force_download: bool = False
        default_split: Optional[str] = None
        data: dict = None

        def prepare(self):
            split = self.default_split or "test"
            fp = _find_local_table(self.name, split)
            self.data = {split: pq.read_table(fp).to_pylist()}
            print(f"lh_eval_api(local): {self.namespace}.{self.name} <- {fp} "
                  f"({len(self.data[split])} rows, split={split})")
            super().prepare()

    return LakeHouseLoader


# Defined at import (not lazily): the plain knowledge eval loads cards whose loader is
# "__type__": "lake_house_loader" but never imports LakeHouseLoader itself, and unitxt
# resolves that type only if the class was registered (by subclass creation) first.
LakeHouseLoader = _make_lakehouse_loader()


__all__ = ["upload_unitxt_results_to_lh", "LakeHouseLoader"]

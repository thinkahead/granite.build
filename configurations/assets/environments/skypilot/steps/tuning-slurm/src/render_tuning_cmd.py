#!/usr/bin/env python3
"""Render the `accelerate launch -m tuning.sft_trainer` command for tuning-slurm.

The SLURM counterpart of the k8s tuning step's helm helpers (tplAccelerateFlags /
tplSftTrainerConfigFilled) and the LSF step's process_values.py: the same flag
conventions, rendered from the step config the run script dumps as JSON. Output is
a bash script with every argument shlex-quoted, so config values never pass
through shell word-splitting.

Flag conventions (identical to the k8s/LSF tuning step):
  * tuning_config: `--key=value`; a one-letter key is `-k`, except LoRA's `r`,
    which sft_trainer spells `--r`. Lists expand to `--key v1 v2 ...`; dicts are
    passed as JSON; None is dropped.
  * fsdp_config: only when more than one GPU is in play and use_fsdp is "true";
    `use_fsdp` becomes the bare `--use_fsdp`, every other key `--key=value`.
  * accelerate_config: `--key=value`; a boolean true is the bare `--key`, false
    is `--key=false`. `use_parallelism_config` is emitted bare only when "true".
"""

import argparse
import json
import shlex
import sys


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def sft_flags(tuning_config: dict) -> list:
    out = []
    for k, v in tuning_config.items():
        if v is None:
            continue
        flag = f"--{k}" if (len(k) > 1 or k == "r") else f"-{k}"
        if isinstance(v, list):
            out.append(flag)
            out.extend(_scalar(x) for x in v)
        else:
            out.append(f"{flag}={_scalar(v)}")
    return out


def fsdp_flags(fsdp_config: dict, total_gpus: int) -> list:
    if total_gpus <= 1 or not _is_true(fsdp_config.get("use_fsdp", "false")):
        return []
    out = []
    for k, v in fsdp_config.items():
        if k == "use_fsdp":
            out.append("--use_fsdp")
        elif v is not None:
            out.append(f"--{k}={_scalar(v)}")
    return out


def accelerate_flags(accelerate_config: dict) -> list:
    out = []
    for k, v in accelerate_config.items():
        if v is None:
            continue
        flag = f"--{k}" if len(k) > 1 else f"-{k}"
        if k == "use_parallelism_config":
            if _is_true(v):
                out.append(flag)
        elif isinstance(v, bool):
            out.append(flag if v else f"{flag}=false")
        elif isinstance(v, list):
            out.append(flag)
            out.extend(_scalar(x) for x in v)
        else:
            out.append(f"{flag}={_scalar(v)}")
    return out


# Topology and paths belong to the step, not the build: they come from the
# allocation and the per-run workdir, so a build value would only disagree.
STEP_OWNED_ACCELERATE = (
    "num_processes",
    "num_machines",
    "machine_rank",
    "main_process_ip",
    "main_process_port",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-json", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, required=True)
    p.add_argument("--node-rank", type=int, required=True)
    p.add_argument("--master-ip", required=True)
    p.add_argument("--master-port", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-model-dir", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--data-config", default="")
    p.add_argument("--controller-config", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    with open(a.config_json, encoding="utf-8") as f:
        cfg = json.load(f)
    tuning_config = dict(cfg.get("tuning_config") or {})
    fsdp_config = dict(cfg.get("fsdp_config") or {})
    accelerate_config = dict(cfg.get("accelerate_config") or {})

    for k in STEP_OWNED_ACCELERATE:
        if accelerate_config.pop(k, None) is not None:
            print(f"tuning-slurm: ignoring accelerate_config.{k}; it comes from the allocation", file=sys.stderr)
    for k in ("output_dir", "save_model_dir", "experiment", "data_config_path",
              "trainer_controller_config_file"):
        if tuning_config.pop(k, None) is not None:
            print(f"tuning-slurm: ignoring tuning_config.{k}; the step sets it", file=sys.stderr)
    if a.model:
        tuning_config.setdefault("model_name_or_path", a.model)
    if not tuning_config.get("model_name_or_path"):
        print("tuning-slurm: FATAL: no model. Bind the model_to_tune input or set "
              "tuning_config.model_name_or_path.", file=sys.stderr)
        return 2
    if a.data_config:
        tuning_config["data_config_path"] = a.data_config
    if a.controller_config:
        tuning_config["trainer_controller_config_file"] = a.controller_config
    tuning_config["output_dir"] = a.output_dir
    tuning_config["save_model_dir"] = a.save_model_dir
    tuning_config["experiment"] = a.experiment

    total_gpus = a.nodes * a.gpus_per_node
    argv = [a.python, "-m", "accelerate.commands.launch",
            f"--num_processes={total_gpus}",
            f"--num_machines={a.nodes}",
            f"--machine_rank={a.node_rank}"]
    if a.nodes > 1:
        argv += [f"--main_process_ip={a.master_ip}", f"--main_process_port={a.master_port}"]
    argv += fsdp_flags(fsdp_config, total_gpus)
    argv += accelerate_flags(accelerate_config)
    argv += ["-m", "tuning.sft_trainer"]
    argv += sft_flags(tuning_config)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(" \\\n    ".join(shlex.quote(x) for x in argv) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

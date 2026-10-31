#!/usr/bin/env python3
"""Run an eval script so vLLM's spawned workers don't re-execute its top-level code.

Usage: run_as_main.py <script.py> [args...]

The defense_knowledge scripts run everything at module level (no __main__ guard). When
vLLM starts its engine with the `spawn` start method, multiprocessing re-imports the
parent's main module in the child (as `__mp_main__`), which re-runs the whole script and
aborts with "An attempt has been made to start a new process before the current process
has finished its bootstrapping phase". vLLM >= 0.30 forces spawn whenever CUDA is already
initialized, so VLLM_WORKER_MULTIPROC_METHOD=fork no longer avoids it.

  * Script WITH an `if __name__ == "__main__":` guard (the tool-calling scripts): run it as
    __main__, exactly as `python <script>` would. Its guard already protects the spawn child.
  * Script WITHOUT one (the knowledge scripts): run it under another module name, so the
    spawn child's main module is this wrapper (whose body is guarded) and the eval isn't
    re-run in it.

sys.argv[0] and __file__ are the script in both cases, so Path(__file__)-relative lookups
in it still resolve.

Tracebacks: the eval scripts end in a `finally:` that reaps vLLM's workers and calls
os._exit(), which ends the process before Python prints the propagating exception, so a
failed eval logged only "exit code 1". os._exit is wrapped here to print the in-flight
exception (if any) first, then exit with the same code.
"""

import os
import re
import runpy
import sys
import traceback

GUARD = re.compile(r"""^if\s+__name__\s*==\s*["']__main__["']\s*:""", re.M)

_real_exit = os._exit


def _exit_with_traceback(code: int) -> None:
    exc = sys.exc_info()
    if exc[0] is not None:
        print("defense-eval: exception at exit (printed before os._exit):", file=sys.stderr)
        traceback.print_exception(*exc, file=sys.stderr)
        sys.stderr.flush()
    _real_exit(code)


def main() -> None:
    os._exit = _exit_with_traceback
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    with open(script, encoding="utf-8") as f:
        guarded = bool(GUARD.search(f.read()))
    runpy.run_path(script, run_name="__main__" if guarded else "_eval_main")


if __name__ == "__main__":
    main()

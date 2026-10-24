#!/usr/bin/env python3
"""Stream a trainer log into a readable copy without progress-bar redraws.

  clean_log.py <in> <out> --stop-file <f>   # follow <in>; stop after EOF once <f> exists
  clean_log.py <in> <out> --once            # clean what is there now and exit

Progress bars dominate the raw log: tqdm redraws in place with a carriage return, and
multi-rank bars (ODM's per-item "Reward computation over eval dataset": 547k frames on
the 5000-step 3b run) also arrive as separate, interleaved lines. This keeps every
non-bar line and only the latest frame of each bar between them, so each bar appears
once with its final count; lone cursor-up lines are dropped. On the 5000-step 3b run:
57 MB / 634k lines raw -> 1.2 MB / 9.2k lines, with every loss, eval, checkpoint and
train_runtime line kept. Output is flushed as it is written, so the file is readable
while training runs. The raw log is untouched; this only writes <out>.
"""

import argparse
import os
import re
import sys
import time

CR = "\r"
NL = "\n"
CURSOR_UP = "\x1b[A"
# A tqdm frame: optional "<desc>:" then "NN%|<bar>| n/total [elapsed<remaining, rate]".
BAR = re.compile(r"^(?P<desc>[^|]*?)\s*\d+%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*\]\s*$")


def last_frame(line):
    """The final redraw of a line that holds carriage-return redraws."""
    frames = [f for f in line.split(CR) if f.strip()]
    return frames[-1] if frames else ""


class Collapser:
    """Writes lines, keeping only the latest frame of each progress bar.

    Every rank draws its own bars, so frames of different bars interleave. A bar's
    latest frame is held (one per bar description) and written once a non-bar line
    arrives, i.e. at the end of each run of bar output.
    """

    def __init__(self, dst):
        self.dst = dst
        self.held = {}  # bar description -> latest frame, in first-seen order

    def line(self, text):
        bare = text.replace(CURSOR_UP, "").strip()
        if not bare:
            return  # a lone cursor-up / blank redraw line
        m = BAR.match(text.replace(CURSOR_UP, ""))
        if m:
            self.held[m.group("desc").strip()] = text
            return
        self.flush_held()
        self.dst.write(text + NL)

    def flush_held(self):
        for frame in self.held.values():
            self.dst.write(frame + NL)
        self.held.clear()


def feed(src, out, pending=""):
    """Feed the complete lines readable now to `out`; return the trailing partial line."""
    while True:
        chunk = src.read(1 << 16)
        if not chunk:
            return pending
        *lines, pending = (pending + chunk).split(NL)
        for line in lines:
            out.line(last_frame(line) if CR in line else line)
        out.dst.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--once", action="store_true")
    p.add_argument("--stop-file", default=None)
    a = p.parse_args()

    while not os.path.exists(a.src):
        if a.once:
            print(f"clean_log: {a.src} does not exist", file=sys.stderr)
            return 1
        time.sleep(2)

    with open(a.src, encoding="utf-8", errors="replace", newline="") as src, \
            open(a.dst, "w", encoding="utf-8") as dst:
        out = Collapser(dst)
        pending = feed(src, out)
        while not a.once and not (a.stop_file and os.path.exists(a.stop_file)):
            time.sleep(5)
            pending = feed(src, out, pending)
        pending = feed(src, out, pending)
        if last_frame(pending):
            out.line(last_frame(pending))
        out.flush_held()
        dst.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Exact-string, in-place file patch for setup_commands (no perl/sed dependency).

  patch_file.py <file> <old> <new> [--count N | --min N] [--skip-if-present TEXT]

Replaces every occurrence of <old> with <new> in <file>. Fails (exit 1) unless the
number of occurrences found satisfies --count N (exactly N, default) or --min N, so an
upstream change that moves the target fails the run instead of silently skipping the
patch. --skip-if-present makes it idempotent: if TEXT is already in the file, nothing is
changed and it exits 0. Backslash escapes (\\n, \\t) in <old>/<new> are decoded.

Added because quay.io/modh/fms-hf-tuning:v3.2.2 ships without perl.
"""

import argparse
import codecs
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("old")
    p.add_argument("new")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--count", type=int, default=None)
    g.add_argument("--min", type=int, default=None)
    p.add_argument("--skip-if-present", default=None)
    a = p.parse_args()

    old = codecs.decode(a.old, "unicode_escape")
    new = codecs.decode(a.new, "unicode_escape")
    with open(a.file, encoding="utf-8") as f:
        text = f.read()
    if a.skip_if_present is not None and codecs.decode(a.skip_if_present, "unicode_escape") in text:
        print(f"patch_file: {a.file}: already patched; skipping")
        return 0
    n = text.count(old)
    if a.min is not None:
        ok, want = n >= a.min, f">= {a.min}"
    else:
        want_n = 1 if a.count is None else a.count
        ok, want = n == want_n, f"== {want_n}"
    if not ok:
        print(f"patch_file: FATAL: {a.file}: found {n} occurrence(s) of the target, expected {want}",
              file=sys.stderr)
        return 1
    with open(a.file, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    print(f"patch_file: {a.file}: replaced {n} occurrence(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

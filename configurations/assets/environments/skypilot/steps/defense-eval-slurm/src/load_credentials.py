#!/usr/bin/env python3
"""Turn a `NAME: <base64 value>` file into shell `export` lines.

Usage: load_credentials.py <file>  > env.sh ; set -a; . env.sh; set +a

Prints the exports to stdout (meant to go straight into a 0600 temp file that is
sourced and deleted) and only the key NAMES to stderr, so values never reach the
build log. Also derives the alias names the watsonx clients read, without
overwriting any name the file sets itself:
  unitxt WML engine:   WML_URL, WML_APIKEY, WML_PROJECT_ID
  litellm (judge):     WATSONX_URL, WATSONX_API_BASE, WATSONX_APIKEY, WATSONX_PROJECT_ID,
                       WX_API_KEY, WX_PROJECT_ID
"""

import base64
import re
import shlex
import sys

LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S+)\s*$")

# alias -> source names, first present wins
ALIASES = {
    "WML_URL": ("WML_URL", "WATSONX_API_URL", "WATSONX_URL"),
    "WML_APIKEY": ("WML_APIKEY", "WATSONX_API_KEY", "WATSONX_APIKEY"),
    "WML_PROJECT_ID": ("WML_PROJECT_ID", "WATSONX_PROJECT_ID"),
    "WATSONX_URL": ("WATSONX_URL", "WATSONX_API_URL", "WML_URL"),
    "WATSONX_API_BASE": ("WATSONX_API_BASE", "WATSONX_API_URL", "WML_URL"),
    "WATSONX_APIKEY": ("WATSONX_APIKEY", "WATSONX_API_KEY", "WML_APIKEY"),
    "WATSONX_API_KEY": ("WATSONX_API_KEY", "WATSONX_APIKEY", "WML_APIKEY"),
    "WATSONX_PROJECT_ID": ("WATSONX_PROJECT_ID", "WML_PROJECT_ID"),
    "WX_API_KEY": ("WX_API_KEY", "WATSONX_API_KEY", "WML_APIKEY"),
    "WX_PROJECT_ID": ("WX_PROJECT_ID", "WATSONX_PROJECT_ID", "WML_PROJECT_ID"),
}


def main() -> int:
    vals = {}
    with open(sys.argv[1], encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = LINE.match(line)
            if not m:
                print(f"load_credentials: line {n}: expected 'NAME: <base64>'", file=sys.stderr)
                return 2
            name, b64 = m.groups()
            try:
                vals[name] = base64.b64decode(b64, validate=True).decode("utf-8").strip()
            except Exception:  # pylint: disable=broad-except
                print(f"load_credentials: line {n} ({name}): not valid base64", file=sys.stderr)
                return 2
    derived = {}
    for alias, sources in ALIASES.items():
        if alias in vals:
            continue
        for src in sources:
            if src in vals:
                derived[alias] = vals[src]
                break
    for k, v in {**vals, **derived}.items():
        print(f"export {k}={shlex.quote(v)}")
    print(f"load_credentials: set {sorted(vals)}; derived {sorted(derived)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

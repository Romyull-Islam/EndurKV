#!/usr/bin/env bash
# Phase D — thin wrapper that delegates to scripts/10_run_study.py.
# (We rewrote the bash JSON-in-pipe loop as Python; bash heredoc + pipe
# stdin-aliasing made the shell version brittle.)
set -e
cd "$(dirname "$0")/.."
exec python3 scripts/10_run_study.py "$@"

#!/usr/bin/env bash
set -euo pipefail
python scripts/00_audit_and_split.py
python scripts/01_preflight.py --data-only

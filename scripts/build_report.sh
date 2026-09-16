#!/usr/bin/env bash
# Build report/main.pdf with tectonic.
set -euo pipefail
cd "$(dirname "$0")/.."
tectonic report/main.tex --outdir report
echo "Built report/main.pdf"

#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  echo "Expected rf-mcp virtual environment at $project_dir/.venv" >&2
  exit 1
fi

sudo apt update
sudo apt install -y git libsndfile1

"$project_dir/.venv/bin/python" -m pip install --upgrade \
  "git+https://github.com/colaclanth/sstv.git"

if [[ ! -x "$project_dir/.venv/bin/sstv" ]]; then
  echo "SSTV decoder installation did not create $project_dir/.venv/bin/sstv" >&2
  exit 1
fi

"$project_dir/.venv/bin/sstv" --help >/dev/null
echo "SSTV WAV decoder is ready. Restart rf-mcp before use."

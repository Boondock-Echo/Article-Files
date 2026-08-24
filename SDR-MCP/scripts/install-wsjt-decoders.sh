#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y wsjtx

missing=0
for decoder in jt9 wsprd; do
  if command -v "$decoder" >/dev/null 2>&1; then
    echo "$decoder: $(command -v "$decoder")"
  else
    echo "$decoder was not installed on PATH" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "The installed WSJT-X package does not expose all required decoders." >&2
  exit 1
fi

echo "WSJT-X command-line decoders are ready. Restart rf-mcp before use."

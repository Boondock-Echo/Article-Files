#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as your normal user; it will invoke sudo when needed." >&2
  exit 1
fi

token="${1:-}"
if [[ -z "$token" ]]; then
  token="$(openssl rand -hex 32)"
fi
if [[ "${#token}" -lt 32 || ! "$token" =~ ^[A-Za-z0-9._~-]+$ ]]; then
  echo "The token must be at least 32 characters using letters, numbers, . _ ~ or -." >&2
  exit 1
fi

temporary_file="$(mktemp)"
trap 'rm -f "$temporary_file"' EXIT
printf 'RF_MCP_API_TOKEN=%s\n' "$token" > "$temporary_file"
sudo install -o root -g root -m 0600 "$temporary_file" /etc/SDR-MCP.env
if systemctl cat SDR-MCP.service >/dev/null 2>&1; then
  sudo systemctl restart SDR-MCP.service
fi

echo "Authentication enabled. Save this token in your MCP client:"
printf '%s\n' "$token"
echo "Authorization header: Bearer $token"

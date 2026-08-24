#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_user="${SUDO_USER:-$USER}"
service_home="$(getent passwd "$service_user" | cut -d: -f6)"

if [[ -z "$service_home" || "$service_home" == "/" ]]; then
  echo "Unable to resolve a safe home directory for $service_user" >&2
  exit 1
fi
if [[ "$project_dir" != "$service_home/rf-mcp" ]]; then
  echo "Install the project at $service_home/rf-mcp before enabling Fldigi." >&2
  exit 1
fi

sudo apt update
sudo apt install -y fldigi xvfb xauth alsa-utils

echo snd-aloop | sudo tee /etc/modules-load.d/rf-mcp-snd-aloop.conf >/dev/null
sudo modprobe snd-aloop

mkdir -p "$service_home/rf-mcp-data/fldigi-config" "$service_home/rf-mcp-data/fldigi"
service_tmp="$(mktemp)"
trap 'rm -f "$service_tmp"' EXIT
sed \
  -e "s|__RF_MCP_USER__|$service_user|g" \
  -e "s|__RF_MCP_HOME__|$service_home|g" \
  "$project_dir/systemd/rf-mcp-fldigi.service" > "$service_tmp"

sudo install -m 0644 "$service_tmp" /etc/systemd/system/rf-mcp-fldigi.service
sudo systemctl daemon-reload
sudo systemctl enable --now rf-mcp-fldigi.service
sleep 3
sudo systemctl --no-pager --full status rf-mcp-fldigi.service

echo "Fldigi is running on loopback-only XML-RPC port 7362."
echo "Restart rf-mcp, then call get_fldigi_status and list_fldigi_modes."

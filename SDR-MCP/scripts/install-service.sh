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
  echo "Install the project at $service_home/rf-mcp before enabling the service." >&2
  exit 1
fi

mkdir -p \
  "$service_home/rf-mcp-data/captures" \
  "$service_home/rf-mcp-data/plots" \
  "$service_home/rf-mcp-data/results" \
  "$service_home/rf-mcp-data/audio" \
  "$service_home/rf-mcp-data/fm-surveys" \
  "$service_home/rf-mcp-data/weak-signal"
mkdir -p "$service_home/rf-mcp-data/fldigi" "$service_home/rf-mcp-data/fldigi-config"
mkdir -p "$service_home/rf-mcp-data/sstv"
mkdir -p "$service_home/rf-mcp-data/matplotlib-cache"
service_tmp="$(mktemp)"
trap 'rm -f "$service_tmp"' EXIT
sed \
  -e "s|__RF_MCP_USER__|$service_user|g" \
  -e "s|__RF_MCP_HOME__|$service_home|g" \
  "$project_dir/systemd/rf-mcp.service" > "$service_tmp"

sudo install -m 0644 "$service_tmp" /etc/systemd/system/rf-mcp.service
sudo systemctl daemon-reload
sudo systemctl enable --now rf-mcp.service
sudo systemctl --no-pager --full status rf-mcp.service

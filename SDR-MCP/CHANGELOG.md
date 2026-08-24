# Changelog

## 1.0.1 — 2026-08-13

- Fixes the dashboard content-security policy so its authenticated same-origin
  stylesheet is permitted to load.
- Adds regression checks for the stylesheet directive and high-contrast theme
  tokens, preventing a fallback to unusable browser-default rendering.

## 1.0.0 — 2026-08-13

- Declares the RF MCP API 1.0 compatibility contract and stable core tools.
- Adds production-readiness reporting and optional hardware probing.
- Stabilizes Airspy HF+ and RTL-SDR capture, receiver onboarding, durable leases,
  recovery behavior, application services, calibration, and qualification work.
- Adds continuous integration, security guidance, and contributor checks.
- Preserves v0.69 data files and SQLite catalogs without a destructive migration.

## 0.69.0

- Added persistent receiver calibration, RTL-SDR PPM correction, traceable dBm
  conversion, and receiver qualification.

## 0.68.0

- Added shared application services and authenticated dashboard assets.

## 0.67.0

- Added durable SQLite receiver leases and formal catalog migration history.

## 0.66.0

- Added guided receiver discovery and registration in the dashboard.

## 0.65.0

- Added the functional RTL-SDR backend.

## 0.64.0

- Introduced the receiver-backend interface.

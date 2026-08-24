# Security policy

## Supported release

Security fixes are provided for the current 1.x release line. Upgrade to the
latest available patch release before reporting a suspected issue.

## Deployment boundary

RF MCP is a receive-only private-LAN appliance service. Do not expose its HTTP
endpoint directly to the public Internet. Enable bearer authentication on a
shared LAN and use a VPN or TLS reverse proxy across untrusted networks.

The bearer token must not be placed in issue reports, screenshots, logs, test
fixtures, or example configuration. RF recordings and decoded content may also
contain sensitive information and should be handled according to local policy.

## Reporting

Report suspected vulnerabilities privately to the project owner or maintainer.
Include the RF MCP version, operating system, reproduction steps, and impact.
Do not include real tokens or sensitive recordings. Avoid public disclosure
until a fix or mitigation is available.

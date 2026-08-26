#!/usr/bin/env python3
"""Measure streaming headers and first data at one or more network stages."""
import argparse
import json
import platform
import time
import urllib.request


def measure(base, path, token):
    request = urllib.request.Request(base.rstrip("/") + path)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    started = time.perf_counter()
    response = urllib.request.urlopen(request, timeout=30)
    headers = time.perf_counter()
    first = response.read(1)
    data = time.perf_counter()
    response.close()
    return {"status": response.status, "content_type": response.headers.get("content-type"),
            "header_ms": round((headers-started)*1000, 2),
            "first_data_ms": round((data-started)*1000, 2), "received": bool(first)}


parser = argparse.ArgumentParser()
parser.add_argument("base_urls", nargs="+", help="direct, proxy, and remote base URLs")
parser.add_argument("--token")
parser.add_argument("--backend-model", required=True)
parser.add_argument("--sample-rate", type=int, required=True)
parser.add_argument("--proxy-topology", default="direct")
parser.add_argument("--path", default="/api/live-waterfall?center_frequency_hz=100000000&fft_size=256&span_hz=100000&display_bins=64")
args = parser.parse_args()
result = {"backend_model": args.backend_model, "sample_rate_hz": args.sample_rate,
          "host": platform.platform(), "host_load": list(__import__("os").getloadavg()),
          "proxy_topology": args.proxy_topology,
          "stages": {base: measure(base, args.path, args.token) for base in args.base_urls}}
print(json.dumps(result, indent=2))

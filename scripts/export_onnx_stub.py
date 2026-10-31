#!/usr/bin/env python
"""Deprecated stub — use scripts/export_onnx.py (real exporter)."""

from __future__ import annotations

import sys

print(
    "scripts/export_onnx_stub.py is deprecated.\n"
    "Use: py -3 scripts/export_onnx.py 4x-spanx4-ch48 -o models/4x-spanx4-ch48.onnx",
    file=sys.stderr,
)
sys.exit(2)

#!/usr/bin/env python
"""Launch the pyTGF web interface.

Usage:
    python server.py                  # auto-open browser at http://127.0.0.1:5000
    python server.py --port 8080      # custom port
    python server.py --no-browser     # headless mode
"""
import os, sys

# Ensure pyTGF package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyTGF.web.server import main
sys.exit(main())

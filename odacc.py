#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from odacc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

"""
Generates one identifier per run, shared by Results.py, Plots.py, and
Model_Resolution.py, so that each run's outputs land in their own
Results/<RUN_ID>/ subfolder instead of overwriting the previous run.

Usage: python MicroGrids.py [optional_tag]
  python MicroGrids.py                 -> Results/20260723_143200/...
  python MicroGrids.py 9sc_MILP_final  -> Results/20260723_143200_9sc_MILP_final/...
"""
import re
import sys
from datetime import datetime

_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_tag = sys.argv[1] if len(sys.argv) > 1 else None

if _tag:
    _tag = re.sub(r'[^A-Za-z0-9_-]+', '_', _tag)
    RUN_ID = f"{_timestamp}_{_tag}"
else:
    RUN_ID = _timestamp

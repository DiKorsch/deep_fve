#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import sys
import unittest
import torch as th

th.autograd.set_detect_anomaly(True)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from tests import *  # noqa: E402 E403

logging.disable(logging.CRITICAL)
unittest.main()

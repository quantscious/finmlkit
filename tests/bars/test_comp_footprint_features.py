import pytest
import numpy as np
import pandas as pd
import os
from numba import njit
from numpy.typing import NDArray
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
from finmlkit.bar.base import comp_footprint_features



"""
Regime-Switching Vector Autoregression (RS-VAR) for Multivariate Market Dynamics (Step 4c).
Models joint cross-asset regime transitions across Nifty returns, realised volatility, and market breadth.
Computes regime-conditional innovation covariances and impulse responses.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseRegimeModel, REGIME_NAMES
from .ms_var import MarkovSwitchingVAR

logger = logging.getLogger(__name__)

__all__ = ["MarkovSwitchingVAR"]

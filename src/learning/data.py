"""Thin data-loading helpers on top of statarb.data.api.DataAPI.

Every exercise in this package should get its input series through here rather
than importing DataAPI directly, so the warehouse-access pattern stays in one
place.
"""

from __future__ import annotations

import numpy as np
from statarb.data.api import DataAPI


def load_close_series(ticker: str, adjusted: str = "raw") -> np.ndarray:
    """Chronological 1-D array of daily close prices for one ticker."""
    api = DataAPI()
    sid = api.sid_for_ticker(ticker)
    df = (
        api.get_prices([sid], adjusted=adjusted, columns=["sid", "d", "px_close"])
        .sort("d")
    )
    return df["px_close"].to_numpy()

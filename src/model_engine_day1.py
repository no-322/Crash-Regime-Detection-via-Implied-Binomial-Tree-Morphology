from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Person 2 – Day 1 modeling engine

This module connects:

- iv_fitter.build_dummy_smile        → base cubic-spline IV smile
- implied_tree.build_tree_from_smile → local-vol binomial tree
- summary_stats.compute_summary_stats → tail prob, skew, kurtosis, etc.

and is designed to be compatible with Person 1's output file
`data/crash_vol_inputs.csv` which should contain columns:

    crash_id, S0, T_years, sigma_atm_from_VIX, sigma_realized_pre, VIX_crash
"""


from typing import Tuple, Dict
from pathlib import Path

import os
import numpy as np
import pandas as pd

from iv_fitter import build_dummy_smile
from implied_tree import build_tree_from_smile
from summary_stats import compute_summary_stats


# ---------------- USER PARAMETERS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CRASH_VOL_FILE = DATA_DIR / "crash_vol_inputs.csv"

# Non-recombining tree: N must be quite small (e.g. 8–10), since complexity is O(2^N)
DEFAULT_N_STEPS = 8
DEFAULT_R = 0.0      # short rate; for 30-day horizons you can set this ~0 for now
TAIL_M = 0.8         # tail threshold as multiple of S0
SIGMA_FLOOR = 0.01   # lower bound for local vol
SIGMA_CAP = 2.0      # upper bound for local vol
M_MIN, M_MAX = 0.8, 1.2  # domain where the base dummy smile is reliable
# -------------------------------------------------


def make_local_vol_from_vix(
    sigma_atm: float,
    base_smile=None,
    m_min: float = M_MIN,
    m_max: float = M_MAX,
    sigma_floor: float = SIGMA_FLOOR,
    sigma_cap: float = SIGMA_CAP,
):
    """
    Construct a local volatility function sigma(m) rescaled to match
    a target ATM volatility sigma_atm from VIX.

    We start from a "shape" smile sigma_base(m) from iv_fitter.build_dummy_smile,
    then scale it so that sigma(1.0) = sigma_atm, and finally clamp to
    [sigma_floor, sigma_cap] to avoid absurd vols in the tree.
    """
    if base_smile is None:
        base_smile = build_dummy_smile()

    base_atm = float(base_smile(1.0))
    if not np.isfinite(base_atm) or base_atm <= 0:
        raise ValueError(f"Base smile ATM vol is invalid: {base_atm}")

    scale = float(sigma_atm) / base_atm

    def sigma_of_m(m: float) -> float:
        m = float(m)
        # Clip moneyness to smile domain so we don't extrapolate crazily
        m_clipped = min(max(m, m_min), m_max)
        sigma = scale * float(base_smile(m_clipped))

        # Fallback if spline returns nonsense
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(sigma_atm)

        # Clamp for numerical stability
        sigma = max(sigma_floor, min(sigma_cap, sigma))
        return float(sigma)

    return sigma_of_m


def run_tree_for_crash_row(
    row: pd.Series,
    r: float = DEFAULT_R,
    N: int = DEFAULT_N_STEPS,
    tail_m: float = TAIL_M,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    For a single crash (one row of crash_vol_inputs.csv), build the local-vol
    tree, get the terminal distribution, and compute summary stats.

    Parameters
    ----------
    row : pd.Series
        One row from crash_vol_inputs.csv. Needs fields:
            - 'S0'
            - 'T_years'
            - 'sigma_atm_from_VIX'
    r : float
        Risk-free rate.
    N : int
        Number of time steps in the non-recombining tree (keep small).
    tail_m : float
        Tail threshold multiple for the summary stats.

    Returns
    -------
    S_T : np.ndarray
        Terminal prices.
    probs : np.ndarray
        Probabilities.
    stats : dict
        Summary statistics for this crash.
    """
    S0 = float(row["S0"])
    T_years = float(row["T_years"])
    sigma_atm = float(row["sigma_atm_from_VIX"])

    # Build local-vol function sigma(m) anchored to this crash's VIX ATM level
    sigma_of_m = make_local_vol_from_vix(sigma_atm)

    # Build binomial terminal distribution
    S_T, probs = build_tree_from_smile(S0, r, T_years, N, sigma_of_m)

    # Summary stats
    stats = compute_summary_stats(S_T, probs, S0=S0, tail_m=tail_m)
    return S_T, probs, stats


def run_all_crashes(
    crash_vol_path: str = CRASH_VOL_FILE,
    r: float = DEFAULT_R,
    N: int = DEFAULT_N_STEPS,
    tail_m: float = TAIL_M,
) -> pd.DataFrame:
    """
    Convenience helper: loop over all crashes in crash_vol_inputs.csv and
    return a DataFrame with summary stats for each crash_id.
    """
    df = pd.read_csv(crash_vol_path)
    rows = []

    for _, row in df.iterrows():
        S_T, probs, stats = run_tree_for_crash_row(row, r=r, N=N, tail_m=tail_m)
        stats_row = {"crash_id": int(row["crash_id"])}
        stats_row.update(stats)
        rows.append(stats_row)

    stats_df = pd.DataFrame(rows)
    return stats_df


def main():
    """
    Simple CLI entry point:
    - Load crash_vol_inputs.csv
    - Run tree + stats for each crash
    - Print the resulting stats table
    """
    if not os.path.exists(CRASH_VOL_FILE):
        raise FileNotFoundError(
            f"{CRASH_VOL_FILE} not found. Make sure Person 1 ran crash_vix_pipeline.py "
            f"and produced crash_vol_inputs.csv in the 'data/' folder."
        )

    stats_df = run_all_crashes()
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()

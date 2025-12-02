#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 01:27:53 2025

@author: atharva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Day 3 plotting script for Person 2.

For each crash in data/crash_vol_inputs.csv (Person 1 output), this script:

1. Builds the VIX-anchored local-vol function sigma(m).
2. Builds the full non-recombining binomial lattice and saves a tree plot.
3. Uses implied_tree.build_tree_from_smile to get the terminal distribution
   and saves a bar plot of S_T vs probability.
"""

import os
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iv_fitter import build_dummy_smile
from implied_tree import build_tree_from_smile
from tree_plots import build_local_vol_lattice, plot_binomial_tree_lattice

# -------- paths & constants --------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" 
CRASH_VOL_FILE = os.path.join(DATA_DIR, "crash_vol_inputs.csv")
PLOT_DIR = "plots_binomial"
os.makedirs(PLOT_DIR, exist_ok=True)

DEFAULT_N_STEPS = 8
DEFAULT_R = 0.0
SIGMA_FLOOR = 0.01
SIGMA_CAP = 2.0
M_MIN, M_MAX = 0.8, 1.2
# -----------------------------------


def make_local_vol_from_vix(
    sigma_atm: float,
    m_min: float = M_MIN,
    m_max: float = M_MAX,
    sigma_floor: float = SIGMA_FLOOR,
    sigma_cap: float = SIGMA_CAP,
) -> Callable[[float], float]:
    """Same VIX-anchored smile as used in the JSON script."""
    base_smile = build_dummy_smile()
    base_atm = float(base_smile(1.0))
    if base_atm <= 0 or not np.isfinite(base_atm):
        raise ValueError(f"Base smile ATM vol invalid: {base_atm}")

    scale = float(sigma_atm) / base_atm

    def sigma_of_m(m: float) -> float:
        m = float(m)
        m_clipped = min(max(m, m_min), m_max)
        sigma = scale * float(base_smile(m_clipped))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(sigma_atm)
        sigma = max(sigma_floor, min(sigma_cap, sigma))
        return float(sigma)

    return sigma_of_m


def plot_terminal_distribution(
    S_T: np.ndarray,
    probs: np.ndarray,
    S0: float,
    crash_id: int,
    out_dir: str = PLOT_DIR,
) -> str:
    """Simple bar plot of terminal distribution for a given crash."""
    S_T = np.asarray(S_T, float).ravel()
    probs = np.asarray(probs, float).ravel()
    probs = probs / probs.sum()

    width = (S_T.max() - S_T.min()) / max(len(S_T), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(S_T, probs, width=width, alpha=0.7)
    ax.axvline(S0, color="black", linestyle="--", label="S0")
    ax.set_xlabel("S_T")
    ax.set_ylabel("Probability")
    ax.set_title(f"Terminal distribution – Crash {crash_id}")
    ax.legend()

    fname = os.path.join(out_dir, f"crash{crash_id}_terminal_dist.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname


def main():
    if not os.path.exists(CRASH_VOL_FILE):
        raise FileNotFoundError(
            f"{CRASH_VOL_FILE} not found. Run crash_vix_pipeline.py first."
        )

    crash_df = pd.read_csv(CRASH_VOL_FILE)

    for _, row in crash_df.iterrows():
        crash_id = int(row["crash_id"])
        S0 = float(row["S0"])
        T_years = float(row["T_years"])
        sigma_atm = float(row["sigma_atm_from_VIX"])

        sigma_of_m = make_local_vol_from_vix(sigma_atm)

        # 1) Tree lattice plot
        lattice = build_local_vol_lattice(
            S0, DEFAULT_R, T_years, DEFAULT_N_STEPS, sigma_of_m
        )
        tree_path = os.path.join(PLOT_DIR, f"crash{crash_id}_tree.png")
        plot_binomial_tree_lattice(lattice, S0, tree_path)

        # 2) Terminal distribution plot
        S_T, probs = build_tree_from_smile(
            S0, DEFAULT_R, T_years, DEFAULT_N_STEPS, sigma_of_m
        )
        term_path = plot_terminal_distribution(S_T, probs, S0, crash_id)

        print(
            f"Crash {crash_id}: saved tree plot to {tree_path} and "
            f"terminal distribution to {term_path}"
        )


if __name__ == "__main__":
    main()

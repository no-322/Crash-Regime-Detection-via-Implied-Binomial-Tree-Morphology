#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 01:07:57 2025

@author: atharva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model engine Day 1 with JSON output.

Reads crash_vol_inputs.csv (from Person 1),
builds the VIX-anchored local-vol tree for each crash,
computes summary stats, and writes a JSON file with entries like:

{
  "crash_id": "Crash1",
  "S0": 450.23,
  "T_years": 0.119048,
  "mean": 430.5,
  "std": 52.1,
  "skew": -0.85,
  "kurtosis": 3.7,
  "tail_prob_<0.8S0": 0.32
}
"""

import json
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

from iv_fitter import build_dummy_smile
from implied_tree import build_tree_from_smile
from summary_stats import compute_summary_stats


# ---------- paths & constants ----------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
JSON_DIR = BASE_DIR / "results"
CRASH_VOL_FILE = DATA_DIR / "crash_vol_inputs.csv"
JSON_OUT_FILE = JSON_DIR / "crash_stats_day1.json"

DEFAULT_N_STEPS = 10     # non-recombining ⇒ keep small
DEFAULT_R = 0.0         # short rate for ~30 days
TAIL_M = 0.8            # tail threshold multiple
SIGMA_FLOOR = 0.01
SIGMA_CAP = 2.0
M_MIN, M_MAX = 0.8, 1.2
# --------------------------------------


def make_local_vol_from_vix(
    sigma_atm: float,
    m_min: float = M_MIN,
    m_max: float = M_MAX,
    sigma_floor: float = SIGMA_FLOOR,
    sigma_cap: float = SIGMA_CAP,
):
    """
    Dummy cubic-spline smile rescaled so that sigma(1.0) = sigma_atm from VIX.
    """
    base_smile = build_dummy_smile()
    base_atm = float(base_smile(1.0))
    if not np.isfinite(base_atm) or base_atm <= 0:
        raise ValueError(f"Base smile ATM vol invalid: {base_atm}")

    scale = float(sigma_atm) / base_atm

    def sigma_of_m(m: float) -> float:
        m = float(m)
        # clamp moneyness to smile domain
        m_clipped = min(max(m, m_min), m_max)
        sigma = scale * float(base_smile(m_clipped))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(sigma_atm)
        # numeric safety
        sigma = max(sigma_floor, min(sigma_cap, sigma))
        return float(sigma)

    return sigma_of_m


def run_tree_for_crash_row(
    row: pd.Series,
    return_distribution: bool = False,
) -> Dict[str, Any] | tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    One crash row → terminal distribution → JSON-friendly dict.
    """
    S0 = float(row["S0"])
    T_years = float(row["T_years"])
    sigma_atm = float(row["sigma_atm_from_VIX"])

    sigma_of_m = make_local_vol_from_vix(sigma_atm)
    S_T, probs = build_tree_from_smile(
        S0, DEFAULT_R, T_years, DEFAULT_N_STEPS, sigma_of_m
    )

    stats = compute_summary_stats(S_T, probs, S0=S0, tail_m=TAIL_M)

    tail_key = f"tail_prob_<{stats['tail_return_threshold']:.3f}"
    out: Dict[str, Any] = {
        # string like "Crash1" to match your example
        "crash_id": f"Crash{int(row['crash_id'])}",
        "S0": S0,
        "T_years": T_years,
        "mean": float(stats["mean_ret"]),
        "std": float(stats["std_ret"]),
        "skew": float(stats["skew_ret"]),
        "kurtosis": float(stats.get("kurt_ret", np.nan)),
        tail_key: float(stats["tail_prob"]),
        f"VaR_{int(stats['var_alpha']*100)}pct": float(stats["var_ret"]),
        f"CVaR_{int(stats['var_alpha']*100)}pct": float(stats["cvar_ret"]),
    }
    if return_distribution:
        return S_T, probs, out
    return out


def generate_crash_summaries(crash_vol_file: Path = CRASH_VOL_FILE) -> list[Dict[str, Any]]:
    """
    Read crash_vol_inputs.csv and compute summary stats for every crash.
    """
    crash_vol_file = Path(crash_vol_file)
    if not crash_vol_file.exists():
        raise FileNotFoundError(
            f"{crash_vol_file} not found. Run crash_vix_pipeline.py first."
        )

    df = pd.read_csv(crash_vol_file)
    results: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        results.append(run_tree_for_crash_row(row))
    return results


def write_summary_jsons(
    summaries: list[Dict[str, Any]],
    json_dir: Path = JSON_DIR,
    aggregate_path: Path = JSON_OUT_FILE,
) -> None:
    """
    Persist both the aggregate list and per-crash JSON files.
    """
    json_dir.mkdir(parents=True, exist_ok=True)

    if aggregate_path is not None:
        with Path(aggregate_path).open("w") as f:
            json.dump(summaries, f, indent=2)

    for summary in summaries:
        crash_id = summary["crash_id"]
        per_file = json_dir / f"{crash_id}_summary.json"
        with per_file.open("w") as f:
            json.dump(summary, f, indent=2)


def main():
    summaries = generate_crash_summaries()
    write_summary_jsons(summaries)

    print(f"Wrote {len(summaries)} crash records to {JSON_DIR}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One-click orchestrator for the crash project.

Order of operations:
1) Person A data pipeline (crash_options.main) → data/*.csv
2) Person B tree engine (model_engine_day1_json.generate_crash_summaries) → results/*.json
3) Person C crash signature (crash_signature.build_crash_signature/classify_crash) → results/crash3_classification.json

This keeps the individual modules unchanged but provides a single entry point
that stitches everything together in the intended sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crash_options import main as run_person_a_pipeline
from model_engine_day1_json import (
    CRASH_VOL_FILE,
    JSON_DIR,
    M_MAX,
    M_MIN,
    generate_crash_summaries,
    make_local_vol_from_vix,
    run_tree_for_crash_row,
    write_summary_jsons,
)
from crash_signature import build_crash_signature, classify_crash, plot_crash_metrics

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
P2_PLOTS = PLOTS_DIR / "person2"
P3_PLOTS = PLOTS_DIR / "person3"
CLASSIFICATION_FILE = RESULTS_DIR / "crash3_classification.json"


def run_person_a() -> None:
    """Runs the data/VIX pipeline (downloads SPY + VIX, builds CSV files)."""
    run_person_a_pipeline()


def run_person_b(crash_vol_path: Path = CRASH_VOL_FILE) -> List[Dict[str, Any]]:
    """
    Runs the tree engine for every crash and writes JSON summaries.
    """
    summaries = generate_crash_summaries(crash_vol_path)
    write_summary_jsons(summaries)
    return summaries


def plot_person2_outputs(crash_vol_path: Path = CRASH_VOL_FILE, out_dir: Path = P2_PLOTS) -> None:
    """
    Generate Person 2 plots: local-vol smile and terminal distribution per crash.
    """
    crash_vol_path = Path(crash_vol_path)
    df = pd.read_csv(crash_vol_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        crash_id = int(row["crash_id"])
        S0 = float(row["S0"])
        T_years = float(row["T_years"])
        sigma_atm = float(row["sigma_atm_from_VIX"])

        sigma_of_m = make_local_vol_from_vix(sigma_atm)

        # Smile plot
        m_grid = np.linspace(M_MIN, M_MAX, 200)
        sigmas = [sigma_of_m(m) for m in m_grid]
        fig, ax = plt.subplots()
        ax.plot(m_grid, sigmas, label="Local vol (scaled dummy smile)")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Moneyness m = S/S0")
        ax.set_ylabel("sigma(m)")
        ax.set_title(f"Crash {crash_id} local-vol smile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"crash{crash_id}_smile.png", dpi=150)
        plt.close(fig)

        # Terminal distribution plot (reuse tree computation)
        S_T, probs, _summary = run_tree_for_crash_row(row, return_distribution=True)
        probs = np.asarray(probs, float) / np.sum(probs)
        width = (np.max(S_T) - np.min(S_T)) / max(len(S_T), 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(S_T, probs, width=width, alpha=0.7)
        ax.axvline(S0, color="black", linestyle="--", label="S0")
        ax.set_xlabel("S_T")
        ax.set_ylabel("Probability")
        ax.set_title(f"Terminal distribution – Crash {crash_id}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"crash{crash_id}_terminal.png", dpi=150)
        plt.close(fig)


def run_person_c(
    summaries: List[Dict[str, Any]],
    results_dir: Path = RESULTS_DIR,
) -> Optional[Dict[str, Any]]:
    """
    Build a crash signature from Crash1 & Crash2, then classify Crash3.
    """
    if len(summaries) < 3:
        print("Need at least three crashes to build and test a signature.")
        return None

    signature = build_crash_signature(summaries[0], summaries[1])
    classification = classify_crash(summaries[2], signature)
    plot_person3_metrics(summaries)

    payload = {
        "signature": signature,
        "tested_crash_id": summaries[2]["crash_id"],
        "classification": classification,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with CLASSIFICATION_FILE.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Crash {summaries[2]['crash_id']} classification saved to {CLASSIFICATION_FILE}")
    return payload


def plot_person3_metrics(summaries: List[Dict[str, Any]], out_dir: Path = P3_PLOTS) -> Optional[Path]:
    """
    Generate the crash metrics comparison plot used by Person 3.
    """
    if not summaries:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [s["crash_id"] for s in summaries]
    save_path = out_dir / "crash_signature_metrics.png"
    plot_crash_metrics(summaries, names, save_path=save_path)
    return save_path


def run_full_pipeline(include_person_a: bool = False) -> Dict[str, Any]:
    """
    Run the full chain. By default Person A is skipped because it downloads
    data; set include_person_a=True if you want to refresh the CSVs.
    """
    if include_person_a:
        print("Running Person A pipeline (data download + crash selection)...")
        run_person_a()

    print("Running Person B pipeline (tree summaries + plots)...")
    summaries = run_person_b()
    plot_person2_outputs()

    print("Running Person C pipeline (signature + classification)...")
    classification = run_person_c(summaries)

    return {"summaries": summaries, "classification": classification}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full crash pipeline.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Run Person A data download first (requires network access).",
    )
    args = parser.parse_args()

    run_full_pipeline(include_person_a=args.refresh_data)

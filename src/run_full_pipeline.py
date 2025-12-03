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
    DEFAULT_R,
    DEFAULT_N_STEPS,
    generate_crash_summaries,
    make_local_vol_from_vix,
    run_tree_for_crash_row,
    write_summary_jsons,
)
from crash_signature import build_crash_signature, classify_crash, plot_crash_metrics
from tree_plots import build_local_vol_lattice, plot_binomial_tree_lattice
from summary_stats import compute_summary_stats
import datetime as dt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
P2_PLOTS = PLOTS_DIR / "person2"
P3_PLOTS = PLOTS_DIR / "person3"
VERIFICATION_FILE = RESULTS_DIR / "model_vs_realized.json"
CLASSIFICATION_FILE = RESULTS_DIR / "crash3_classification.json"
PRECRASH_FILE = RESULTS_DIR / "crash3_precrash_predictions.json"
VERIFICATION_PLOT = PLOTS_DIR / "model_vs_realized.png"


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

        # Lattice plot
        lattice = build_local_vol_lattice(
            S0, DEFAULT_R, T_years, DEFAULT_N_STEPS, sigma_of_m
        )
        plot_binomial_tree_lattice(
            lattice,
            S0,
            out_path=str(out_dir / f"crash{crash_id}_lattice.png"),
            title=f"Crash {crash_id} lattice",
        )


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


def compute_realized_window_stats(
    horizon_days: int = 30,
    var_alpha: float = 0.05,
) -> Dict[int, Dict[str, float]]:
    """
    For each crash in crash_meta.csv, compute realized returns over a window
    after the crash date and report tail prob, mean/std/skew/kurt, VaR/CVaR.
    """
    spy_file = BASE_DIR / "data" / "spy_vix_full.csv"
    meta_file = BASE_DIR / "data" / "crash_meta.csv"
    if not spy_file.exists() or not meta_file.exists():
        return {}

    spy = pd.read_csv(spy_file, parse_dates=["Date"])
    meta = pd.read_csv(meta_file, parse_dates=["crash_date"])
    spy = spy.sort_values("Date").set_index("Date")

    realized: Dict[int, Dict[str, float]] = {}
    for _, row in meta.iterrows():
        cid = int(row["crash_id"])
        crash_date = row["crash_date"]
        S0 = float(row["S0"])

        window = spy.loc[crash_date:].iloc[1 : horizon_days + 1].copy()
        if window.empty:
            continue

        S_T = window["SPY_close"].to_numpy(dtype=float)
        probs = np.ones_like(S_T, dtype=float) / len(S_T)
        stats = compute_summary_stats(S_T, probs, S0=S0, tail_m=0.8, var_alpha=var_alpha)
        realized[cid] = {
            "tail_prob": stats["tail_prob"],
            "mean_ret": stats["mean_ret"],
            "std_ret": stats["std_ret"],
            "skew_ret": stats["skew_ret"],
            "kurt_ret": stats["kurt_ret"],
            "var_ret": stats["var_ret"],
            "cvar_ret": stats["cvar_ret"],
            "var_alpha": stats["var_alpha"],
            "window_days": horizon_days,
        }
    return realized


def compare_model_realized(
    summaries: List[Dict[str, Any]],
    realized: Dict[int, Dict[str, float]],
    out_path: Path = VERIFICATION_FILE,
    plot_path: Path = VERIFICATION_PLOT,
) -> Optional[Path]:
    """
    Compare model vs realized metrics for each crash and write JSON.
    """
    if not summaries or not realized:
        return None

    records = []
    plot_rows = []
    for s in summaries:
        cid = int(str(s["crash_id"]).replace("Crash", ""))
        real = realized.get(cid)
        if not real:
            continue
        record = {
            "crash_id": s["crash_id"],
            "model": {
                "tail_prob": s[[k for k in s.keys() if k.startswith("tail_prob_")][0]],
                "mean_ret": s["mean"],
                "std_ret": s["std"],
                "skew_ret": s["skew"],
                "kurt_ret": s.get("kurtosis"),
            },
            "realized": real,
        }
        records.append(record)
        plot_rows.append(
            {
                "crash": s["crash_id"],
                "model_tail": record["model"]["tail_prob"],
                "real_tail": real["tail_prob"],
                "model_skew": record["model"]["skew_ret"],
                "real_skew": real["skew_ret"],
                "model_vol": record["model"]["std_ret"],
                "real_vol": real["std_ret"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote model vs realized verification to {out_path}")

    # Plot comparison bars if we have data
    if plot_rows:
        df_plot = pd.DataFrame(plot_rows)
        x = np.arange(len(df_plot))
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Tail prob
        axes[0].bar(x - width / 2, df_plot["model_tail"], width, label="Model")
        axes[0].bar(x + width / 2, df_plot["real_tail"], width, label="Realized")
        axes[0].set_title("Tail prob (<0.8 S0)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_plot["crash"])
        axes[0].legend()

        # Skew
        axes[1].bar(x - width / 2, df_plot["model_skew"], width, label="Model")
        axes[1].bar(x + width / 2, df_plot["real_skew"], width, label="Realized")
        axes[1].set_title("Skew (returns)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_plot["crash"])

        # Volatility (std)
        axes[2].bar(x - width / 2, df_plot["model_vol"], width, label="Model")
        axes[2].bar(x + width / 2, df_plot["real_vol"], width, label="Realized")
        axes[2].set_title("Volatility (std of returns)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(df_plot["crash"])

        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote model vs realized plot to {plot_path}")

    return out_path


def pre_crash_prediction_metrics(
    signature: Dict[str, Any],
    lookback_days: int = 7,
    horizon_days: int = 30,
    out_path: Path = PRECRASH_FILE,
) -> Optional[Path]:
    """
    For the latest crash (Crash3), compute model metrics for each trading day
    in the week leading up to the crash date and classify them with the
    existing signature. Intended to see if the signature would have flagged
    the crash ahead of time.
    """
    spy_file = BASE_DIR / "data" / "spy_vix_full.csv"
    meta_file = BASE_DIR / "data" / "crash_meta.csv"
    if not spy_file.exists() or not meta_file.exists():
        return None

    spy = pd.read_csv(spy_file, parse_dates=["Date"])
    meta = pd.read_csv(meta_file, parse_dates=["crash_date"])
    spy = spy.sort_values("Date").set_index("Date")
    meta = meta.sort_values("crash_date")

    # Take the latest crash (chronological ordering assumed)
    crash3_row = meta.iloc[-1]
    crash_date = crash3_row["crash_date"]

    window_mask = (spy.index >= crash_date - pd.Timedelta(days=lookback_days)) & (
        spy.index < crash_date
    )
    window_df = spy.loc[window_mask].copy()
    if window_df.empty:
        return None

    records = []
    for i, (date, row) in enumerate(window_df.iterrows(), start=1):
        S0 = float(row["SPY_close"])
        vix = float(row["VIX_close"])
        # Create a minimal Series for run_tree_for_crash_row
        pseudo_row = pd.Series(
            {
                "crash_id": 300 + i,  # dummy id for labeling
                "S0": S0,
                "T_years": horizon_days / 252,
                "sigma_atm_from_VIX": vix / 100.0,
            }
        )
        metrics = run_tree_for_crash_row(pseudo_row)
        classification = classify_crash(metrics, signature)
        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "spy_close": S0,
                "vix_close": vix,
                "model_metrics": metrics,
                "classification": classification,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote pre-crash prediction metrics to {out_path}")
    return out_path


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

    print("Running model vs realized verification (VaR/CVaR, tail)...")
    realized = compute_realized_window_stats()
    compare_model_realized(summaries, realized)

    if classification and "signature" in classification:
        print("Running pre-crash prediction metrics for Crash3 lookback window...")
        precrash = pre_crash_prediction_metrics(classification["signature"])
    else:
        precrash = None

    return {
        "summaries": summaries,
        "classification": classification,
        "realized": realized,
        "precrash": precrash,
    }


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

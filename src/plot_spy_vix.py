#!/usr/bin/env python3
"""
Quick VIX vs SPY plot using existing cached data (no download).

Reads data/spy_vix_full.csv (produced by crash_options.py) and saves
an overlay chart to plots/spy_vix_overlay.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "spy_vix_full.csv"
OUT_FILE = BASE_DIR / "plots" / "spy_vix_overlay.png"


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run crash_options.py first.")

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values("Date")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df["Date"], df["SPY_close"], color="tab:blue", label="SPY close")
    ax2.plot(df["Date"], df["VIX_close"], color="tab:red", label="VIX close", alpha=0.7)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("SPY close", color="tab:blue")
    ax2.set_ylabel("VIX close", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUT_FILE}")


if __name__ == "__main__":
    main()

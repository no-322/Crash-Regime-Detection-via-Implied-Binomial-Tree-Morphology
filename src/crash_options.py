# crash_vix_pipeline.py
"""
Day 1–3: SPY crashes, VIX alignment, vol inputs, and sanity plots

- Day 1: Identify crashes & align VIX (using pre-downloaded CSV)
- Day 2: Compute ATM vol from VIX + realized vol pre-crash
- Day 3: Plot SPY & VIX around crashes for sanity
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import os
import json

from pathlib import Path

# ---------------- USER PARAMETERS ----------------
SPY_TICKER = "SPY"

# base dir = project root (one level above src/)
BASE_DIR = Path(__file__).resolve().parents[1]

VIX_FILE   = BASE_DIR / "data" / "VIX_History.csv"
OUTPUT_DIR = BASE_DIR / "data"
PLOT_DIR   = BASE_DIR / "plots"
TOP_N_CRASHES = 3
PRE_CRASH_WINDOW = 20 # days for realized vol
POST_PREP_WINDOW = 10 # optional, for context
# -------------------------------------------------


# ---------------- HELPERS ----------------
def load_vix(vix_file):
    vix = pd.read_csv(vix_file)
    vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')
    vix.set_index('DATE', inplace=True)
    vix.index = vix.index.tz_localize(None)  # tz-naive
    vix = vix.rename(columns={'CLOSE':'VIX_close'})
    return vix

def compute_log_returns(series):
    return np.log(series/series.shift(1))

def select_diverse_crashes(df, n=3):
    """Select top n crash days, ensuring one per year"""
    df = df.copy()
    df['year'] = df.index.year
    crashes = []
    used_years = set()
    for idx, row in df.iterrows():
        if row['SPY_ret'] >= 0:  # only negative returns
            continue
        if row['year'] in used_years:
            continue
        crashes.append((idx, row))
        used_years.add(row['year'])
        if len(crashes) >= n:
            break
    return crashes

def realized_volatility(prices, window):
    """Compute annualized realized volatility (daily log returns)"""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv

# ---------------- MAIN ----------------
def main():
    # 1️⃣ Day 1 — Download SPY, load VIX
    print("=== Day 1: Download SPY & align VIX ===")
    end = dt.datetime.now()
    start = dt.datetime(2000, 1, 1)
    spy = yf.Ticker(SPY_TICKER)
    hist = spy.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", actions=False)
    hist = hist[~hist.index.duplicated(keep='first')]
    hist.index = hist.index.tz_localize(None)  # tz-naive

    hist['SPY_close'] = hist['Close']
    hist['SPY_ret'] = hist['Close'].pct_change()
    hist = hist[['SPY_close', 'SPY_ret']]

    vix = load_vix(VIX_FILE)

    full_df = hist.merge(vix[['VIX_close']], left_index=True, right_index=True, how='left')
    full_df.to_csv(os.path.join(OUTPUT_DIR, 'spy_vix_full.csv'))

    # Select top negative returns (crashes)
    crashes_df = full_df.dropna(subset=['SPY_ret']).sort_values('SPY_ret')
    crashes = select_diverse_crashes(crashes_df, TOP_N_CRASHES)

    # Build crash_meta.csv
    meta_rows = []
    for i, (crash_date, row) in enumerate(crashes, start=1):
        pre_vix = full_df['VIX_close'].loc[:crash_date].tail(PRE_CRASH_WINDOW).mean()
        post_vix = full_df['VIX_close'].loc[crash_date:].head(POST_PREP_WINDOW).mean()
        meta_rows.append({
            "crash_id": i,
            "crash_date": crash_date.strftime("%Y-%m-%d"),
            "S0": row['SPY_close'],
            "VIX_crash": row['VIX_close'],
            "VIX_minus10": pre_vix,
            "VIX_plus10": post_vix
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'crash_meta.csv'), index=False)
    print(f"Saved crash metadata for {TOP_N_CRASHES} crashes.")

    # 2️⃣ Day 2 — Compute vol inputs per crash
    print("=== Day 2: Compute vol inputs per crash ===")
    vol_inputs = []
    for _, row in meta_df.iterrows():
        crash_date = pd.to_datetime(row['crash_date'])
        pre_prices = full_df.loc[:crash_date].tail(PRE_CRASH_WINDOW+1)['SPY_close']
        sigma_realized_pre = realized_volatility(pre_prices, PRE_CRASH_WINDOW).iloc[-1]
        sigma_atm = row['VIX_crash'] / 100  # annualized approx
        T_years = 30/252  # 30 trading days horizon
        vol_inputs.append({
            "crash_id": row['crash_id'],
            "S0": row['S0'],
            "T_years": T_years,
            "sigma_atm_from_VIX": sigma_atm,
            "sigma_realized_pre": sigma_realized_pre,
            "VIX_crash": row['VIX_crash']
        })
    vol_df = pd.DataFrame(vol_inputs)
    vol_df.to_csv(os.path.join(OUTPUT_DIR, 'crash_vol_inputs.csv'), index=False)
    print("Saved crash vol inputs.")

    # 3️⃣ Day 3 — Sanity checks & plots
    print("=== Day 3: Plot SPY & VIX around crashes ===")
    notes = []
    for _, row in meta_df.iterrows():
        crash_date = pd.to_datetime(row['crash_date'])
        window = 20
        sub_df = full_df.loc[crash_date - pd.Timedelta(days=window): crash_date + pd.Timedelta(days=window)]
        plt.figure(figsize=(10,5))
        plt.plot(sub_df.index, sub_df['SPY_close'], label='SPY Close')
        plt.plot(sub_df.index, sub_df['VIX_close'], label='VIX', color='red')
        plt.axvline(crash_date, color='black', linestyle='--', label='Crash Date')
        plt.title(f"Crash {row['crash_id']} around {row['crash_date']}")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price / VIX")
        plot_file = os.path.join(PLOT_DIR, f"spy_vix_crash{row['crash_id']}.png")
        plt.savefig(plot_file)
        plt.close()
        notes.append(f"Crash {row['crash_id']} on {row['crash_date']}: SPY drop={row['S0']:.2f}, VIX spike={row['VIX_crash']:.2f}")

    # Save notes
    with open(os.path.join(PLOT_DIR, 'crash_notes.txt'), 'w') as f:
        for line in notes:
            f.write(line + "\n")
    print("Saved plots and notes.")

if __name__ == "__main__":
    main()
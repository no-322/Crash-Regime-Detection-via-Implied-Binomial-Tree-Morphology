# src/crash_signature.py
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import json
from pathlib import Path

def summarize_distribution(S, p, S0, tail_cut=0.8):
    """
    S: array of terminal prices
    p: array of probabilities (sum to 1)
    S0: spot at start
    """
    S = np.array(S, dtype=float)
    p = np.array(p, dtype=float)
    p = p / p.sum()

    mean = np.sum(S * p)
    var = np.sum((S - mean)**2 * p)
    std = np.sqrt(var)

    # For skew & kurt, expand discrete dist to sample representation
    # (approximation, good enough for this project)
    scaled_counts = np.maximum((p * 10000).astype(int), 1)
    sample = np.repeat(S, scaled_counts)

    sk = skew(sample)
    kt = kurtosis(sample)

    tail_prob = np.sum(p[S < tail_cut * S0])

    return {
        "mean": float(mean),
        "std": float(std),
        "skew": float(sk),
        "kurtosis": float(kt),
        "tail_prob_<{}S0".format(tail_cut): float(tail_prob),
    }


def build_crash_signature(summary1, summary2, tail_key=None):
    """
    Build a crash signature from two crash summaries.

    By default, uses tail_prob_<0.8S0 in the dict (infer tail_key).
    We set thresholds conservatively:
      - tail_thresh = min(tail1, tail2) (require at least the smaller tail mass)
      - skew_thresh = min(skew1, skew2) (require skew no higher than the more negative)
    """
    if tail_key is None:
        # auto-detect key like 'tail_prob_<0.8S0'
        tail_keys = [k for k in summary1.keys() if k.startswith("tail_prob_<")]
        if len(tail_keys) != 1:
            raise ValueError("Ambiguous tail key, please pass tail_key explicitly")
        tail_key = tail_keys[0]

    tail1 = summary1[tail_key]
    tail2 = summary2[tail_key]
    skew1 = summary1["skew"]
    skew2 = summary2["skew"]

    return {
        "tail_key": tail_key,
        "tail_thresh": min(tail1, tail2),
        "skew_thresh": min(skew1, skew2),  # more negative (or smaller) skew
    }


def build_crash_signature_loose(summary1, summary2, tail_key=None):
    """
    Legacy/looser signature: use averages of tail/skew from crash 1 and 2.
    """
    if tail_key is None:
        tail_keys = [k for k in summary1.keys() if k.startswith("tail_prob_<")]
        if len(tail_keys) != 1:
            raise ValueError("Ambiguous tail key, please pass tail_key explicitly")
        tail_key = tail_keys[0]

    tail1 = summary1[tail_key]
    tail2 = summary2[tail_key]
    skew1 = summary1["skew"]
    skew2 = summary2["skew"]

    return {
        "tail_key": tail_key,
        "tail_thresh": (tail1 + tail2) / 2.0,
        "skew_thresh": (skew1 + skew2) / 2.0,
    }

def classify_crash(summary, signature):
    """
    Given a crash summary and a signature (from build_crash_signature),
    return:
      - is_crash_like: bool
      - score: float (smaller => closer to signature)
      - components: tail_diff, skew_diff
    """
    tail_key = signature["tail_key"]
    tail_val = summary[tail_key]
    skew_val = summary["skew"]

    tail_diff = abs(tail_val - signature["tail_thresh"])
    skew_diff = abs(skew_val - signature["skew_thresh"])
    score = tail_diff + skew_diff

    # Heuristic classification rule:
    # Require tail at least the signature minimum and skew no higher than the
    # more negative of the two calibration crashes.
    is_crash_like = (tail_val >= signature["tail_thresh"]) and (skew_val <= signature["skew_thresh"])

    return {
        "is_crash_like": bool(is_crash_like),
        "score": float(score),
        "tail_val": float(tail_val),
        "skew_val": float(skew_val),
        "tail_thresh": float(signature["tail_thresh"]),
        "skew_thresh": float(signature["skew_thresh"]),
    }


def classify_crash_loose(summary, signature):
    """
    Legacy/looser rule: tail must be at least half the signature tail,
    skew must be no higher than the signature skew.
    """
    tail_key = signature["tail_key"]
    tail_val = summary[tail_key]
    skew_val = summary["skew"]

    tail_diff = abs(tail_val - signature["tail_thresh"])
    skew_diff = abs(skew_val - signature["skew_thresh"])
    score = tail_diff + skew_diff

    is_crash_like = (
        (tail_val >= 0.5 * signature["tail_thresh"])
        and (skew_val <= signature["skew_thresh"])
    )

    return {
        "is_crash_like": bool(is_crash_like),
        "score": float(score),
        "tail_val": float(tail_val),
        "skew_val": float(skew_val),
        "tail_thresh": float(signature["tail_thresh"]),
        "skew_thresh": float(signature["skew_thresh"]),
    }

def plot_crash_metrics(summaries, names, tail_key=None, save_path=None):
    if tail_key is None:
        tail_keys = [k for k in summaries[0].keys() if k.startswith("tail_prob_<")]
        if len(tail_keys) != 1:
            raise ValueError("Ambiguous tail key")
        tail_key = tail_keys[0]

    tail_vals = [s[tail_key] for s in summaries]
    skew_vals = [s["skew"] for s in summaries]

    x = np.arange(len(names))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(x - 0.15, tail_vals, width=0.3, label="Tail prob")
    ax2.bar(x + 0.15, skew_vals, width=0.3, label="Skew", alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Tail prob (<0.8 S0)")
    ax2.set_ylabel("Skewness")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")

    plt.title("Crash metrics comparison")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# crash_signature.py



def load_tree_summary(path):
    """
    Load a single crash tree summary JSON produced
    """
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return data

def load_multiple_summaries(summary_dir, crash_ids):
    """
    summary_dir: folder with JSONs like 'Crash1_summary.json'
    crash_ids: list of strings, e.g. ["Crash1", "Crash2", "Crash3"]
    """
    summary_dir = Path(summary_dir)
    summaries = []
    for cid in crash_ids:
        fp = summary_dir / f"{cid}_summary.json"
        summaries.append(load_tree_summary(fp))
    return summaries

def infer_tail_key(summary_dict):
    tail_keys = [k for k in summary_dict.keys() if k.startswith("tail_prob_<")]
    if len(tail_keys) != 1:
        raise ValueError(f"Could not uniquely infer tail key, got: {tail_keys}")
    return tail_keys[0]


def compute_feature_vector(summary, tail_key=None, use_std=True, use_var=False, use_cvar=False):
    if tail_key is None:
        tail_key = infer_tail_key(summary)
    feats = {
        "tail": summary[tail_key],
        "skew": summary["skew"],
    }
    if use_std:
        feats["std"] = summary["std"]
    if use_var:
        # use VaR if present
        var_keys = [k for k in summary.keys() if k.lower().startswith("var_")]
        if var_keys:
            feats["var"] = summary[var_keys[0]]
    if use_cvar:
        cvar_keys = [k for k in summary.keys() if k.lower().startswith("cvar_")]
        if cvar_keys:
            feats["cvar"] = summary[cvar_keys[0]]
    return feats

def feature_distance(feats, center, weights=None):
    """
    feats, center: dicts with same keys
    weights: dict of weights per key or None (all weight 1)
    """
    if weights is None:
        weights = {k: 1.0 for k in feats.keys()}
    sq = 0.0
    for k, v in feats.items():
        diff = v - center[k]
        sq += weights[k] * diff*diff
    return np.sqrt(sq)


def classify_crash_feature_space(summary, feature_signature, threshold=None, weights=None):
    tail_key = feature_signature["tail_key"]
    use_std = feature_signature["use_std"]
    use_var = feature_signature.get("use_var", False)
    use_cvar = feature_signature.get("use_cvar", False)
    center = feature_signature["center"]
    if threshold is None:
        threshold = feature_signature.get("threshold")

    feats = compute_feature_vector(summary, tail_key, use_std, use_var, use_cvar)
    dist = feature_distance(feats, center, weights)

    # Heuristic threshold: if not provided, set to 2 * average distance of Crash1 & 2 from center
    if threshold is None:
        threshold = 0.0  # explicit default if none supplied

    return dist, dist <= threshold, feats

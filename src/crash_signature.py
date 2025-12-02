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
    is_crash_like = (
        (tail_val >= 0.5 * signature["tail_thresh"]) and  # enough left tail
        (skew_val <= signature["skew_thresh"])            # at least as negatively skewed
    )

    return {
        "is_crash_like": bool(is_crash_like),
        "score": float(score),
        "tail_val": float(tail_val),
        "skew_val": float(skew_val),
        "tail_thresh": float(signature["tail_thresh"]),
        "skew_thresh": float(signature["skew_thresh"]),
    }

def plot_crash_metrics(summaries, names, tail_key=None):
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


def compute_feature_vector(summary, tail_key=None, use_std=True):
    if tail_key is None:
        tail_key = infer_tail_key(summary)
    feats = {
        "tail": summary[tail_key],
        "skew": summary["skew"],
    }
    if use_std:
        feats["std"] = summary["std"]
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
    center = feature_signature["center"]

    feats = compute_feature_vector(summary, tail_key, use_std)
    dist = feature_distance(feats, center, weights)

    # Heuristic threshold: if not provided, set to 2 * average distance of Crash1 & 2 from center
    if threshold is None:
        # For the two building crashes, distance to center equals same value,
        # but if you want you can pass threshold explicitly in the notebook.
        threshold = 0.0  # placeholder, real value set in notebook

    return dist, dist <= threshold, feats

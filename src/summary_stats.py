from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:35:56 2025

@author: atharva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary statistics for terminal distributions from the implied tree.

Now we compute statistics on returns rather than prices:
    R_T = S_T / S0 - 1
Tail probability is defined on returns via the same tail_m input
(e.g., tail_m=0.8 → tail return threshold = -0.2).
"""


from typing import Dict
import numpy as np


def compute_summary_stats(
    S_T: np.ndarray,
    probs: np.ndarray,
    S0: float,
    tail_m: float = 0.8,
) -> Dict[str, float]:
    """
    Compute summary statistics on returns R_T = S_T / S0 - 1.

    Parameters
    ----------
    S_T : np.ndarray
        Terminal prices (shape (n,))
    probs : np.ndarray
        Probabilities associated with S_T (shape (n,))
    S0 : float
        Spot / initial underlying price.
    tail_m : float, optional
        Tail threshold as a multiple of S0 (default 0.8). Internally this
        maps to a return threshold of (tail_m - 1), e.g. -0.2 for tail_m=0.8.

    Returns
    -------
    stats : dict
        Dictionary with keys:
            - 'S0'
            - 'tail_return_threshold'
            - 'tail_prob'  (on returns)
            - 'mean_ret'
            - 'var_ret'
            - 'std_ret'
            - 'skew_ret'
            - 'kurt_ret'
            - 'excess_kurt_ret'
    """
    S_T = np.asarray(S_T, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()

    if S_T.shape != probs.shape:
        raise ValueError("S_T and probs must have the same shape.")

    if S_T.size == 0:
        raise ValueError("S_T and probs cannot be empty.")

    # Normalize probabilities to sum to 1 (defensive; tree should already do this)
    total_prob = probs.sum()
    if total_prob <= 0:
        raise ValueError("Total probability must be positive.")
    probs = probs / total_prob

    # Compute returns
    returns = S_T / float(S0) - 1.0

    # Tail probability on returns
    tail_return_threshold = float(tail_m - 1.0)
    tail_mask = returns < tail_return_threshold
    tail_prob = float(probs[tail_mask].sum())

    # Moments on returns
    mean_ret = float(np.sum(returns * probs))
    var_ret = float(np.sum(probs * (returns - mean_ret) ** 2))
    std_ret = float(np.sqrt(var_ret))

    if std_ret > 0:
        z = (returns - mean_ret) / std_ret
        skew_ret = float(np.sum(probs * z ** 3))
        kurt_ret = float(np.sum(probs * z ** 4))
        excess_kurt_ret = float(kurt_ret - 3.0)
    else:
        skew_ret = np.nan
        kurt_ret = np.nan
        excess_kurt_ret = np.nan

    stats = {
        "S0": float(S0),
        "tail_return_threshold": tail_return_threshold,
        "tail_prob": tail_prob,
        "mean_ret": mean_ret,
        "var_ret": var_ret,
        "std_ret": std_ret,
        "skew_ret": skew_ret,
        "kurt_ret": kurt_ret,
        "excess_kurt_ret": excess_kurt_ret,
    }
    return stats

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

Day 1 goal for Person 2:
- Given a discrete terminal distribution (S_T, probs) and spot S0,
  compute tail probabilities, moments, skew, kurtosis, etc.
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
    Compute summary statistics for a discrete terminal distribution.

    Parameters
    ----------
    S_T : np.ndarray
        Terminal prices (shape (n,))
    probs : np.ndarray
        Probabilities associated with S_T (shape (n,))
    S0 : float
        Spot / initial underlying price.
    tail_m : float, optional
        Tail threshold as a multiple of S0 (default 0.8).

    Returns
    -------
    stats : dict
        Dictionary with keys:
            - 'S0'
            - 'tail_threshold'
            - 'tail_prob'
            - 'mean_ST'
            - 'var_ST'
            - 'std_ST'
            - 'skew_ST'
            - 'kurt_ST'
            - 'excess_kurt_ST'
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

    # Tail probability
    tail_threshold = float(tail_m * S0)
    tail_mask = S_T < tail_threshold
    tail_prob = float(probs[tail_mask].sum())

    # Moments
    mean_ST = float(np.sum(S_T * probs))
    var_ST = float(np.sum(probs * (S_T - mean_ST) ** 2))
    std_ST = float(np.sqrt(var_ST))

    if std_ST > 0:
        z = (S_T - mean_ST) / std_ST
        skew_ST = float(np.sum(probs * z ** 3))
        kurt_ST = float(np.sum(probs * z ** 4))
        excess_kurt_ST = float(kurt_ST - 3.0)
    else:
        skew_ST = np.nan
        kurt_ST = np.nan
        excess_kurt_ST = np.nan

    stats = {
        "S0": float(S0),
        "tail_threshold": tail_threshold,
        "tail_prob": tail_prob,
        "mean_ST": mean_ST,
        "var_ST": var_ST,
        "std_ST": std_ST,
        "skew_ST": skew_ST,
        "kurt_ST": kurt_ST,
        "excess_kurt_ST": excess_kurt_ST,
    }
    return stats

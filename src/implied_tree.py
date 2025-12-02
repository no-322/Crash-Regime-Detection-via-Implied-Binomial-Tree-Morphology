#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:50:03 2025

@author: atharva
"""

# implied_tree.py

from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np


def build_local_vol_binomial_terminal(
    S0: float,
    r: float,
    T: float,
    N: int,
    sigma_of_m: Callable[[float], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a non-recombining binomial tree with local volatility sigma(m = S/S0),
    and return the terminal price distribution.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    r : float
        Risk-free rate (continuous compounding).
    T : float
        Time to maturity (in years).
    N : int
        Number of binomial time steps.
    sigma_of_m : callable
        Function mapping moneyness (S/S0) to local volatility sigma.

    Returns
    -------
    terminal_prices : np.ndarray, shape (2**N,)
        Asset prices at maturity for each path.
    terminal_probs : np.ndarray, shape (2**N,)
        Probability of each terminal price (paths weights).
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    dt = T / N
    disc = np.exp(r * dt)

    # List of (price, probability) nodes at current time step
    nodes: List[Tuple[float, float]] = [(S0, 1.0)]

    for _step in range(N):
        new_nodes: List[Tuple[float, float]] = []

        for S, prob in nodes:
            if prob == 0.0:
                continue

            m = S / S0
            sigma_loc = sigma_of_m(m)

            # Up/down factors and risk-neutral probability at this node
            u = float(np.exp(sigma_loc * np.sqrt(dt)))
            d = 1.0 / u

            # Safety: ensure u != d to avoid division by zero
            if np.isclose(u, d):
                raise ValueError("Up and down factors are equal; check sigma or dt.")

            p_up = (disc - d) / (u - d)
            p_down = 1.0 - p_up

            # (Optional) clamp p to [0, 1] to avoid numerical nastiness
            p_up = float(np.clip(p_up, 0.0, 1.0))
            p_down = float(np.clip(p_down, 0.0, 1.0))

            # Generate children
            S_up = S * u
            S_down = S * d

            new_nodes.append((S_up, prob * p_up))
            new_nodes.append((S_down, prob * p_down))

        nodes = new_nodes

    # Extract terminal arrays
    terminal_prices = np.array([S for (S, _) in nodes], dtype=float)
    terminal_probs = np.array([p for (_, p) in nodes], dtype=float)

    # Normalize probabilities to sum to 1 (clean up numeric drift)
    total_prob = terminal_probs.sum()
    if total_prob <= 0:
        raise ValueError("Total probability is non-positive; something went wrong.")
    terminal_probs /= total_prob

    return terminal_prices, terminal_probs

def build_tree_from_smile(
    S0: float,
    r: float,
    T: float,
    N: int,
    sigma_func,
):
    """
    Wrapper for compatibility with project spec.

    Parameters
    ----------
    S0, r, T, N : as in build_local_vol_binomial_terminal
    sigma_func : callable
        Function sigma(m) returning local volatility given moneyness.

    Returns
    -------
    (S_T, probs) : tuple of np.ndarray
        Terminal prices and associated risk-neutral probabilities.
    """
    return build_local_vol_binomial_terminal(S0, r, T, N, sigma_func)

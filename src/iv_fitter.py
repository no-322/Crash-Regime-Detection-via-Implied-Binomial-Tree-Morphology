#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:39:31 2025

@author: atharva
"""

# iv_fitter.py

from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


def fit_iv_spline(
    m_points: Iterable[float],
    iv_points: Iterable[float],
    spline_degree: int = 2,
) -> InterpolatedUnivariateSpline:
    """
    Fit a spline implied volatility smile sigma(m).

    Parameters
    ----------
    m_points : iterable of float
        Grid of moneyness values (e.g. K/S0 or S/S0) in ascending order.
    iv_points : iterable of float
        Implied volatilities corresponding to m_points.
    spline_degree : int, optional
        Degree of the spline; we default to 2 (quadratic) per request.

    Returns
    -------
    InterpolatedUnivariateSpline
        A callable spline object sigma(m) that maps moneyness to IV.
    """
    m_arr = np.asarray(m_points, dtype=float)
    iv_arr = np.asarray(iv_points, dtype=float)

    if m_arr.ndim != 1 or iv_arr.ndim != 1:
        raise ValueError("m_points and iv_points must be 1D arrays.")
    if m_arr.shape[0] != iv_arr.shape[0]:
        raise ValueError("m_points and iv_points must have the same length.")

    # SciPy expects strictly increasing x-values
    sort_idx = np.argsort(m_arr)
    m_sorted = m_arr[sort_idx]
    iv_sorted = iv_arr[sort_idx]

    spline = InterpolatedUnivariateSpline(m_sorted, iv_sorted, k=spline_degree)
    return spline


def build_dummy_smile() -> InterpolatedUnivariateSpline:
    """
    Convenience helper for your Day 1 dummy data.

    Returns
    -------
    InterpolatedUnivariateSpline
        Quadratic spline for m = [0.8,0.9,1.0,1.1,1.2], iv = [0.35,0.30,0.25,0.28,0.32]
    """
    m = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    iv = np.array([0.35, 0.30, 0.25, 0.28, 0.32])
    return fit_iv_spline(m, iv)

def plot_dummy_smile(num_points: int = 200) -> None:
    """
    Plot the quadratic-spline IV smile for the Day 1 dummy data.
    """
    # build the spline
    sigma = build_dummy_smile()

    # original nodes (for scatter plot)
    m_nodes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    iv_nodes = np.array([0.35, 0.30, 0.25, 0.28, 0.32])

    # dense grid for a smooth curve
    m_grid = np.linspace(m_nodes.min(), m_nodes.max(), num_points)
    iv_grid = sigma(m_grid)

    # plot
    plt.figure()
    plt.plot(m_grid, iv_grid, label="Quadratic-spline IV smile")
    plt.scatter(m_nodes, iv_nodes, marker="o", label="Input points")
    plt.xlabel("Moneyness  m = K / S0")
    plt.ylabel("Implied volatility")
    plt.title("Dummy IV Smile (Quadratic Spline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

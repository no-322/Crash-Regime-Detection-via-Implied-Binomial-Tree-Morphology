#!/usr/bin/env python3
"""
Simple helpers to build and plot the non-recombining local-vol binomial lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LatticeNode:
    step: int
    idx: int
    S: float
    prob: float


def build_local_vol_lattice(
    S0: float,
    r: float,
    T: float,
    N: int,
    sigma_of_m: Callable[[float], float],
) -> Tuple[List[List[LatticeNode]], List[Tuple[LatticeNode, LatticeNode]]]:
    """
    Build the non-recombining binomial lattice and return nodes grouped by step
    along with edges for plotting.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    dt = T / N
    disc = np.exp(r * dt)

    levels: List[List[LatticeNode]] = [[LatticeNode(step=0, idx=0, S=S0, prob=1.0)]]
    edges: List[Tuple[LatticeNode, LatticeNode]] = []

    for step in range(N):
        current = levels[-1]
        next_level: List[LatticeNode] = []
        for node in current:
            m = node.S / S0
            sigma_loc = sigma_of_m(m)
            u = float(np.exp(sigma_loc * np.sqrt(dt)))
            d = 1.0 / u
            p_up = (disc - d) / (u - d)
            p_up = float(np.clip(p_up, 0.0, 1.0))
            p_down = 1.0 - p_up

            idx_base = node.idx * 2
            up_node = LatticeNode(step=step + 1, idx=idx_base, S=node.S * u, prob=node.prob * p_up)
            dn_node = LatticeNode(step=step + 1, idx=idx_base + 1, S=node.S * d, prob=node.prob * p_down)

            next_level.extend([up_node, dn_node])
            edges.append((node, up_node))
            edges.append((node, dn_node))
        levels.append(next_level)

    # normalize terminal probs (small drift can accumulate)
    terminal_probs = sum(n.prob for n in levels[-1])
    if terminal_probs > 0:
        for n in levels[-1]:
            n.prob /= terminal_probs

    return levels, edges


def plot_binomial_tree_lattice(
    lattice: Tuple[List[List[LatticeNode]], List[Tuple[LatticeNode, LatticeNode]]],
    S0: float,
    out_path: str,
) -> None:
    """
    Plot lattice nodes and edges, using log-price on y to keep spacing reasonable.
    """
    levels, edges = lattice

    fig, ax = plt.subplots(figsize=(10, 6))

    # draw edges
    for parent, child in edges:
        ax.plot(
            [parent.step, child.step],
            [np.log(parent.S), np.log(child.S)],
            color="lightgray",
            linewidth=0.8,
        )

    # draw nodes
    for level in levels:
        xs = [n.step for n in level]
        ys = [np.log(n.S) for n in level]
        sizes = [50 + 200 * n.prob for n in level]
        ax.scatter(xs, ys, s=sizes, alpha=0.7, color="tab:blue", edgecolors="k", linewidths=0.3)

    ax.axhline(np.log(S0), color="black", linestyle="--", linewidth=1, label="log S0")
    ax.set_xlabel("Step")
    ax.set_ylabel("log Price")
    ax.set_title("Local-vol binomial lattice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

"""
City graph with Markov-chain traffic dynamics.

States: FREE=0, MODERATE=1, CONGESTED=2, BLOCKED=3
Transition matrix rows = current state, cols = next state.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

FREE, MODERATE, CONGESTED, BLOCKED = 0, 1, 2, 3

TRAFFIC_LABELS      = ["Libre", "Moderado", "Congestionado", "Bloqueado"]
TRAFFIC_COLORS      = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
TRAFFIC_MULTIPLIERS = [1.0, 1.8, 3.5, 10.0]

# P[i][j] = P(next=j | current=i)
TRANSITION_MATRIX = np.array([
    [0.70, 0.20, 0.08, 0.02],   # FREE
    [0.30, 0.50, 0.15, 0.05],   # MODERATE
    [0.10, 0.30, 0.50, 0.10],   # CONGESTED
    [0.20, 0.30, 0.30, 0.20],   # BLOCKED
], dtype=float)


class CityGraph:
    """
    N×N grid city.  Nodes: integers 0..(N²-1).
    Each edge carries a base travel time and a Markov traffic state.
    """

    def __init__(self, grid_size: int = 7, seed: int = 42):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

        # 2D position for visualisation: node id → (col, row)
        self.pos_2d: Dict[int, Tuple[float, float]] = {
            r * grid_size + c: (float(c), float(r))
            for r in range(grid_size)
            for c in range(grid_size)
        }

        # Build graph
        G_grid = nx.grid_2d_graph(grid_size, grid_size)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(grid_size * grid_size))

        for (r1, c1), (r2, c2) in G_grid.edges():
            u = r1 * grid_size + c1
            v = r2 * grid_size + c2
            base = 1.0 + float(self.rng.exponential(0.4))
            self.G.add_edge(u, v, base_time=base)

        self.n_nodes = self.G.number_of_nodes()

        # Traffic states — start mostly free
        self.traffic: Dict[Tuple[int, int], int] = {}
        for u, v in self.G.edges():
            state = int(self.rng.choice(4, p=[0.60, 0.25, 0.12, 0.03]))
            self.traffic[self._key(u, v)] = state

    # ------------------------------------------------------------------ helpers

    def _key(self, u: int, v: int) -> Tuple[int, int]:
        return (min(u, v), max(u, v))

    def travel_time(self, u: int, v: int) -> float:
        base  = self.G[u][v]["base_time"]
        state = self.traffic[self._key(u, v)]
        return base * TRAFFIC_MULTIPLIERS[state]

    # ------------------------------------------------------------------ dynamics

    def update_traffic(self) -> None:
        """Advance every edge one step of the Markov chain."""
        for key in self.traffic:
            cur = self.traffic[key]
            self.traffic[key] = int(self.rng.choice(4, p=TRANSITION_MATRIX[cur]))

    def trigger_accident(self) -> Tuple[int, int]:
        """Force a random edge to BLOCKED and return that edge key."""
        edges = list(self.traffic.keys())
        edge  = edges[int(self.rng.integers(len(edges)))]
        self.traffic[edge] = BLOCKED
        return edge

    # ------------------------------------------------------------------ routing

    def shortest_path(self, src: int, dst: int) -> Tuple[List[int], float]:
        if src == dst:
            return [src], 0.0
        for u, v in self.G.edges():
            self.G[u][v]["weight"] = self.travel_time(u, v)
        try:
            path = nx.dijkstra_path(self.G, src, dst, weight="weight")
            cost = nx.dijkstra_path_length(self.G, src, dst, weight="weight")
            return path, float(cost)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    # ------------------------------------------------------------------ metrics

    def traffic_distribution(self) -> List[int]:
        counts = [0, 0, 0, 0]
        for s in self.traffic.values():
            counts[s] += 1
        return counts

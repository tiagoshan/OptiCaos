"""
Ant Colony Optimisation — multi-stop Vehicle Routing.

Each call to optimize_route() runs `n_iterations` of `n_ants` ants.
Ants choose which waypoint to visit next using:

    p(i→j) = τ(i,j)^α · η(i,j)^β  /  Σ_k τ(i,k)^α · η(i,k)^β

where τ = pheromone, η = 1/distance (heuristic).
After each iteration pheromones evaporate and are reinforced:

    τ ← (1-ρ)·τ + Σ_k  Q / L_k
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from city import CityGraph


class ACORouter:
    def __init__(
        self,
        city: CityGraph,
        n_ants: int = 20,
        n_iterations: int = 35,
        alpha: float = 1.0,    # pheromone weight
        beta: float = 2.5,     # heuristic weight
        rho: float = 0.15,     # evaporation rate
        Q: float = 100.0,      # pheromone deposit constant
    ):
        self.city = city
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        n = city.n_nodes
        self.pheromones: np.ndarray = np.full((n, n), 0.1, dtype=float)
        self.convergence: List[float] = []   # best cost per iteration

    # ------------------------------------------------------------------ public

    def optimize_route(
        self, start: int, waypoints: List[int]
    ) -> Tuple[List[int], float]:
        """
        Return (full_path_as_node_list, total_travel_cost).
        Visits every node in `waypoints` exactly once, starting from `start`.
        """
        if not waypoints:
            return [start], 0.0
        if len(waypoints) == 1:
            return self.city.shortest_path(start, waypoints[0])

        # Pre-compute all pairwise shortest paths among relevant nodes
        nodes = list(dict.fromkeys([start] + waypoints))
        cache: Dict[Tuple[int, int], Tuple[List[int], float]] = {}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                path, cost = self.city.shortest_path(u, v)
                cache[(u, v)] = (path, cost)
                cache[(v, u)] = (list(reversed(path)), cost)

        best_order: Optional[List[int]] = None
        best_cost = float("inf")

        for _iter in range(self.n_iterations):
            solutions: List[Tuple[List[int], float]] = []

            for _ in range(self.n_ants):
                remaining = list(waypoints)
                current   = start
                order: List[int] = []
                cost = 0.0

                while remaining:
                    probs = np.zeros(len(remaining))
                    for i, dest in enumerate(remaining):
                        _, d = cache.get((current, dest), ([], float("inf")))
                        if d < float("inf"):
                            tau = self.pheromones[current][dest] ** self.alpha
                            eta = (1.0 / d) ** self.beta
                            probs[i] = tau * eta

                    total = probs.sum()
                    if total < 1e-12:
                        probs = np.ones(len(remaining)) / len(remaining)
                    else:
                        probs /= total

                    idx = int(np.random.choice(len(remaining), p=probs))
                    nxt = remaining.pop(idx)
                    _, seg = cache.get((current, nxt), ([], float("inf")))
                    cost += seg
                    order.append(nxt)
                    current = nxt

                solutions.append((order, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_order = order

            # --- pheromone update ---
            self.pheromones *= (1.0 - self.rho)
            self.pheromones = np.clip(self.pheromones, 0.01, 200.0)

            for order, c in solutions:
                if 0 < c < float("inf"):
                    deposit = self.Q / c
                    prev = start
                    for wp in order:
                        self.pheromones[prev][wp] += deposit
                        self.pheromones[wp][prev] += deposit
                        prev = wp

            self.convergence.append(best_cost)

        if best_order is None:
            return [start], float("inf")

        # Stitch together the full node-by-node path
        full_path = [start]
        current = start
        for wp in best_order:
            segment, _ = cache.get((current, wp), ([current, wp], 1.0))
            full_path.extend(segment[1:])
            current = wp

        return full_path, best_cost

    # ------------------------------------------------------------------ viz helper

    def top_pheromone_edges(self, top_n: int = 20) -> List[Tuple[int, int, float]]:
        """Return the top-N edges by pheromone intensity (for golden overlay)."""
        result = [
            (u, v, float(self.pheromones[u][v]))
            for u, v in self.city.G.edges()
        ]
        result.sort(key=lambda x: -x[2])
        return result[:top_n]

"""
Stochastic simulation engine.

Order arrivals   → Poisson(λ)  each time step
Traffic changes  → Markov chain (see city.py)
Accidents        → Bernoulli(p=0.04) each step  →  re-trigger ACO for all deliverers
Assignment       → greedy nearest-deliverer heuristic
Movement         → discrete edge traversal with fractional progress
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from city import CityGraph


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    id:           int
    origin:       int
    destination:  int
    created_at:   float
    deadline:     float
    assigned_to:  int   = -1
    picked_up:    bool  = False
    delivered:    bool  = False
    pickup_time:  float = -1.0
    delivery_time: float = -1.0


@dataclass
class Deliverer:
    id:              int
    position:        int
    route:           List[int] = field(default_factory=list)
    assigned_orders: List[int] = field(default_factory=list)
    deliveries_done: int   = 0
    time_moving:     float = 0.0


@dataclass
class SimEvent:
    time:     float
    etype:    str
    msg:      str
    severity: str = "info"   # info | warning | critical


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class SimEngine:
    def __init__(
        self,
        city: CityGraph,
        n_deliverers: int  = 4,
        lambda_orders: float = 0.3,
        accident_prob: float = 0.04,
        seed: int = 99,
    ):
        self.city          = city
        self.lambda_orders = lambda_orders
        self.accident_prob = accident_prob
        self.rng           = np.random.default_rng(seed)
        self.dt            = 1.0

        self.time: float = 0.0
        self._order_id   = 0

        self.orders:    Dict[int, Order] = {}
        self.pending:   List[int] = []   # unassigned
        self.active:    List[int] = []   # assigned, in transit
        self.completed: List[int] = []

        self.events: List[SimEvent] = []

        # ── statistics ──
        self.total_delivered  = 0
        self.on_time          = 0
        self.delivery_times:  List[float] = []
        self.accidents_count  = 0
        self.recalculations   = 0
        self.orders_per_step: List[int]   = []

        # ── deliverers ──
        starts = self.rng.choice(city.n_nodes, size=n_deliverers, replace=False)
        self.deliverers: Dict[int, Deliverer] = {
            i: Deliverer(id=i, position=int(starts[i]))
            for i in range(n_deliverers)
        }

    # ──────────────────────────────────────────── private helpers

    def _log(self, etype: str, msg: str, severity: str = "info") -> None:
        self.events.append(SimEvent(self.time, etype, msg, severity))
        if len(self.events) > 120:
            self.events = self.events[-120:]

    def _new_order(self) -> Order:
        n    = self.city.n_nodes
        orig = int(self.rng.integers(n))
        dest = int(self.rng.integers(n))
        while dest == orig:
            dest = int(self.rng.integers(n))
        o = Order(
            id          = self._order_id,
            origin      = orig,
            destination = dest,
            created_at  = self.time,
            deadline    = self.time + 15.0 + float(self.rng.exponential(8.0)),
        )
        self._order_id += 1
        return o

    def _recalc(self, did: int, optimizer) -> None:
        """Rebuild route for deliverer using sequential Dijkstra.

        Waypoints are always in pickup-before-delivery order so the ACO
        ordering bug (dest before origin) cannot occur.
        """
        d = self.deliverers[did]

        # Build waypoints respecting pickup constraint:
        # for each order: origin (if not yet picked up) → destination
        waypoints: List[int] = []
        for oid in d.assigned_orders:
            o = self.orders[oid]
            if not o.picked_up and d.position != o.origin:
                waypoints.append(o.origin)
            waypoints.append(o.destination)

        if not waypoints:
            d.route = [d.position]
            return

        # ACO: updates pheromones + convergence data for the chart/overlay.
        # We run it but do NOT use its suggested order — the pickup constraint
        # (origin before destination) must be respected, which ACO can violate.
        optimizer.optimize_route(d.position, waypoints)

        # Sequential Dijkstra with guaranteed correct pickup ordering
        route   = [d.position]
        current = d.position
        for wp in waypoints:
            if wp == current:
                continue
            seg, _ = self.city.shortest_path(current, wp)
            if seg:
                route.extend(seg[1:])
                current = wp

        d.route = route
        self.recalculations += 1

    def _assign_pending(self, optimizer) -> None:
        for oid in list(self.pending):
            o = self.orders[oid]
            best_d, best_score = -1, float("inf")

            for did, d in self.deliverers.items():
                if len(d.assigned_orders) >= 3:
                    continue
                _, dist = self.city.shortest_path(d.position, o.origin)
                score   = dist + len(d.assigned_orders) * 4.0
                if score < best_score:
                    best_score = score
                    best_d     = did

            if best_d >= 0:
                o.assigned_to = best_d
                self.deliverers[best_d].assigned_orders.append(oid)
                self.pending.remove(oid)
                self.active.append(oid)
                self._recalc(best_d, optimizer)
                self._log("ASIGNACIÓN", f"Pedido #{oid} → Repartidor {best_d}")

    # ──────────────────────────────────────────── public API

    def step(self, optimizer, force_accident: bool = False) -> None:
        """Advance simulation one time step."""
        self.time += self.dt

        # 1. Traffic / accidents
        if force_accident or self.rng.random() < self.accident_prob:
            edge = self.city.trigger_accident()
            self.accidents_count += 1
            self._log("ACCIDENTE", f"¡Colisión en arista {edge}! Recalculando rutas…", "critical")
            for did in self.deliverers:
                self._recalc(did, optimizer)
        else:
            self.city.update_traffic()

        # 2. New orders — Poisson arrival
        n_new = int(self.rng.poisson(self.lambda_orders))
        self.orders_per_step.append(n_new)
        for _ in range(n_new):
            o = self._new_order()
            self.orders[o.id] = o
            self.pending.append(o.id)
            self._log("PEDIDO", f"#{o.id}: nodo {o.origin} → {o.destination}")

        # 3. Assign unassigned orders
        self._assign_pending(optimizer)

        # 4. Move deliverers — one node per step (traffic affects route choice, not speed)
        for did, d in self.deliverers.items():
            if len(d.route) < 2:
                continue

            d.position    = d.route[1]
            d.route       = d.route[1:]
            d.time_moving += self.dt

            for oid in list(d.assigned_orders):
                o = self.orders[oid]

                if not o.picked_up and d.position == o.origin:
                    o.picked_up   = True
                    o.pickup_time = self.time
                    self._log("RECOGIDA", f"R{did} recogió pedido #{oid}")

                elif o.picked_up and not o.delivered and d.position == o.destination:
                    o.delivered    = True
                    o.delivery_time = self.time
                    d.assigned_orders.remove(oid)
                    self.active.remove(oid)
                    self.completed.append(oid)
                    d.deliveries_done += 1

                    elapsed  = o.delivery_time - o.created_at
                    in_time  = o.delivery_time <= o.deadline
                    self.total_delivered += 1
                    self.delivery_times.append(elapsed)
                    if in_time:
                        self.on_time += 1
                        self._log("ENTREGA ✓", f"#{oid} en {elapsed:.1f}u (a tiempo)")
                    else:
                        self._log("ENTREGA ✗", f"#{oid} en {elapsed:.1f}u (tarde)", "warning")

"""
FastAPI + WebSocket backend para el Optimizador de Caos.

El loop de simulación corre como tarea asyncio en el fondo.
El cliente envía comandos por WebSocket; el servidor pushea snapshots completos.
"""

import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from city import CityGraph
from aco import ACORouter
from simulation import SimEngine

# ──────────────────────────────────────────────────────────────────────────────

DELIVERER_COLORS = [
    "#e91e63", "#00bcd4", "#cddc39", "#ff5722",
    "#9c27b0", "#03a9f4", "#8bc34a", "#ff9800",
]


def make_sim(grid_size=7, n_deliverers=4, lambda_orders=0.3, seed=42) -> dict:
    city      = CityGraph(grid_size=grid_size, seed=seed)
    optimizer = ACORouter(city, n_ants=20, n_iterations=30)
    sim       = SimEngine(city, n_deliverers=n_deliverers,
                          lambda_orders=lambda_orders, seed=seed + 1)
    return {"city": city, "optimizer": optimizer, "sim": sim,
            "running": False, "speed": 0.6, "step": 0}


STATE: dict = make_sim()
CLIENTS: Set[WebSocket] = set()

# ──────────────────────────────────────────────────────────────────────────────

def snapshot() -> dict:
    city = STATE["city"]
    sim  = STATE["sim"]
    opt  = STATE["optimizer"]

    edges = [
        [float(city.pos_2d[u][0]), float(city.pos_2d[u][1]),
         float(city.pos_2d[v][0]), float(city.pos_2d[v][1]),
         int(city.traffic[city._key(u, v)])]
        for u, v in city.G.edges()
    ]

    nodes = [[float(city.pos_2d[n][0]), float(city.pos_2d[n][1])]
             for n in city.G.nodes()]

    deliverers = []
    for did, d in sim.deliverers.items():
        cx, cy = city.pos_2d[d.position]
        route  = [[float(city.pos_2d[n][0]), float(city.pos_2d[n][1])]
                  for n in d.route]
        deliverers.append({
            "id":     did,
            "pos":    [float(cx), float(cy)],
            "route":  route,
            "done":   d.deliveries_done,
            "orders": len(d.assigned_orders),
            "color":  DELIVERER_COLORS[did % len(DELIVERER_COLORS)],
        })

    orders = []
    for oid in sim.pending + sim.active:
        o  = sim.orders[oid]
        ox, oy = city.pos_2d[o.origin]
        dx, dy = city.pos_2d[o.destination]
        orders.append({
            "id":     oid,
            "origin": [float(ox), float(oy)],
            "dest":   [float(dx), float(dy)],
            "state":  "pending" if oid in sim.pending else "active",
        })

    phi_raw = opt.top_pheromone_edges(top_n=20)
    max_phi = max((p for *_, p in phi_raw), default=1.0) or 1.0
    pheromones = [
        [float(city.pos_2d[u][0]), float(city.pos_2d[u][1]),
         float(city.pos_2d[v][0]), float(city.pos_2d[v][1]),
         float(phi / max_phi)]
        for u, v, phi in phi_raw
    ]

    lam     = sim.lambda_orders
    theory  = [float((lam**k * math.exp(-lam)) / math.factorial(k)) for k in range(8)]
    n_steps = max(len(sim.orders_per_step), 1)
    observed = [float(sim.orders_per_step.count(k) / n_steps) for k in range(8)]

    avg_t = sum(sim.delivery_times) / len(sim.delivery_times) if sim.delivery_times else 0.0
    pct   = round(100 * sim.on_time / sim.total_delivered) if sim.total_delivered else 0

    return {
        "time":       float(sim.time),
        "step":       int(STATE["step"]),
        "grid_size":  city.grid_size,
        "running":    STATE["running"],
        "edges":      edges,
        "nodes":      nodes,
        "deliverers": deliverers,
        "orders":     orders,
        "pheromones": pheromones,
        "events": [
            {"t": float(e.time), "type": e.etype, "msg": e.msg, "sev": e.severity}
            for e in sim.events[-14:]
        ],
        "metrics": {
            "delivered": sim.total_delivered,
            "on_time":   pct,
            "avg_time":  round(float(avg_t), 1),
            "accidents": sim.accidents_count,
            "backlog":   len(sim.pending) + len(sim.active),
            "recalcs":   sim.recalculations,
        },
        "traffic_dist": [int(x) for x in city.traffic_distribution()],
        "poisson": {"theory": theory, "observed": observed, "lam": lam},
        "convergence":    [float(x) for x in opt.convergence[-80:]],
        "delivery_times": [float(x) for x in sim.delivery_times[-80:]],
    }


async def broadcast(data: dict) -> None:
    dead = set()
    for ws in list(CLIENTS):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    CLIENTS.difference_update(dead)  # in-place mutation, no rebinding


# ──────────────────────────────────────────────────────────────────────────────

async def _sim_loop() -> None:
    while True:
        try:
            if STATE["running"]:
                # Run in thread so Dijkstra doesn't block the async event loop
                await asyncio.to_thread(STATE["sim"].step, STATE["optimizer"])
                STATE["step"] += 1
                await broadcast(snapshot())
        except Exception as exc:
            import traceback
            print(f"[SIM ERROR en paso {STATE['step']}]: {exc}")
            traceback.print_exc()
        await asyncio.sleep(STATE["speed"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_sim_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> HTMLResponse:
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    CLIENTS.add(websocket)
    await websocket.send_json(snapshot())
    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("cmd", "")

            if cmd == "step":
                await asyncio.to_thread(STATE["sim"].step, STATE["optimizer"])
                STATE["step"] += 1
                await broadcast(snapshot())

            elif cmd == "accident":
                await asyncio.to_thread(
                    STATE["sim"].step, STATE["optimizer"], True
                )
                STATE["step"] += 1
                await broadcast(snapshot())

            elif cmd == "toggle":
                STATE["running"] = not STATE["running"]
                await broadcast(snapshot())

            elif cmd == "speed":
                STATE["speed"] = max(0.05, float(msg.get("value", 0.6)))

            elif cmd == "reset":
                p = msg.get("params", {})
                STATE.update(make_sim(
                    grid_size     = int(p.get("grid_size",    7)),
                    n_deliverers  = int(p.get("n_deliverers", 4)),
                    lambda_orders = float(p.get("lambda",     0.3)),
                    seed          = int(p.get("seed",         42)),
                ))
                await broadcast(snapshot())

    except WebSocketDisconnect:
        CLIENTS.discard(websocket)

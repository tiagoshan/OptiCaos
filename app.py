"""
Optimizador de Caos — Simulador de Logística Urbana Estocástica
================================================================
• Tráfico urbano   → Cadenas de Markov (4 estados)
• Llegada pedidos  → Proceso de Poisson(λ)
• Optimización     → Colonia de Hormigas (ACO)  — recalcula en cada evento
"""

import math
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aco import ACORouter
from city import (
    TRAFFIC_COLORS,
    TRAFFIC_LABELS,
    TRANSITION_MATRIX,
    CityGraph,
)
from simulation import SimEngine

# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Optimizador de Caos",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 1rem; }
      .stMetric label  { font-size: 0.75rem !important; }
      div[data-testid="metric-container"] { background:#1e2a3a; border-radius:8px; padding:0.5rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap
# ──────────────────────────────────────────────────────────────────────────────

DELIVERER_COLORS = [
    "#e91e63", "#00bcd4", "#cddc39", "#ff5722",
    "#9c27b0", "#03a9f4", "#8bc34a", "#ff9800",
]


def build_simulation(grid_size, n_deliverers, lambda_orders, seed):
    city      = CityGraph(grid_size=grid_size, seed=seed)
    optimizer = ACORouter(city, n_ants=20, n_iterations=30)
    sim       = SimEngine(city, n_deliverers=n_deliverers,
                          lambda_orders=lambda_orders, seed=seed + 1)
    return city, optimizer, sim


if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.running     = False
    st.session_state.step_count  = 0


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — controls + math
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parámetros")
    st.markdown("---")

    grid_size    = st.slider("Tamaño de ciudad (N×N)", 5, 10, 7, key="grid")
    n_deliverers = st.slider("Repartidores",           2,  8, 4, key="ndeliv")
    lambda_val   = st.slider("λ — tasa Poisson (pedidos/paso)", 0.1, 1.5, 0.3, 0.05, key="lam")
    speed        = st.slider("Segundos por paso (modo auto)",   0.1, 2.0, 0.5, 0.1,  key="spd")
    seed         = st.number_input("Semilla aleatoria", 0, 9999, 42, key="seed")

    st.markdown("---")

    if st.button("🔄 Reiniciar", use_container_width=True, type="primary"):
        city, optimizer, sim = build_simulation(grid_size, n_deliverers, lambda_val, int(seed))
        st.session_state.city      = city
        st.session_state.optimizer = optimizer
        st.session_state.sim       = sim
        st.session_state.initialized = True
        st.session_state.running     = False
        st.session_state.step_count  = 0
        st.rerun()

    # First-time init
    if not st.session_state.initialized:
        city, optimizer, sim = build_simulation(grid_size, n_deliverers, lambda_val, int(seed))
        st.session_state.city      = city
        st.session_state.optimizer = optimizer
        st.session_state.sim       = sim
        st.session_state.initialized = True

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        label = "⏸ Pausa" if st.session_state.running else "▶ Auto"
        if st.button(label, use_container_width=True):
            st.session_state.running = not st.session_state.running
            st.rerun()
    with col_b:
        if st.button("⏭ +1 Paso", use_container_width=True):
            st.session_state.sim.step(st.session_state.optimizer)
            st.session_state.step_count += 1
            st.rerun()

    if st.button("💥 Forzar Accidente", use_container_width=True):
        st.session_state.sim.step(st.session_state.optimizer, force_accident=True)
        st.session_state.step_count += 1
        st.rerun()

    st.markdown("---")
    st.markdown("### 📐 Modelo Matemático")

    st.markdown("**Llegada de pedidos — Poisson:**")
    st.latex(r"P(k;\lambda)=\frac{\lambda^{k}e^{-\lambda}}{k!}")

    st.markdown("**Tráfico — Cadena de Markov:**")
    st.latex(r"P\!\left(S_{t+1}=j\mid S_t=i\right)=\pi_{ij}")

    st.markdown("**ACO — selección de nodo:**")
    st.latex(
        r"p_{ij}=\frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}"
        r"{\displaystyle\sum_{k}\tau_{ik}^{\alpha}\,\eta_{ik}^{\beta}}"
    )

    st.markdown("**ACO — actualización de feromonas:**")
    st.latex(r"\tau_{ij}\leftarrow(1-\rho)\,\tau_{ij}+\sum_{k}\frac{Q}{L_k}")

    st.markdown("---")
    st.caption("ACO: α=1.0 · β=2.5 · ρ=0.15 · Q=100")


# ──────────────────────────────────────────────────────────────────────────────
# Aliases
# ──────────────────────────────────────────────────────────────────────────────

city:      CityGraph  = st.session_state.city
sim:       SimEngine  = st.session_state.sim
optimizer: ACORouter  = st.session_state.optimizer

# ──────────────────────────────────────────────────────────────────────────────
# Header + KPIs
# ──────────────────────────────────────────────────────────────────────────────

st.title("🚚 Optimizador de Caos — Logística Urbana Estocástica")
st.caption(
    f"Tiempo: **{sim.time:.0f} u** &nbsp;|&nbsp; "
    f"Paso: **{st.session_state.step_count}** &nbsp;|&nbsp; "
    f"Ciudad: **{city.grid_size}×{city.grid_size}** &nbsp;|&nbsp; "
    f"Repartidores: **{len(sim.deliverers)}** &nbsp;|&nbsp; "
    f"λ = **{sim.lambda_orders}**"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

pct_time = (
    round(100 * sim.on_time / sim.total_delivered)
    if sim.total_delivered > 0 else 0
)
avg_t = round(np.mean(sim.delivery_times), 1) if sim.delivery_times else 0.0
backlog = len(sim.pending) + len(sim.active)

k1.metric("📦 Entregados",  sim.total_delivered)
k2.metric("✅ A tiempo",    f"{pct_time}%")
k3.metric("⏱ Tiempo medio", f"{avg_t} u")
k4.metric("🚧 Accidentes",  sim.accidents_count)
k5.metric("📬 En espera",   backlog)
k6.metric("🔄 Recalcs ACO", sim.recalculations)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Main layout: map (left) + panels (right)
# ──────────────────────────────────────────────────────────────────────────────

map_col, side_col = st.columns([3, 1])

# ── City map ──────────────────────────────────────────────────────────────────
with map_col:
    fig = go.Figure()

    # Edges grouped by traffic state (4 traces → 4 legend entries)
    edges_by_state: dict = {s: {"x": [], "y": []} for s in range(4)}
    for u, v in city.G.edges():
        s  = city.traffic[city._key(u, v)]
        ux, uy = city.pos_2d[u]
        vx, vy = city.pos_2d[v]
        edges_by_state[s]["x"] += [ux, vx, None]
        edges_by_state[s]["y"] += [uy, vy, None]

    for s in range(4):
        if edges_by_state[s]["x"]:
            fig.add_trace(go.Scatter(
                x=edges_by_state[s]["x"],
                y=edges_by_state[s]["y"],
                mode="lines",
                line=dict(color=TRAFFIC_COLORS[s], width=2.5),
                name=TRAFFIC_LABELS[s],
                hoverinfo="none",
                legendgroup=f"traffic_{s}",
            ))

    # Pheromone overlay — golden glow on high-intensity edges
    phi_edges = optimizer.top_pheromone_edges(top_n=18)
    if phi_edges:
        max_phi = max(p for *_, p in phi_edges) or 1.0
        for u, v, phi in phi_edges:
            a  = min(phi / max_phi, 1.0) * 0.55
            ux, uy = city.pos_2d[u]
            vx, vy = city.pos_2d[v]
            fig.add_trace(go.Scatter(
                x=[ux, vx], y=[uy, vy],
                mode="lines",
                line=dict(color=f"rgba(255,215,0,{a:.2f})", width=5),
                showlegend=False,
                hoverinfo="none",
            ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=[city.pos_2d[n][0] for n in city.G.nodes()],
        y=[city.pos_2d[n][1] for n in city.G.nodes()],
        mode="markers",
        marker=dict(size=7, color="#dfe6e9", line=dict(width=1, color="#636e72")),
        name="Nodos",
        hovertemplate="Nodo %{text}<extra></extra>",
        text=[str(n) for n in city.G.nodes()],
    ))

    # Active orders — origin (blue star) / destination (orange square)
    for oid in sim.pending + sim.active:
        o  = sim.orders[oid]
        ox, oy = city.pos_2d[o.origin]
        dx, dy = city.pos_2d[o.destination]
        fig.add_trace(go.Scatter(
            x=[ox], y=[oy], mode="markers",
            marker=dict(size=14, color="#74b9ff", symbol="star"),
            showlegend=False,
            hovertemplate=f"Pedido #{oid}<br>Origen<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[dx], y=[dy], mode="markers",
            marker=dict(size=12, color="#fd7b2c", symbol="square"),
            showlegend=False,
            hovertemplate=f"Pedido #{oid}<br>Destino<extra></extra>",
        ))

    # Deliverers — dashed route + avatar circle
    for did, d in sim.deliverers.items():
        color = DELIVERER_COLORS[did % len(DELIVERER_COLORS)]

        if len(d.route) > 1:
            fig.add_trace(go.Scatter(
                x=[city.pos_2d[n][0] for n in d.route],
                y=[city.pos_2d[n][1] for n in d.route],
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                showlegend=False,
                hoverinfo="none",
            ))

        px_, py_ = city.pos_2d[d.position]
        fig.add_trace(go.Scatter(
            x=[px_], y=[py_],
            mode="markers+text",
            marker=dict(
                size=24, color=color, symbol="circle",
                line=dict(width=2, color="white"),
            ),
            text=[f"R{did}"],
            textposition="middle center",
            textfont=dict(size=9, color="white", family="Arial Black"),
            name=f"R{did} ({d.deliveries_done}✓)",
            hovertemplate=(
                f"Repartidor {did}<br>"
                f"Posición: {d.position}<br>"
                f"Entregas: {d.deliveries_done}<br>"
                f"Pedidos activos: {len(d.assigned_orders)}<extra></extra>"
            ),
        ))

    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(22,33,62,0.92)",
            font=dict(color="white", size=11),
            bordercolor="#34495e", borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.6, city.grid_size - 0.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.6, city.grid_size - 0.4]),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="white"),
        height=540,
        margin=dict(l=5, r=165, t=35, b=5),
        title=dict(text="Mapa Urbano en Tiempo Real", font=dict(size=13, color="#b2bec3")),
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Right-hand panels ─────────────────────────────────────────────────────────
with side_col:

    # Traffic distribution bar
    st.markdown("**Estado del tráfico**")
    counts = city.traffic_distribution()
    fig_tr = go.Figure(go.Bar(
        x=TRAFFIC_LABELS,
        y=counts,
        marker_color=TRAFFIC_COLORS,
        text=counts,
        textposition="auto",
        textfont=dict(size=10),
    ))
    fig_tr.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="white", size=9),
        height=190, margin=dict(l=5, r=5, t=5, b=35),
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-25),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_tr, use_container_width=True)

    # ACO convergence
    st.markdown("**Convergencia ACO**")
    if optimizer.convergence:
        conv_window = optimizer.convergence[-60:]
        fig_aco = go.Figure(go.Scatter(
            y=conv_window,
            mode="lines",
            line=dict(color="#f39c12", width=2),
            fill="tozeroy",
            fillcolor="rgba(243,156,18,0.15)",
        ))
        fig_aco.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="white", size=9),
            height=155, margin=dict(l=5, r=5, t=5, b=30),
            xaxis=dict(title="Iteración", showgrid=False),
            yaxis=dict(title="Costo", showgrid=False),
        )
        st.plotly_chart(fig_aco, use_container_width=True)
    else:
        st.caption("Esperando optimizaciones…")

    # Event log
    st.markdown("**Eventos recientes**")
    icons = {"info": "📌", "warning": "⚠️", "critical": "🚨"}
    for ev in reversed(sim.events[-10:]):
        ic = icons.get(ev.severity, "📌")
        color = {"info": "#b2bec3", "warning": "#fdcb6e", "critical": "#ff7675"}.get(ev.severity, "#b2bec3")
        st.markdown(
            f"<span style='color:#636e72;font-size:0.72rem'>{ev.time:.0f}u</span> "
            f"{ic} <span style='color:{color};font-size:0.75rem'>{ev.msg}</span>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Bottom charts
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
bc1, bc2, bc3 = st.columns(3)

# ── Poisson: theory vs observed ───────────────────────────────────────────────
with bc1:
    st.markdown(f"**Distribución de Poisson (λ={sim.lambda_orders})**")
    k_range = np.arange(0, 8)
    lam     = sim.lambda_orders
    theory  = np.array([
        (lam ** k * math.exp(-lam)) / math.factorial(k)
        for k in k_range
    ])
    n_steps  = max(len(sim.orders_per_step), 1)
    observed = np.array([sim.orders_per_step.count(int(k)) / n_steps for k in k_range])

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=k_range, y=theory,
        name="Teórico", marker_color="rgba(52,152,219,0.75)",
    ))
    fig_p.add_trace(go.Bar(
        x=k_range, y=observed,
        name="Observado", marker_color="rgba(231,76,60,0.75)",
    ))
    fig_p.update_layout(
        barmode="group",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="white", size=10),
        height=230, margin=dict(l=5, r=5, t=10, b=35),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="k pedidos/paso", showgrid=False, dtick=1),
        yaxis=dict(title="P(k)", showgrid=False),
    )
    st.plotly_chart(fig_p, use_container_width=True)

# ── Delivery time histogram ───────────────────────────────────────────────────
with bc2:
    st.markdown("**Histograma de Tiempos de Entrega**")
    if len(sim.delivery_times) >= 3:
        mu  = np.mean(sim.delivery_times)
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=sim.delivery_times,
            nbinsx=16,
            marker_color="#00b894",
            opacity=0.85,
            name="Tiempo entrega",
        ))
        fig_h.add_vline(
            x=mu, line_dash="dash", line_color="#e17055",
            annotation_text=f"μ={mu:.1f}u",
            annotation_font_color="#e17055",
        )
        fig_h.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="white", size=10),
            height=230, margin=dict(l=5, r=5, t=10, b=35),
            showlegend=False,
            xaxis=dict(title="Tiempo (u)", showgrid=False),
            yaxis=dict(title="Frecuencia",  showgrid=False),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Esperando datos de entrega…")

# ── Markov transition matrix heatmap ─────────────────────────────────────────
with bc3:
    st.markdown("**Cadena de Markov — Π (transición de tráfico)**")
    fig_mk = go.Figure(go.Heatmap(
        z=TRANSITION_MATRIX,
        x=TRAFFIC_LABELS,
        y=TRAFFIC_LABELS,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in TRANSITION_MATRIX],
        texttemplate="%{text}",
        textfont=dict(size=11, color="black"),
        showscale=False,
        hoverongaps=False,
    ))
    fig_mk.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="white", size=10),
        height=230, margin=dict(l=5, r=5, t=10, b=35),
        xaxis=dict(title="Estado siguiente", tickangle=-20),
        yaxis=dict(title="Estado actual"),
    )
    st.plotly_chart(fig_mk, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-run loop (must be LAST — triggers st.rerun)
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.running:
    sim.step(optimizer)
    st.session_state.step_count += 1
    time.sleep(speed)
    st.rerun()

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import sqlite3
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional

# Chrysalis imports
import sys

# Ensure project root is in sys.path
script_dir: Path = Path(__file__).parent.absolute()
project_root: Path = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chrysalis.simulations.phase_transitions import (  # noqa: E402
    ising_2d,
    potts_2d,
    percolation_2d,
)

st.set_page_config(page_title="Chrysalis Dashboard", layout="wide", page_icon="🦋")

st.title("🦋 Chrysalis: Phase Transition Dashboard")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Navigation")
mode: str = st.sidebar.radio(
    "Mode",
    ["Protocol Signature Viewer", "Knowledge Graph Explorer", "Live Simulation Lab"],
)


def load_results(exp_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not exp_dir.exists():
        return results
    for run_dir in exp_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "results.json").exists():
            try:
                with open(run_dir / "results.json", "r") as f:
                    results.append(json.load(f))
            except Exception:
                continue
    return results


def load_knowledge_graph() -> nx.Graph:
    db_path = project_root / "data" / "chrysalis.db"
    G = nx.Graph()
    if not db_path.exists():
        return G
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT subject, subject_type, predicate, object, object_type FROM knowledge_triples"
        )
        rows = cursor.fetchall()
        for row in rows:
            subj, subj_type, pred, obj, obj_type = row
            G.add_node(subj, type=subj_type)
            G.add_node(obj, type=obj_type)
            G.add_edge(subj, obj, predicate=pred)
        conn.close()
    except Exception as e:
        st.error(f"Error loading Knowledge Graph: {e}")
    return G


if mode == "Protocol Signature Viewer":
    st.sidebar.subheader("Experiment Selection")
    results_dir: Path = project_root / "simulations" / "results"

    experiments: List[str] = []
    if not results_dir.exists():
        st.warning("No simulation results found in `simulations/results/`.")
    else:
        experiments = sorted(
            [d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True
        )

    selected_exp: Optional[str] = st.sidebar.selectbox("Select Experiment", experiments)

    if selected_exp:
        exp_path: Path = results_dir / selected_exp
        all_runs: List[Dict[str, Any]] = load_results(exp_path)

        if all_runs:
            st.sidebar.success(f"Loaded {len(all_runs)} runs.")
            # Format options for selectbox
            run_options: List[str] = []
            for i, r in enumerate(all_runs):
                p: Dict[str, Any] = r["params"]
                desc: str = f"Run {i+1}: " + ", ".join(
                    [
                        f"{k}={v}"
                        for k, v in p.items()
                        if k not in ["output_dir", "seed"]
                    ]
                )
                run_options.append(desc)

            selected_run_idx: int = st.sidebar.selectbox(
                "Select Run Details",
                range(len(run_options)),
                format_func=lambda i: run_options[i],
            )
            run_data: Dict[str, Any] = all_runs[selected_run_idx]

            # --- Layout ---
            col1, col2 = st.columns([1, 1])

            main_res: Dict[str, Any] = run_data["main_results"]
            x_key: str = "T" if "T" in main_res else "p"
            x_vals: List[float] = main_res[x_key]
            x_label: str = "Temperature (T)" if x_key == "T" else "Probability (p)"

            with col1:
                st.subheader("📊 Physical Transition")
                # Order Parameter
                op_val: List[float] = main_res.get(
                    "magnetization", main_res.get("order_param")
                )
                fig_op = px.line(
                    x=x_vals,
                    y=op_val,
                    markers=True,
                    labels={"x": x_label, "y": "Order Parameter (m or P∞)"},
                    title="Order Parameter vs Control Variable",
                )
                st.plotly_chart(fig_op, use_container_width=True)

                # Susceptibility
                sus_val: List[float] = main_res.get("susceptibility")
                fig_sus = px.line(
                    x=x_vals,
                    y=sus_val,
                    markers=True,
                    labels={"x": x_label, "y": "Susceptibility (χ)"},
                    title="Susceptibility / Response Sensitivity",
                )
                st.plotly_chart(fig_sus, use_container_width=True)

            with col2:
                st.subheader("📜 Protocol Signature")
                if "protocol_metrics" in run_data:
                    prot_metrics: Dict[str, List[float]] = run_data["protocol_metrics"]
                    fig_prot = go.Figure()
                    for step, values in prot_metrics.items():
                        fig_prot.add_trace(
                            go.Scatter(
                                x=x_vals, y=values, name=step, mode="lines+markers"
                            )
                        )
                    fig_prot.update_layout(
                        title="Eight-Step Protocol Dynamics",
                        xaxis_title=x_label,
                        yaxis_title="Normalized Intensity",
                        legend_title="Protocol Phase",
                    )
                    st.plotly_chart(fig_prot, use_container_width=True)
                else:
                    st.info("No Protocol Metrics mapping found for this run.")

            # Lattice Snapshots
            if "snapshots" in main_res and main_res["snapshots"]:
                st.markdown("---")
                st.subheader("🖼️ State Snapshots")
                snaps: Dict[str, List[List[int]]] = main_res["snapshots"]
                snap_cols = st.columns(len(snaps))
                for i, (t_val, lattice) in enumerate(snaps.items()):
                    with snap_cols[i]:
                        st.caption(f"{x_key} = {t_val}")
                        # Color logic based on model
                        lat_arr: np.ndarray = np.array(lattice)
                        fig_snap = px.imshow(
                            lat_arr,
                            color_continuous_scale=(
                                "RdBu_r" if lat_arr.min() < 0 else "Viridis"
                            ),
                        )
                        fig_snap.update_layout(
                            coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0)
                        )
                        st.plotly_chart(fig_snap, use_container_width=True)

elif mode == "Knowledge Graph Explorer":
    st.subheader("🕸️ Chrysalis Knowledge Graph Explorer")
    st.markdown(
        "Interactive exploration of the cross-modal knowledge extracted from simulations and ethnography."
    )

    G = load_knowledge_graph()

    if G.number_of_nodes() == 0:
        st.info("Knowledge Graph is empty. Run `chrysalis-cli extract-graph` first.")
    else:
        # Sidebar filters
        st.sidebar.subheader("Graph Filters")
        node_types = sorted(list(set(nx.get_node_attributes(G, "type").values())))
        selected_types = st.sidebar.multiselect(
            "Filter by Node Type", node_types, default=node_types
        )
        view_dim = st.sidebar.radio("View Dimension", ["2D", "3D"], index=0)

        # Filter graph
        nodes_to_keep = [
            n for n, attr in G.nodes(data=True) if attr.get("type") in selected_types
        ]
        subgraph = G.subgraph(nodes_to_keep)

        # Layout
        with st.spinner(f"Computing {view_dim} Graph Layout..."):
            dim = 2 if view_dim == "2D" else 3
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50, dim=dim)

        if view_dim == "2D":
            # Create Plotly Edge trace
            edge_x = []
            edge_y = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            # Create Plotly Node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []

            type_color_map = {
                "ProtocolStep": "royalblue",
                "SimulationModel": "firebrick",
                "Concept": "forestgreen",
                "EthnographicEvent": "orange",
                "Author": "mediumpurple",
            }

            for node in subgraph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                attr = subgraph.nodes[node]
                ntype = attr.get("type", "Unknown")
                node_text.append(f"Node: {node}<br>Type: {ntype}")
                node_color.append(type_color_map.get(ntype, "grey"))

                # Size nodes by degree
                node_size.append(10 + 5 * subgraph.degree(node))

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line_width=2,
                ),
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Semantic Web of Transformation (2D)",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )
        else:
            # 3D View
            edge_x = []
            edge_y = []
            edge_z = []
            for edge in subgraph.edges():
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            node_x = []
            node_y = []
            node_z = []
            node_text = []
            node_color = []
            node_size = []

            type_color_map = {
                "ProtocolStep": "royalblue",
                "SimulationModel": "firebrick",
                "Concept": "forestgreen",
                "EthnographicEvent": "orange",
                "Author": "mediumpurple",
            }

            for node in subgraph.nodes():
                x, y, z = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                attr = subgraph.nodes[node]
                ntype = attr.get("type", "Unknown")
                node_text.append(f"Node: {node}<br>Type: {ntype}")
                node_color.append(type_color_map.get(ntype, "grey"))
                node_size.append(5 + 2 * subgraph.degree(node))

            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    opacity=0.8,
                ),
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Semantic Web of Transformation (3D)",
                    titlefont_size=16,
                    showlegend=False,
                    scene=dict(
                        xaxis=dict(showticklabels=False, title=""),
                        yaxis=dict(showticklabels=False, title=""),
                        zaxis=dict(showticklabels=False, title=""),
                    ),
                    margin=dict(b=20, l=5, r=5, t=40),
                ),
            )

        st.plotly_chart(fig, use_container_width=True)

        # Triples Table
        st.markdown("---")
        st.subheader("📋 Underlying Triples")
        triples_data = []
        for u, v, attr in subgraph.edges(data=True):
            triples_data.append(
                {
                    "Subject": u,
                    "Subject Type": subgraph.nodes[u].get("type"),
                    "Predicate": attr.get("predicate"),
                    "Object": v,
                    "Object Type": subgraph.nodes[v].get("type"),
                }
            )
        st.table(triples_data[:50])  # Display first 50

else:
    # --- Live Simulation Lab ---
    st.sidebar.subheader("Model Configuration")
    model_type: str = st.sidebar.selectbox(
        "Model Selection",
        ["Ising (Binary)", "Potts (Multi-State)", "Percolation (Connectivity)"],
    )
    N: int = st.sidebar.slider("Lattice Size (N x N)", 10, 100, 40)

    T: float = 0.0
    p: float = 0.0

    if model_type == "Ising (Binary)":
        T = st.sidebar.slider("Temperature (T)", 0.1, 5.0, 2.269, step=0.1)
        st.sidebar.info("Critical Point Tc ≈ 2.269")
    elif model_type == "Potts (Multi-State)":
        T = st.sidebar.slider("Temperature (T)", 0.1, 2.5, 0.995, step=0.05)
        st.sidebar.info("Critical Point Tc ≈ 0.995 (q=3)")
    else:
        p = st.sidebar.slider("Occupation Probability (p)", 0.0, 1.0, 0.5927, step=0.01)
        st.sidebar.info("Critical Point pc ≈ 0.5927")

    run_btn: bool = st.sidebar.button("🚀 Execute Live Run")

    if run_btn:
        st.subheader(f"Live Simulation: {model_type}")

        # Placeholders
        col_vis, col_chart, col_proto = st.columns([1, 1, 1])
        with col_vis:
            lattice_spot = st.empty()
        with col_chart:
            metrics_spot = st.empty()
        with col_proto:
            proto_spot = st.empty()

        progress_text = st.empty()
        progress_bar = st.progress(0)

        history: List[float] = []
        rng = np.random.default_rng()

        # Initialization
        grid: np.ndarray
        beta: float
        critical_point: float

        if "Ising" in model_type:
            grid = ising_2d.init_lattice(N, rng)
            beta = 1.0 / T
            critical_point = 2.269
            current_control = T
        elif "Potts" in model_type:
            grid = potts_2d.init_lattice(N, rng)
            beta = 1.0 / T
            critical_point = 0.995
            current_control = T
        else:
            # Percolation is static per step or we can animate 'p' sweep
            grid = percolation_2d.generate_lattice(N, 0, rng)
            critical_point = 0.5927
            current_control = 0.0

        def _get_protocol_relevance(val: float, crit: float) -> Dict[str, float]:
            """
            Estimate protocol phase relevance based on distance from criticality.
            Simplified mapping for live dashboard.
            """
            dist = val - crit
            sigma = 0.1 * crit  # Width of the critical window

            # Liminality peaks at critical point
            liminality = np.exp(-(dist**2) / (2 * sigma**2))

            # Dissolution/Purification (Ordered phase)
            # For thermal (Ising/Potts): T < Tc is ordered
            # For Percolation: p > pc is ordered (connected)
            is_thermal = crit > 1.0 or crit == 0.995
            if is_thermal:
                ordered = 1.0 / (
                    1.0 + np.exp(dist / sigma)
                )  # Sigmoid decreasing with T
                disordered = 1.0 / (
                    1.0 + np.exp(-dist / sigma)
                )  # Sigmoid increasing with T
            else:
                # Percolation: p < pc is disordered (dust), p > pc is ordered (spanning)
                disordered = 1.0 / (1.0 + np.exp(dist / sigma))
                ordered = 1.0 / (1.0 + np.exp(-dist / sigma))

            return {
                "Anchoring": ordered * 0.8,
                "Dissolution": ordered * 0.6 + liminality * 0.2,
                "Liminality": liminality,
                "Emergence": disordered * 0.6 + liminality * 0.4,
                "Integration": disordered * 0.8,
            }

        total_steps: int = 100
        for s in range(total_steps):
            m: float
            if "Ising" in model_type:
                ising_2d.metropolis_sweep(grid, beta, rng)
                m = np.abs(np.sum(grid)) / (N * N)
            elif "Potts" in model_type:
                potts_2d.metropolis_sweep(grid, beta, rng)
                m = potts_2d.order_parameter(grid)
            else:
                # For percolation live, we sweep 'p' from 0 to 1
                curr_p: float = s / total_steps
                current_control = curr_p
                grid = percolation_2d.generate_lattice(N, curr_p, rng)
                labeled, sizes = percolation_2d.find_clusters_uf(grid)
                m = percolation_2d.largest_cluster_fraction(labeled, N)

            history.append(m)

            if s % 2 == 0 or s == total_steps - 1:
                # Update Visualization
                lattice_spot.plotly_chart(
                    px.imshow(
                        grid,
                        color_continuous_scale=(
                            "RdBu_r" if grid.min() < 0 else "Viridis"
                        ),
                        title=f"Lattice State (Step {s})",
                    ),
                    use_container_width=True,
                )

                # Update Chart
                fig_h = px.line(
                    y=history,
                    labels={"x": "Step", "y": "Order Parameter"},
                    title="Evolution of Order",
                )
                metrics_spot.plotly_chart(fig_h, use_container_width=True)

                # Update Protocol Signature
                relevance = _get_protocol_relevance(current_control, critical_point)
                fig_p = px.bar(
                    x=list(relevance.keys()),
                    y=list(relevance.values()),
                    labels={"x": "Phase", "y": "Relevance"},
                    title="Live Protocol Signature",
                    range_y=[0, 1],
                )
                proto_spot.plotly_chart(fig_p, use_container_width=True)

                progress_bar.progress((s + 1) / total_steps)
                progress_text.text(f"Step {s}/{total_steps} | Current Order: {m:.4f}")

            time.sleep(0.01)

        st.balloons()
        st.success("Analysis Complete.")

st.sidebar.markdown("---")
st.sidebar.caption("Chrysalis Framework | Semantic Phase Transitions")

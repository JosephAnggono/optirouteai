import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make src importable when running Streamlit from project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import (
    compute_gap_percent,
    solve_with_nearest_neighbor,
)
from optirouteai.optimization.hybrid_solver import HybridNeuralSolver
from optirouteai.optimization.neural_solver import MLPNeuralSolver
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools


st.set_page_config(
    page_title="OptiRouteAI Dashboard",
    page_icon="🚚",
    layout="wide",
)


def plot_routes_streamlit(instance, routes, title):
    """
    Create route visualization as a matplotlib figure for Streamlit.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        instance.customers[:, 0],
        instance.customers[:, 1],
        c="tab:blue",
        s=45,
        label="Customers",
        alpha=0.8,
    )

    ax.scatter(
        instance.depot[0],
        instance.depot[1],
        c="red",
        s=180,
        marker="*",
        label="Depot",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    for idx, customer in enumerate(instance.customers):
        ax.text(
            customer[0] + 0.008,
            customer[1] + 0.008,
            str(idx),
            fontsize=7,
        )

    colors = [
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    for vehicle_idx, route in enumerate(routes):
        if not route:
            continue

        color = colors[vehicle_idx % len(colors)]

        x_coords = [instance.depot[0]]
        y_coords = [instance.depot[1]]

        for customer_idx in route:
            customer = instance.customers[customer_idx]
            x_coords.append(customer[0])
            y_coords.append(customer[1])

        x_coords.append(instance.depot[0])
        y_coords.append(instance.depot[1])

        route_load = sum(instance.demands[idx] for idx in route)

        ax.plot(
            x_coords,
            y_coords,
            marker="o",
            linewidth=2,
            color=color,
            alpha=0.85,
            label=f"Vehicle {vehicle_idx + 1} | Load {route_load}/{instance.vehicle_capacity}",
        )

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig


@st.cache_resource
def load_neural_solver():
    return MLPNeuralSolver(
        checkpoint_path="models/mlp_policy_step9.pkl",
    )


@st.cache_resource
def load_hybrid_solver(alpha):
    return HybridNeuralSolver(
        checkpoint_path="models/mlp_policy_step9.pkl",
        alpha=alpha,
    )


def run_selected_solver(instance, solver_name, alpha, ortools_time_limit):
    """
    Run one selected solver.
    """
    if solver_name == "Nearest Neighbor":
        return solve_with_nearest_neighbor(instance)

    if solver_name == "OR-Tools":
        return solve_cvrp_with_ortools(
            instance=instance,
            time_limit_seconds=ortools_time_limit,
        )

    if solver_name == "Neural MLP":
        solver = load_neural_solver()
        return solver.solve(instance)

    if solver_name == "Hybrid MLP":
        solver = load_hybrid_solver(alpha)
        return solver.solve(instance)

    raise ValueError(f"Unsupported solver: {solver_name}")


def run_all_solvers(instance, alpha, ortools_time_limit):
    """
    Run all available solvers.
    """
    results = []

    nn_result = solve_with_nearest_neighbor(instance)
    ortools_result = solve_cvrp_with_ortools(
        instance=instance,
        time_limit_seconds=ortools_time_limit,
    )

    results.append(nn_result)
    results.append(ortools_result)

    try:
        neural_result = load_neural_solver().solve(instance)
        hybrid_result = load_hybrid_solver(alpha).solve(instance)

        results.append(neural_result)
        results.append(hybrid_result)

    except FileNotFoundError:
        st.warning(
            "Neural checkpoint not found. Run Step 9 first to enable Neural MLP and Hybrid MLP."
        )

    return results


def build_results_table(results):
    """
    Convert solver results into a comparison table.
    """
    reference_distance = None

    for result in results:
        if (
            result.solver_name == "OR-Tools"
            and result.is_feasible
            and math.isfinite(result.total_distance)
            and result.total_distance > 0
        ):
            reference_distance = result.total_distance
            break

    rows = []

    for result in results:
        if reference_distance is None:
            gap = None
        else:
            gap = compute_gap_percent(
                solver_distance=result.total_distance,
                reference_distance=reference_distance,
            )

        rows.append(
            {
                "Solver": result.solver_name,
                "Total Distance": result.total_distance,
                "Runtime (s)": result.runtime_seconds,
                "Feasible": result.is_feasible,
                "Gap vs OR-Tools (%)": gap,
                "Routes": result.routes,
            }
        )

    return pd.DataFrame(rows)


def main():
    st.title("🚚 OptiRouteAI: Neural Combinatorial Optimization Dashboard")

    st.markdown(
        """
        This dashboard demonstrates a CVRP solver system comparing:
        **Nearest Neighbor**, **OR-Tools**, **Neural MLP**, and **Hybrid MLP** solvers.
        """
    )

    st.sidebar.header("Problem Settings")

    num_customers = st.sidebar.slider(
        "Number of Customers",
        min_value=5,
        max_value=80,
        value=25,
        step=5,
    )

    num_vehicles = st.sidebar.slider(
        "Number of Vehicles",
        min_value=1,
        max_value=15,
        value=5,
    )

    vehicle_capacity = st.sidebar.slider(
        "Vehicle Capacity",
        min_value=5,
        max_value=100,
        value=45,
        step=5,
    )

    demand_min = st.sidebar.number_input(
        "Minimum Demand",
        min_value=1,
        max_value=20,
        value=1,
    )

    demand_max = st.sidebar.number_input(
        "Maximum Demand",
        min_value=1,
        max_value=30,
        value=8,
    )

    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
    )

    st.sidebar.header("Solver Settings")

    solver_mode = st.sidebar.radio(
        "Run Mode",
        ["Compare All Solvers", "Single Solver"],
    )

    selected_solver = st.sidebar.selectbox(
        "Solver",
        ["Nearest Neighbor", "OR-Tools", "Neural MLP", "Hybrid MLP"],
    )

    alpha = st.sidebar.slider(
        "Hybrid Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = distance only, 1 = pure neural MLP",
    )

    ortools_time_limit = st.sidebar.slider(
        "OR-Tools Time Limit (seconds)",
        min_value=1,
        max_value=10,
        value=1,
    )

    run_button = st.sidebar.button("Run Solver")

    if demand_max < demand_min:
        st.error("Maximum demand must be greater than or equal to minimum demand.")
        return

    instance = generate_cvrp_instance(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        demand_min=demand_min,
        demand_max=demand_max,
        seed=int(seed),
    )

    total_demand = int(instance.demands.sum())
    total_capacity = int(instance.num_vehicles * instance.vehicle_capacity)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", num_customers)
    col2.metric("Vehicles", num_vehicles)
    col3.metric("Total Demand", total_demand)
    col4.metric("Total Capacity", total_capacity)

    if total_demand > total_capacity:
        st.warning(
            "Total demand exceeds total vehicle capacity. Some solvers may not find a feasible solution."
        )

    st.subheader("Generated CVRP Instance")

    base_fig = plot_routes_streamlit(
        instance=instance,
        routes=[],
        title="Generated Customer Locations",
    )
    st.pyplot(base_fig)

    if run_button:
        st.subheader("Solver Results")

        if solver_mode == "Compare All Solvers":
            with st.spinner("Running all solvers..."):
                results = run_all_solvers(
                    instance=instance,
                    alpha=alpha,
                    ortools_time_limit=ortools_time_limit,
                )
        else:
            with st.spinner(f"Running {selected_solver}..."):
                results = [
                    run_selected_solver(
                        instance=instance,
                        solver_name=selected_solver,
                        alpha=alpha,
                        ortools_time_limit=ortools_time_limit,
                    )
                ]

        results_table = build_results_table(results)

        display_table = results_table.drop(columns=["Routes"]).copy()
        display_table["Total Distance"] = display_table["Total Distance"].round(4)
        display_table["Runtime (s)"] = display_table["Runtime (s)"].round(4)
        display_table["Gap vs OR-Tools (%)"] = display_table[
            "Gap vs OR-Tools (%)"
        ].round(4)

        st.dataframe(display_table, use_container_width=True)

        st.subheader("Route Visualizations")

        for result in results:
            st.markdown(f"### {result.solver_name}")

            fig = plot_routes_streamlit(
                instance=instance,
                routes=result.routes,
                title=(
                    f"{result.solver_name} | "
                    f"Distance={result.total_distance:.4f} | "
                    f"Runtime={result.runtime_seconds:.4f}s | "
                    f"Feasible={result.is_feasible}"
                ),
            )

            st.pyplot(fig)


if __name__ == "__main__":
    main()
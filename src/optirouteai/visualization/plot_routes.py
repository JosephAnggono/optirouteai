from pathlib import Path

import matplotlib.pyplot as plt

from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.constraints import solution_loads
from optirouteai.optimization.distance import solution_distance


def plot_routes(
    instance: CVRPInstance,
    routes: list[list[int]],
    title: str = "CVRP Route Visualization",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot CVRP routes for multiple vehicles.

    Args:
        instance: CVRP instance.
        routes: List of vehicle routes.
        title: Plot title.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    plt.figure(figsize=(9, 7))

    # Plot customers
    plt.scatter(
        instance.customers[:, 0],
        instance.customers[:, 1],
        c="tab:blue",
        s=45,
        label="Customers",
        alpha=0.8,
    )

    # Plot depot
    plt.scatter(
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

    # Annotate customers
    for idx, customer in enumerate(instance.customers):
        plt.text(
            customer[0] + 0.008,
            customer[1] + 0.008,
            str(idx),
            fontsize=8,
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

    loads = solution_loads(routes, instance)
    total_dist = solution_distance(routes, instance)

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

        label = (
            f"Vehicle {vehicle_idx + 1} "
            f"(load={loads[vehicle_idx]}/{instance.vehicle_capacity})"
        )

        plt.plot(
            x_coords,
            y_coords,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
            alpha=0.85,
        )

    plt.title(f"{title}\nTotal Distance: {total_dist:.4f}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()
import matplotlib.pyplot as plt

from src.utils import Point


def plot_solution(
    points: list[Point],
    cycle: list[int],
    title: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Draw the cycle. Node cost is shown by both colour and size.
    Nodes not in the solution are small light-grey dots."""
    all_costs = [p.cost for p in points]
    c_min, c_max = min(all_costs), max(all_costs)
    span = (c_max - c_min) or 1

    selected = set(cycle)
    fig, ax = plt.subplots(figsize=(8, 7))

    # unselected nodes
    rest = [p for i, p in enumerate(points) if i not in selected]
    ax.scatter(
        [p.x for p in rest],
        [p.y for p in rest],
        s=12,
        color="lightgrey",
        zorder=1,
        label="not selected",
    )

    # closed cycle (last node connects back to the first)
    loop = cycle + cycle[:1]
    ax.plot(
        [points[i].x for i in loop],
        [points[i].y for i in loop],
        color="black",
        linewidth=0.8,
        zorder=2,
    )

    # selected nodes: colour and size by cost
    chosen = [points[i] for i in cycle]
    sizes = [25 + 100 * (p.cost - c_min) / span for p in chosen]
    sc = ax.scatter(
        [p.x for p in chosen],
        [p.y for p in chosen],
        c=[p.cost for p in chosen],
        s=sizes,
        cmap="viridis_r",
        vmin=c_min,
        vmax=c_max,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label="node cost")

    # mark the start node
    start = points[cycle[0]]
    ax.scatter(
        [start.x],
        [start.y],
        marker="*",
        s=250,
        color="red",
        edgecolors="black",
        zorder=4,
        label="start node",
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

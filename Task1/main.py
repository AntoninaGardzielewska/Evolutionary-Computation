from pathlib import Path

from src.greedy_cycle import GreedyCycle
from src.greedy_regret import Greedy2RegretAnyPosition, Greedy2RegretCycle
from src.nearest_neighbor_any import NearestNeighborAnyPosition
from src.nearest_neighbor_end import NearestNeighborEndPosition
from src.random_solution import Random
from src.utils import build_distance_matrix, read_file
from src.visualization import plot_solution

WEIGHTS = (1.0, 1.0)

solvers = (
    ("Random", Random, {}),
    ("NearestNeighborEndPosition", NearestNeighborEndPosition, {}),
    ("NearestNeighborAnyPosition", NearestNeighborAnyPosition, {}),
    ("GreedyCycle", GreedyCycle, {}),
    ("Greedy2RegretAnyPosition", Greedy2RegretAnyPosition, {}),
    ("Greedy2RegretCycle", Greedy2RegretCycle, {}),
    (
        "Greedy2RegretAnyPositionWeighted",
        Greedy2RegretAnyPosition,
        {"weights": WEIGHTS},
    ),
    (
        "Greedy2RegretCycleWeighted",
        Greedy2RegretCycle,
        {"weights": WEIGHTS},
    ),
)

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    points = read_file(BASE_DIR / "data" / "TSPA.csv")
    distances = build_distance_matrix(points)
    costs = [point.cost for point in points]
    for method, solver_class, kwargs in solvers:
        solver = solver_class(distances, costs, **kwargs)
        result = solver.get_statistics()
        print(f"{method}: min={result.min} max={result.max} avg={result.average:.1f}")

        plot_solution(
            points,
            result.best_cycle,
            title=f"{method} (cost {result.min})",
            save_path=str(out / f"{method}.png"),
            show=False,
        )

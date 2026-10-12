from pathlib import Path

from src.greedy_cycle import GreedyCycle
from src.nearest_neighbor_any import NearestNeighborAnyPosition
from src.nearest_neighbor_end import NearestNeighborEndPosition
from src.random_solution import Random
from src.utils import build_distance_matrix, read_file
from src.visualization import plot_solution

solvers = (Random, NearestNeighborEndPosition, NearestNeighborAnyPosition, GreedyCycle)
BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    points = read_file(BASE_DIR / "data" / "TSPA.csv")
    distances = build_distance_matrix(points)
    costs = [point.cost for point in points]
    for solver_class in solvers:
        method = solver_class.__name__
        solver = solver_class(distances, costs)
        result = solver.get_statistics()
        print(f"{method}: min={result.min} max={result.max} avg={result.average:.1f}")

        plot_solution(
            points,
            result.best_cycle,
            title=f"{method} (cost {result.min})",
            save_path= str(out / f"{method}.png"),
            show=False,
        )

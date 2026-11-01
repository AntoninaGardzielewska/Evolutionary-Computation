import logging
from pathlib import Path

from src.greedy_cycle import GreedyCycle
from src.greedy_regret import Greedy2RegretAnyPosition, Greedy2RegretCycle
from src.local_search import LocalSearch
from src.nearest_neighbor_any import NearestNeighborAnyPosition
from src.nearest_neighbor_end import NearestNeighborEndPosition
from src.random_solution import Random
from src.utils import build_distance_matrix, read_file, build_report
from src.visualization import plot_solution

logger = logging.getLogger(__name__)

WEIGHTS = (1.0, 1.0)
RUNS = 200
BASE_DIR = Path(__file__).resolve().parent

AVAILABLE_CONSTRUCTION_SOLVERS = {
    "Random": (Random, {}),
    "NearestNeighborEndPosition": (NearestNeighborEndPosition, {}),
    "NearestNeighborAnyPosition": (NearestNeighborAnyPosition, {}),
    "GreedyCycle": (GreedyCycle, {}),
    "Greedy2RegretAnyPosition": (Greedy2RegretAnyPosition, {}),
    "Greedy2RegretCycle": (Greedy2RegretCycle, {}),
    "Greedy2RegretAnyPositionWeighted": (
        Greedy2RegretAnyPosition,
        {"weights": WEIGHTS},
    ),
    "Greedy2RegretCycleWeighted": (Greedy2RegretCycle, {"weights": WEIGHTS}),
}

# Uncomment construction method names here when you want to include them.
CONSTRUCTION_METHODS = [
    # "Random",
    # "NearestNeighborEndPosition",
    # "NearestNeighborAnyPosition",
    # "GreedyCycle",
    # "Greedy2RegretAnyPosition",
    # "Greedy2RegretCycle",
    # "Greedy2RegretAnyPositionWeighted",
    # "Greedy2RegretCycleWeighted",
]

# Each entry is (report name, search type, intra-route move, start, candidate list size).
# None means the full neighborhood; an integer enables candidate moves.
LOCAL_SEARCH_METHODS = [
    # ("LocalSearch-steepest-edges-random", "steepest", "edges", "random", None),
    (
        "LocalSearch-steepest-edges-random-candidates-10",
        "steepest",
        "edges",
        "random",
        10,
    ),
    # ("LocalSearch-steepest-edges-random-candidates-5", "steepest", "edges", "random", 5),
    # ("LocalSearch-steepest-edges-random-candidates-15", "steepest", "edges", "random", 15),
    # ("LocalSearch-steepest-nodes-random-candidates-10", "steepest", "nodes", "random", 10),
    # ("LocalSearch-greedy-edges-random-candidates-10", "greedy", "edges", "random", 10),
    # ("LocalSearch-steepest-edges-greedy-candidates-10", "steepest", "edges", "greedy", 10),
]


def run_solver(solver, runs: int = RUNS, progress_label: str = "") -> dict:
    def log_progress(completed: int, total: int) -> None:
        logger.info("%s: completed %d/%d runs", progress_label, completed, total)

    result = solver.get_statistics(runs, progress_callback=log_progress)
    return {
        "min": result.min,
        "max": result.max,
        "average": result.average,
        "best_cycle": result.best_cycle,
        "min_time": result.min_time,
        "max_time": result.max_time,
        "average_time": result.average_time,
    }


def run_experiment() -> str:
    results_by_instance = {}
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance_name in ("TSPA", "TSPB"):
        logger.info("Starting experiment for %s", instance_name)
        points = read_file(BASE_DIR / "data" / f"{instance_name}.csv")
        distances = build_distance_matrix(points)
        costs = [point.cost for point in points]
        instance_results = {}
        construction_results = {}

        for method in CONSTRUCTION_METHODS:
            solver_class, kwargs = AVAILABLE_CONSTRUCTION_SOLVERS[method]
            solver = solver_class(distances, costs, **kwargs)
            result = run_solver(solver, progress_label=f"{instance_name} / {method}")
            instance_results[method] = result
            construction_results[method] = (solver_class, kwargs, result)
            logger.info(
                "%s / %s: average objective %.1f, average runtime %.4f s",
                instance_name,
                method,
                result["average"],
                result["average_time"],
            )

        available_greedy_starts = {
            method: configuration
            for method, configuration in construction_results.items()
            if method != "Random"
        }
        if available_greedy_starts:
            greedy_method = min(
                available_greedy_starts,
                key=lambda name: available_greedy_starts[name][2]["average"],
            )
            greedy_class, greedy_kwargs, _ = available_greedy_starts[greedy_method]
            default_greedy_solver = greedy_class(
                distances, costs, **greedy_kwargs
            )
        else:
            default_greedy_solver = GreedyCycle(distances, costs)

        for method, search_type, move_type, starting_solution, candidate_list_size in (
            LOCAL_SEARCH_METHODS
        ):
            solver = LocalSearch(
                distances,
                costs,
                search_type=search_type,
                intra_route_move=move_type,
                starting_solution=starting_solution,
                greedy_solver=(
                    default_greedy_solver if starting_solution == "greedy" else None
                ),
                candidate_list_size=candidate_list_size,
            )
            result = run_solver(solver, progress_label=f"{instance_name} / {method}")
            instance_results[method] = result
            logger.info(
                "%s / %s: average objective %.1f, average runtime %.4f s",
                instance_name,
                method,
                result["average"],
                result["average_time"],
            )

        results_by_instance[instance_name] = instance_results

        for method, result in instance_results.items():
            filename = (
                f"{method}.png" if instance_name == "TSPA" else f"TSPB_{method}.png"
            )
            plot_solution(
                points,
                result["best_cycle"],
                title=f"{instance_name} {method} (cost {result['min']})",
                save_path=str(output_dir / filename),
                show=False,
            )

    report = build_report(results_by_instance)
    report_path = output_dir / "experiment_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    logger.info("Report saved to %s", report_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_experiment()

import logging
from pathlib import Path

from src.greedy_cycle import GreedyCycle
from src.greedy_regret import Greedy2RegretAnyPosition, Greedy2RegretCycle
from src.local_search import LocalSearch
from src.nearest_neighbor_any import NearestNeighborAnyPosition
from src.nearest_neighbor_end import NearestNeighborEndPosition
from src.random_solution import Random
from src.utils import build_distance_matrix, read_file
from src.visualization import plot_solution

logger = logging.getLogger(__name__)

WEIGHTS = (1.0, 1.0)
RUNS = 200
BASE_DIR = Path(__file__).resolve().parent

CONSTRUCTION_SOLVERS = (
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


def format_summary(result: dict, timing: bool = False) -> str:
    if timing:
        return (
            f"{result['average_time']:.4f} s "
            f"({result['min_time']:.4f} - {result['max_time']:.4f})"
        )
    return f"{result['average']:.1f} ({result['min']} - {result['max']})"


def _table(
    results_by_instance: dict[str, dict[str, dict]],
    key: str,
    title: str,
    fmt: str,
) -> list[str]:
    method_names = list(next(iter(results_by_instance.values())))
    instances = list(results_by_instance)
    lines = [
        title,
        "",
        "| Method | " + " | ".join(instances) + " |",
        "| --- | " + " | ".join("---" for _ in instances) + " |",
    ]
    for method in method_names:
        cells = (
            format(results_by_instance[instance][method][key], fmt)
            for instance in instances
        )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def build_report(results_by_instance: dict[str, dict[str, dict]]) -> str:
    lines: list[str] = []
    lines += _table(results_by_instance, "min", "Objective function - minimum:", ".0f")
    lines += _table(results_by_instance, "max", "Objective function - maximum:", ".0f")
    lines += _table(
        results_by_instance, "average", "Objective function - average:", ".1f"
    )
    lines += _table(
        results_by_instance, "min_time", "Running time (s) - minimum:", ".4f"
    )
    lines += _table(
        results_by_instance, "max_time", "Running time (s) - maximum:", ".4f"
    )
    lines += _table(
        results_by_instance, "average_time", "Running time (s) - average:", ".4f"
    )
    return "\n".join(lines)


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

        for method, solver_class, kwargs in CONSTRUCTION_SOLVERS:
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

        greedy_method = min(
            (name for name, _, _ in CONSTRUCTION_SOLVERS if name != "Random"),
            key=lambda name: construction_results[name][2]["average"],
        )
        greedy_class, greedy_kwargs, _ = construction_results[greedy_method]
        greedy_solver = greedy_class(distances, costs, **greedy_kwargs)
        logger.info(
            "%s: using %s for greedy local-search starting solutions",
            instance_name,
            greedy_method,
        )

        for search_type in ("steepest", "greedy"):
            for move_type in ("nodes", "edges"):
                for starting_solution in ("random", "greedy"):
                    method = (
                        f"LocalSearch-{search_type}-{move_type}-{starting_solution}"
                    )
                    solver = LocalSearch(
                        distances,
                        costs,
                        search_type=search_type,
                        intra_route_move=move_type,
                        starting_solution=starting_solution,
                        greedy_solver=(
                            greedy_solver if starting_solution == "greedy" else None
                        ),
                    )
                    result = run_solver(
                        solver, progress_label=f"{instance_name} / {method}"
                    )
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

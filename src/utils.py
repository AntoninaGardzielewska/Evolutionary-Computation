from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Result:
    min: int
    max: int
    average: float
    best_cycle: list[int]
    min_time: float = 0.0
    max_time: float = 0.0
    average_time: float = 0.0


@dataclass
class Point:
    x: int
    y: int
    cost: int


def read_file(path: str | Path) -> list[Point]:
    df = pd.read_csv(path, header=None, sep=";")
    points_list = [Point(*row[1:]) for row in df.itertuples()]
    return points_list


def build_distance_matrix(points: list[Point]) -> list[list[int]]:
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = (points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2
            dist = dist**0.5
            dist = int(dist + 0.5)
            matrix[i][j] = matrix[j][i] = dist
    return matrix


def cycle_cost(distances: list, costs: list, cycle: list[int]):
    cost = sum(costs[point] for point in cycle)
    distance = 0
    for point_a, point_b in zip(cycle, cycle[1:] + cycle[:1]):
        distance += distances[point_a][point_b]

    return cost + distance


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
    lines.append("Best cycles")
    lines.append("")
    for instance, methods in results_by_instance.items():
        lines.append(f"**{instance}**")
        lines.append("")
        for method, result in methods.items():
            lines.append(f"- {method}: {result['best_cycle']}")
        lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    print(read_file("../data/TSPA.csv"))

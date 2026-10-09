from dataclasses import dataclass
import pandas as pd


@dataclass
class Result:
    min: int
    max: int
    average: float
    best_cycle: list[int]


@dataclass
class Point:
    x: int
    y: int
    cost: int


def read_file(path: str) -> list[Point]:
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


if __name__ == "__main__":
    print(read_file("../data/TSPA.csv"))

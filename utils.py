from dataclasses import dataclass
import pandas as pd

@dataclass
class Result:
    min: int
    max: int
    average: float

@dataclass
class Point:
    x: int
    y: int
    cost: int

def read_file(path: str) -> list[Point]:
    df = pd.read_csv(path, header=None, sep=";")
    points_list = [Point(*row[1:]) for row in df.itertuples()]
    return points_list

def calculate_distance(point1: Point, point2: Point) -> float:
    euclidean = ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5
    return int(round(euclidean))

if __name__ == "__main__":
    print(read_file("TSPA.csv"))

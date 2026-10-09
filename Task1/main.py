from src.random_solution import RandomSolver
from src.utils import read_file, build_distance_matrix, Point

def run_random_solution(data: list[Point]) -> None:
    distances = build_distance_matrix(data)
    costs = [point.cost for point in data]
    solver = RandomSolver(distances, costs)
    print(solver.get_statistics())

if __name__ == '__main__':
    data = read_file("data/TSPA.csv")
    run_random_solution(data)


from src.nearest_neighbor_end import NearestNeighborEndSolver
from src.random_solution import RandomSolver
from src.utils import read_file, build_distance_matrix, Point

solvers = (RandomSolver, NearestNeighborEndSolver)

if __name__ == '__main__':
    data_points = read_file("data/TSPA.csv")
    distances = build_distance_matrix(data_points)
    costs = [point.cost for point in data_points]
    for solver_class in solvers:
        solver = solver_class(distances, costs)
        print(solver_class.__name__, solver.get_statistics())

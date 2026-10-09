from random_solution import RandomSolver
from utils import read_file

def run_random_solution(data: list):
    solver = RandomSolver(data)
    print(solver.get_statistics())

if __name__ == '__main__':
    data = read_file("TSPA.csv")
    run_random_solution(data)


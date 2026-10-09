# Random - take random node (which was not selected yet) add it to the path
# 200 random solutions
import random

from src.utils import cycle_cost, Result

class RandomSolver:
    def __init__(self, distances: list, costs: list):
        self.distances = distances
        self.costs = costs
        self.n = len(costs)
        self.cycle_length = (self.n + 1) // 2

    def generate_cycle(self, init_point: int) -> list[int]:
        if init_point < 0 or init_point >= self.n:
            raise IndexError
        others = [i for i in range(self.n) if i != init_point]
        cycle = [init_point] + random.sample(others, self.cycle_length - 1)
        return cycle

    def get_statistics(self, runs: int = 200):
        cycles = [self.generate_cycle(index) for index in range(runs)]
        results = [cycle_cost(self.distances, self.costs, cycle) for cycle in cycles]
        return Result(
            min = min(results),
            max = max(results),
            average = sum(results) / len(results),
            best_cycle = cycles[results.index(min(results))]
        )
# Random - take random node (which was not selected yet) add it to the path
# 200 random solutions
import random

from utils import Result, Point, calculate_distance

class RandomSolver:
    def __init__(self, data):
        self.data = data

    def generate_solution(self, init_point: int, cycle_length: int) -> int:
        if init_point < 0 or init_point >= len(self.data):
            raise IndexError
        free = list(range(len(self.data)))
        random.shuffle(free)
        free.remove(init_point)
        free = free[:cycle_length - 1]
        free.append(init_point)
        total_cost = 0
        last_point = self.data[init_point]
        for index in free:
            next_point = self.data[index]
            cost = calculate_distance(next_point, last_point) + next_point.cost
            total_cost += cost

        return total_cost


    def get_statistics(self, runs: int = 200):
        results = []
        cycle_length = int((len(self.data) + 1) / 2)
        for index in range(runs):
            results.append(self.generate_solution(index, cycle_length))

        return Result(
            min = min(results),
            max = max(results),
            average = sum(results) / len(results),
        )
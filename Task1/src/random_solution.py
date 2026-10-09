# Random - take random node (which was not selected yet) add it to the path
# 200 random solutions
import random

from src.base_solver import BaseSolver

class RandomSolver(BaseSolver):
    def generate_cycle(self, init_point: int) -> list[int]:
        if init_point < 0 or init_point >= self.n:
            raise IndexError
        others = [i for i in range(self.n)]
        cycle = random.sample(others, self.cycle_length)
        return cycle
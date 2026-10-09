from abc import ABC, abstractmethod

from src.utils import Result, cycle_cost


class BaseSolver(ABC):
    def __init__(self, distances: list[list[int]], costs: list[int]):
        self.distances = distances
        self.costs = costs
        self.n = len(costs)
        self.cycle_length = (self.n + 1) // 2

    @abstractmethod
    def generate_cycle(self, init_point: int) -> list[int]:
        """Build one cycle starting from init_point. Each method implements this."""

    def get_statistics(self, runs: int = 200):
        cycles = [self.generate_cycle(index) for index in range(runs)]
        results = [cycle_cost(self.distances, self.costs, cycle) for cycle in cycles]
        return Result(
            min=min(results),
            max=max(results),
            average=sum(results) / len(results),
            best_cycle=cycles[results.index(min(results))],
        )

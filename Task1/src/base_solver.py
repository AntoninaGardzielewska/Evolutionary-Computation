from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable

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

    def get_statistics(
        self,
        runs: int = 200,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        cycles = []
        runtimes = []
        for index in range(runs):
            started_at = perf_counter()
            cycles.append(self.generate_cycle(index))
            runtimes.append(perf_counter() - started_at)
            completed = index + 1
            if progress_callback and (completed % 10 == 0 or completed == runs):
                progress_callback(completed, runs)

        results = [cycle_cost(self.distances, self.costs, cycle) for cycle in cycles]
        return Result(
            min=min(results),
            max=max(results),
            average=sum(results) / len(results),
            best_cycle=cycles[results.index(min(results))],
            min_time=min(runtimes),
            max_time=max(runtimes),
            average_time=sum(runtimes) / len(runtimes),
        )

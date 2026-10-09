# Nearest neighbor considering adding the node only at the end of the current path
import random

from src.base_solver import BaseSolver


class NearestNeighborEndPosition(BaseSolver):
    def generate_cycle(self, init_point: int) -> list[int]:
        cycle = [init_point]
        visited = {init_point}
        while len(cycle) < self.cycle_length:
            last = cycle[-1]
            best = min(
                (v for v in range(self.n) if v not in visited),
                key=lambda v: self.distances[last][v] + self.costs[v],
            )
            cycle.append(best)
            visited.add(best)
        return cycle

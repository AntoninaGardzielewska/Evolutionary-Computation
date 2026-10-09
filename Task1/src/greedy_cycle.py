from src.base_solver import BaseSolver


class GreedyCycle(BaseSolver):
    """Closed cycle from the start. Each step inserts the unvisited node
    into the cycle edge where the objective increases the least."""

    def generate_cycle(self, init_point: int) -> list[int]:
        d, c = self.distances, self.costs
        unvisited = set(range(self.n)) - {init_point}

        # the 2-node cycle uses the edge twice (there and back)
        second = min(unvisited, key=lambda v: 2 * d[init_point][v] + c[v])
        cycle = [init_point, second]
        unvisited.remove(second)

        while len(cycle) < self.cycle_length:
            best_delta, best_v, best_pos = None, None, None
            size = len(cycle)
            for v in unvisited:
                for i in range(size):
                    a, b = cycle[i], cycle[(i + 1) % size]  # includes closing edge
                    delta = d[a][v] + d[v][b] - d[a][b] + c[v]
                    if best_delta is None or delta < best_delta:
                        best_delta, best_v, best_pos = delta, v, i + 1

            cycle.insert(best_pos, best_v)
            unvisited.remove(best_v)
        return cycle
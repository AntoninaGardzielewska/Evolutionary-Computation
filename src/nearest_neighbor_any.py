from src.base_solver import BaseSolver


class NearestNeighborAnyPosition(BaseSolver):
    """Open path. Each step inserts the unvisited node at the position
    (beginning, inside, or end) with the smallest increase of the objective."""

    def generate_cycle(self, init_point: int) -> list[int]:
        d, c = self.distances, self.costs
        path = [init_point]
        unvisited = set(range(self.n)) - {init_point}

        while len(path) < self.cycle_length:
            best_delta, best_v, best_pos = None, None, None
            for v in unvisited:
                # insert at the beginning
                delta, pos = d[v][path[0]] + c[v], 0
                # insert at the end
                end_delta = d[path[-1]][v] + c[v]
                if end_delta < delta:
                    delta, pos = end_delta, len(path)
                # insert inside the path, between path[i] and path[i + 1]
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    inner = d[a][v] + d[v][b] - d[a][b] + c[v]
                    if inner < delta:
                        delta, pos = inner, i + 1
                if best_delta is None or delta < best_delta:
                    best_delta, best_v, best_pos = delta, v, pos

            path.insert(best_pos, best_v)
            unvisited.remove(best_v)
        return path

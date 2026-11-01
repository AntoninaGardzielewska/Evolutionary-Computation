from src.base_solver import BaseSolver


class _GreedyRegret(BaseSolver):
    def __init__(
        self,
        distances: list[list[int]],
        costs: list[int],
        weights: tuple[float, float] | None = None,
    ):
        super().__init__(distances, costs)
        self.weights = weights

    def generate_cycle(self, init_point: int) -> list[int]:
        d, c = self.distances, self.costs
        cycle = [init_point]
        unvisited = set(range(self.n)) - {init_point}

        if self.closed_cycle:
            second = min(
                unvisited,
                key=lambda v: 2 * d[init_point][v] + c[v],
            )
            cycle.append(second)
            unvisited.remove(second)

        while len(cycle) < self.cycle_length:
            best_key, best_v, best_pos = None, None, None

            for v in unvisited:
                options = []
                if self.closed_cycle:
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                        change = d[a][v] + d[v][b] - d[a][b] + c[v]
                        options.append((change, i + 1))
                else:
                    options.append((d[v][cycle[0]] + c[v], 0))
                    options.append((d[cycle[-1]][v] + c[v], len(cycle)))
                    for i in range(len(cycle) - 1):
                        a, b = cycle[i], cycle[i + 1]
                        change = d[a][v] + d[v][b] - d[a][b] + c[v]
                        options.append((change, i + 1))

                options.sort()
                best_change, position = options[0]
                regret = options[1][0] - best_change
                if self.weights is None:
                    score = regret
                else:
                    regret_weight, change_weight = self.weights
                    score = regret_weight * regret - change_weight * best_change

                # higher score wins; on a tie, the smaller change wins
                key = (score, -best_change)
                if best_key is None or key > best_key:
                    best_key, best_v, best_pos = key, v, position

            cycle.insert(best_pos, best_v)
            unvisited.remove(best_v)

        return cycle


class Greedy2RegretAnyPosition(_GreedyRegret):
    """Build a path using 2-regret, or weighted regret when weights are supplied."""

    closed_cycle = False


class Greedy2RegretCycle(_GreedyRegret):
    """Build a cycle using 2-regret, or weighted regret when weights are supplied."""

    closed_cycle = True

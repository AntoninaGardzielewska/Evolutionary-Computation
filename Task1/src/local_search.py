import random
from typing import Literal

from src.base_solver import BaseSolver


class LocalSearch(BaseSolver):
    """Improve a closed cycle using objective deltas for each candidate move."""

    def __init__(
        self,
        distances: list[list[int]],
        costs: list[int],
        search_type: Literal["steepest", "greedy"],
        intra_route_move: Literal["nodes", "edges"],
        starting_solution: Literal["random", "greedy"],
        greedy_solver: BaseSolver | None = None,
        rng: random.Random | None = None,
    ):
        super().__init__(distances, costs)
        if search_type not in ("steepest", "greedy"):
            raise ValueError(f"Unsupported local search type: {search_type}")
        if intra_route_move not in ("nodes", "edges"):
            raise ValueError(f"Unsupported intra-route move: {intra_route_move}")
        if starting_solution not in ("random", "greedy"):
            raise ValueError(f"Unsupported starting solution: {starting_solution}")
        if starting_solution == "greedy" and greedy_solver is None:
            raise ValueError(
                "A greedy solver is required for greedy starting solutions"
            )

        self.search_type = search_type
        self.intra_route_move = intra_route_move
        self.starting_solution = starting_solution
        self.greedy_solver = greedy_solver
        self.rng = rng or random.Random()

    def generate_cycle(self, init_point: int) -> list[int]:
        if init_point < 0 or init_point >= self.n:
            raise IndexError("Starting point is outside the instance")

        if self.starting_solution == "random":
            cycle = self.rng.sample(range(self.n), self.cycle_length)
        else:
            if self.greedy_solver is None:
                raise RuntimeError("Greedy starting solver was not configured")
            cycle = self.greedy_solver.generate_cycle(init_point)

        while True:
            moves = self._get_moves(cycle)
            if self.search_type == "greedy":
                improving_move = None
                for move in self._random_order(moves):
                    if self._move_delta(cycle, move) < 0:
                        improving_move = move
                        break
                if improving_move is None:
                    return cycle
                self._apply_move(cycle, improving_move)
                continue

            best_move = None
            best_delta = 0
            for move in moves:
                delta = self._move_delta(cycle, move)
                if delta < best_delta:
                    best_delta, best_move = delta, move
            if best_move is None:
                return cycle
            self._apply_move(cycle, best_move)

    def _random_order(self, moves: list[tuple[str, int, int]]):
        remaining = len(moves)
        while remaining:
            selected = self.rng.randrange(remaining)
            last = remaining - 1
            moves[selected], moves[last] = moves[last], moves[selected]
            remaining = last
            yield moves[remaining]

    def _get_moves(self, cycle: list[int]) -> list[tuple[str, int, int]]:
        size = len(cycle)
        if self.intra_route_move == "nodes":
            moves = [
                ("node-swap", first, second)
                for first in range(size)
                for second in range(first + 1, size)
            ]
        else:
            moves = [
                ("two-opt", first_edge, second_edge)
                for first_edge in range(size)
                for second_edge in range(first_edge + 2, size)
                if not (first_edge == 0 and second_edge == size - 1)
            ]

        selected = set(cycle)
        moves.extend(
            ("node-exchange", position, node)
            for position in range(size)
            for node in range(self.n)
            if node not in selected
        )
        return moves

    def _move_delta(self, cycle: list[int], move: tuple[str, int, int]) -> int:
        move_type, first, second = move
        if move_type == "node-swap":
            return self._swap_delta(cycle, first, second)
        if move_type == "two-opt":
            next_first = (first + 1) % len(cycle)
            next_second = (second + 1) % len(cycle)
            return (
                self.distances[cycle[first]][cycle[second]]
                + self.distances[cycle[next_first]][cycle[next_second]]
                - self.distances[cycle[first]][cycle[next_first]]
                - self.distances[cycle[second]][cycle[next_second]]
            )

        old_node = cycle[first]
        previous = cycle[(first - 1) % len(cycle)]
        following = cycle[(first + 1) % len(cycle)]
        edge_delta = (
            self.distances[previous][second]
            + self.distances[second][following]
            - self.distances[previous][old_node]
            - self.distances[old_node][following]
        )
        return edge_delta + self.costs[second] - self.costs[old_node]

    def _swap_delta(self, cycle: list[int], first: int, second: int) -> int:
        size = len(cycle)
        if size == 2:
            return 0
        first_node, second_node = cycle[first], cycle[second]
        if second == (first + 1) % size:
            previous = cycle[(first - 1) % size]
            following = cycle[(second + 1) % size]
            return (
                self.distances[previous][second_node]
                + self.distances[second_node][first_node]
                + self.distances[first_node][following]
                - self.distances[previous][first_node]
                - self.distances[first_node][second_node]
                - self.distances[second_node][following]
            )
        if first == (second + 1) % size:
            previous = cycle[(second - 1) % size]
            following = cycle[(first + 1) % size]
            return (
                self.distances[previous][first_node]
                + self.distances[first_node][second_node]
                + self.distances[second_node][following]
                - self.distances[previous][second_node]
                - self.distances[second_node][first_node]
                - self.distances[first_node][following]
            )

        first_previous = cycle[(first - 1) % size]
        first_following = cycle[(first + 1) % size]
        second_previous = cycle[(second - 1) % size]
        second_following = cycle[(second + 1) % size]
        return (
            self.distances[first_previous][second_node]
            + self.distances[second_node][first_following]
            + self.distances[second_previous][first_node]
            + self.distances[first_node][second_following]
            - self.distances[first_previous][first_node]
            - self.distances[first_node][first_following]
            - self.distances[second_previous][second_node]
            - self.distances[second_node][second_following]
        )

    def _apply_move(self, cycle: list[int], move: tuple[str, int, int]) -> None:
        move_type, first, second = move
        if move_type == "node-swap":
            cycle[first], cycle[second] = cycle[second], cycle[first]
        elif move_type == "two-opt":
            cycle[first + 1 : second + 1] = reversed(cycle[first + 1 : second + 1])
        else:
            cycle[first] = second

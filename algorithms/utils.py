from __future__ import annotations

import heapq
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.layout import DroneLayout

DIRECTIONS: list[tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

_bfs_cache: dict[tuple[str, tuple[int, int], tuple[int, int], bool], int | float] = {}
_dijkstra_cache: dict[
    tuple[str, tuple[int, int], tuple[int, int]],
    tuple[float, list[tuple[int, int]]],
] = {}


def bfs_distance(
    layout: DroneLayout | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    hunter_restricted: bool = False,
) -> int | float:
    """
    BFS shortest-path **distance** (number of steps) on the grid.

    Parameters
    ----------
    layout : DroneLayout | None
        The grid layout.  If *None*, falls back to using Manhattan distance.
    start, goal : tuple[int, int]
        Source and target grid cells.
    hunter_restricted : bool
        When True, only normal terrain ('.' and ' ') is traversable,
        modelling hunter movement that cannot cross fog, mountains or storms.

    Returns
    -------
    int | float
        Step distance, or float('inf') if goal is unreachable.

    The result is cached by (layout.name, start, goal, hunter_restricted)
    so repeated queries within the same game are O(1).
    """
    if start == goal:
        return 0
    if layout is None:
        return manhattan_distance(start, goal)

    key = (layout.name, start, goal, hunter_restricted)
    cached = _bfs_cache.get(key)
    if cached is not None:
        return cached

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            if next_pos in visited:
                continue
            if not (0 <= nx < layout.width and 0 <= ny < layout.height):
                continue
            if layout.walls[nx][ny]:
                continue
            if hunter_restricted:
                terrain = layout.get_terrain(nx, ny)
                if terrain not in (".", " "):
                    continue
            if next_pos == goal:
                _bfs_cache[key] = dist + 1
                return dist + 1
            visited.add(next_pos)
            queue.append((next_pos, dist + 1))

    _bfs_cache[key] = float("inf")
    return float("inf")


def dijkstra(
    layout: DroneLayout | None,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[float, list[tuple[int, int]]]:
    """
    Terrain-weighted shortest path using Dijkstra's algorithm.

    Returns both the total path cost and the ordered list of grid cells
    from start to goal.  If goal is unreachable, returns (float('inf'), [start]).

    Parameters
    ----------
    layout : DroneLayout | None
        The grid layout.  If None, falls back to Manhattan distance /
        direct two-point path.
    start, goal : tuple[int, int]
        Source and target grid cells.

    Returns
    -------
    tuple[float, list[tuple[int, int]]]
        (cost, path) where path is the list of positions from start
        to goal (inclusive on both ends).

    Results are cached by (layout.name, start, goal) so repeated
    queries within the same game are O(1).
    """
    if start == goal:
        return 0.0, [start]
    if layout is None:
        return float(manhattan_distance(start, goal)), [start, goal]

    key = (layout.name, start, goal)
    cached = _dijkstra_cache.get(key)
    if cached is not None:
        return cached

    dist_map: dict[tuple[int, int], float] = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    while pq:
        d, pos = heapq.heappop(pq)
        if pos == goal:
            break
        if d > dist_map.get(pos, float("inf")):
            continue
        x, y = pos
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < layout.width
                and 0 <= ny < layout.height
                and not layout.walls[nx][ny]
            ):
                step_cost = layout.get_terrain_cost(nx, ny)
                nd = d + step_cost
                if nd < dist_map.get((nx, ny), float("inf")):
                    dist_map[(nx, ny)] = nd
                    prev[(nx, ny)] = pos
                    heapq.heappush(pq, (nd, (nx, ny)))

    if goal not in prev:
        result: tuple[float, list[tuple[int, int]]] = (float("inf"), [start])
        _dijkstra_cache[key] = result
        return result

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()

    cost = dist_map[goal]
    result = (cost, path)
    _dijkstra_cache[key] = result
    # Also cache the reverse direction cost (same for undirected grid)
    reverse_key = (layout.name, goal, start)
    if reverse_key not in _dijkstra_cache:
        _dijkstra_cache[reverse_key] = (cost, list(reversed(path)))
    return result


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """
    Manhattan (L1) distance between two grid cells.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

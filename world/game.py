from __future__ import annotations

import sys
import time

from typing import TYPE_CHECKING
from enum import StrEnum
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from world.game_state import GameState
    from world.rules import GameRules
    from view.display import AdversarialDisplay


class Agent(ABC):
    """
    An agent that takes actions in a GameState.
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        The Agent will receive a GameState and must return an action
        from Directions.{North, South, East, West, Stop}
        """
        pass


class Directions(StrEnum):
    """
    The four cardinal directions, plus STOP.
    """

    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    STOP = "Stop"


class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos: tuple[float, float], direction: Directions) -> None:
        self.pos = pos
        self.direction = direction

    def get_position(self):
        """
        Get the current (x,y) position of the configuration.
        """
        return self.pos

    def get_direction(self):
        """
        Get the current direction of travel for the configuration.
        """
        return self.direction

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, Configuration):
            return False
        return self.pos == other.pos and self.direction == other.direction

    def __hash__(self):
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self):
        return "(x,y)=" + str(self.pos) + ", " + str(self.direction)

    def generate_successor(self, vector: tuple[float, float]) -> Configuration:
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vector_to_direction(vector)
        if direction == Directions.STOP:
            direction = self.direction
        return Configuration((x + dx, y + dy), direction)


class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a game board.
    """

    def __init__(self, width: int, height: int, initial_value: bool = False) -> None:
        if initial_value not in [False, True]:
            raise Exception("Grids can only contain booleans")
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initial_value for _ in range(height)] for _ in range(width)]

    def __getitem__(self, i: int) -> list[bool]:
        return self.data[i]

    def __setitem__(self, key: int, item: list[bool]) -> None:
        self.data[key] = item

    def __str__(self):
        out = [
            [str(self.data[x][y])[0] for x in range(self.width)]
            for y in range(self.height)
        ]
        out.reverse()
        return "\n".join(["".join(x) for x in out])

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, Grid):
            return False
        return self.data == other.data

    def __hash__(self):
        base = 1
        h = 0
        for line in self.data:
            for i in line:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        """
        Return a copy of the grid.
        """
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def count(self, item: bool = True) -> int:
        """
        Count the number of occurrences of item in the grid.
        """
        return sum([x.count(item) for x in self.data])


class Actions:
    """
    A collection of static methods for manipulating move actions.
    """

    # Directions
    _directions = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0),
        Directions.STOP: (0, 0),
    }

    _directions_as_list = _directions.items()

    TOLERANCE = 0.001

    @staticmethod
    def vector_to_direction(vector: tuple[float, float]) -> Directions:
        """
        Converts a vector (dx, dy) to a direction.
        """
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP

    @staticmethod
    def direction_to_vector(
        direction: Directions, speed: float = 1.0
    ) -> tuple[float, float]:
        """
        Converts a direction to a vector (dx, dy) of the given speed.
        """
        dx, dy = Actions._directions[direction]
        return (dx * speed, dy * speed)

    @staticmethod
    def get_possible_actions(config: Configuration, walls: Grid) -> list[Directions]:
        """
        Returns a list of possible actions given the walls.
        """
        possible: list[Directions] = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        if abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE:
            return [config.get_direction()]

        for dir, vec in Actions._directions_as_list:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]:
                possible.append(dir)

        return possible

    @staticmethod
    def get_successor(
        position: tuple[float, float], action: Directions
    ) -> tuple[float, float]:
        """
        Returns the successor position after applying the action vector to the position.
        """
        dx, dy = Actions.direction_to_vector(action)
        x, y = position
        return (x + dx, y + dy)


class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(
        self,
        agents: list[Agent],
        display: AdversarialDisplay,
        rules: type[GameRules],
        state: GameState,
        starting_index: int = 0,
    ):
        self.agents = agents
        self.display = display
        self.rules = rules
        self.starting_index = starting_index
        self.game_over = False
        self.state = state

    def run(self):
        """
        Main control loop for game play (multi-agent mode).
        """
        self._run_multi_agent()

    def _run_multi_agent(self):
        """
        Multi-agent game loop for adversarial mode.
        """
        self.display.initialize(state=self.state)
        time.sleep(1)
        self.num_moves = 0

        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.game_over = True
                return

        agent_index = self.starting_index
        num_agents = len(self.agents)

        while not self.game_over:
            agent = self.agents[agent_index]
            observation = self.state.deep_copy()

            action = None
            action = agent.get_action(observation)
            assert action is not None

            self.state = self.state.generate_successor(agent_index, action)

            self.display.update(self.state)
            self.rules.process(self.state, self)
            if agent_index == num_agents - 1:
                self.num_moves += 1
            agent_index = (agent_index + 1) % num_agents

        time.sleep(1)
        self.display.finish()

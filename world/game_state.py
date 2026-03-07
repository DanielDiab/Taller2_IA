from __future__ import annotations

from typing import TYPE_CHECKING

from world.game import Actions, Configuration, Directions

if TYPE_CHECKING:
    from world.game import Grid
    from world.layout import DroneLayout

DELIVERY_SCORE = 500
CAPTURE_SCORE = -1000
TIME_PENALTY = -1


class GameState:
    """
    Represents the full state of the drone vs. hunters adversarial game.

    Agent 0 is the drone (MAX player). Agents 1..n-1 are hunters (MIN players).
    The drone wins by visiting all delivery points (E). The drone loses if any
    hunter occupies the same cell as the drone after a move.
    """

    def __init__(self, layout: DroneLayout) -> None:
        self._drone_position: tuple[int, int] | None = None
        self._hunter_positions: list[tuple[int, int]] = []
        self._pending_deliveries: set[tuple[int, int]] = set()
        self._walls: Grid | None = None
        self._layout: DroneLayout
        self._score: int = 0
        self._win: bool = False
        self._lose: bool = False
        self._num_agents: int = 0
        self._init_from_layout(layout)

    def _init_from_layout(self, layout: DroneLayout) -> None:
        self._layout = layout
        self._walls = layout.walls

        if layout.agent_positions:
            self._drone_position = layout.agent_positions[0]
        if len(layout.agent_positions) > 1:
            self._hunter_positions = list(layout.agent_positions[1:])

        self._pending_deliveries = set(layout.delivery_positions)
        self._num_agents = 1 + len(self._hunter_positions)
        self._score = 0
        self._win = False
        self._lose = False

    def deep_copy(self) -> GameState:
        """
        Create a deep copy of the GameState.
        """
        state = GameState(self._layout)
        state._drone_position = self._drone_position
        state._hunter_positions = list(self._hunter_positions)
        state._pending_deliveries = set(self._pending_deliveries)
        state._walls = self._walls
        state._score = self._score
        state._win = self._win
        state._lose = self._lose
        state._num_agents = self._num_agents
        return state

    def get_drone_position(self) -> tuple[int, int] | None:
        """
        Get the current position of the drone (agent 0).
        """
        return self._drone_position

    def get_hunter_positions(self) -> list[tuple[int, int]]:
        """
        Get the current positions of the hunters (agents 1..n-1).
        """
        return list(self._hunter_positions)

    def get_hunter_position(self, agent_index: int) -> tuple[int, int]:
        """
        Get the current position of the specified hunter agent.
        """
        if agent_index < 1 or agent_index >= self._num_agents:
            raise IndexError(
                f"Invalid hunter agent_index {agent_index}. "
                f"Valid range: 1 to {self._num_agents - 1}"
            )
        return self._hunter_positions[agent_index - 1]

    def get_legal_actions(self, agent_index: int = 0) -> list[Directions]:
        """
        Get the legal actions for the specified agent.
        Agent 0 is the drone, agents 1..n-1 are hunters.
        """
        if self._win or self._lose:
            return []
        pos = self._get_agent_position(agent_index)
        assert pos is not None
        config = Configuration(pos, Directions.STOP)
        assert self._walls is not None
        actions = Actions.get_possible_actions(config, self._walls)

        # Hunters can only move through normal terrain (.)
        if agent_index > 0:
            filtered: list[Directions] = []
            for action in actions:
                successor_pos = Actions.get_successor(pos, action)
                sx, sy = int(successor_pos[0]), int(successor_pos[1])
                terrain = self._layout.get_terrain(sx, sy)
                if terrain in (".", " ") or action == Directions.STOP:
                    filtered.append(action)
            return filtered if filtered else [Directions.STOP]

        return actions

    def generate_successor(self, agent_index: int, action: Directions) -> GameState:
        """
        Generate the successor GameState after the specified agent takes the given action.
        """
        if self._win or self._lose:
            raise Exception("Cannot generate successor of a terminal state.")

        legal = self.get_legal_actions(agent_index)
        if action not in legal:
            raise Exception(
                f"Illegal action {action} for agent {agent_index}. Legal actions: {legal}"
            )

        successor = self.deep_copy()

        if agent_index == 0:
            successor._apply_drone_action(action)
        else:
            successor._apply_hunter_action(agent_index, action)

        successor._check_terminal_conditions()
        return successor

    def _apply_drone_action(self, action: Directions) -> None:
        """
        Update the GameState by applying the drone's action.
        """
        assert self._drone_position is not None
        new_pos = Actions.get_successor(self._drone_position, action)
        self._drone_position = (int(new_pos[0]), int(new_pos[1]))
        self._score += TIME_PENALTY

        if self._drone_position in self._pending_deliveries:
            self._pending_deliveries.remove(self._drone_position)
            self._score += DELIVERY_SCORE

    def _apply_hunter_action(self, agent_index: int, action: Directions) -> None:
        """
        Update the GameState by applying the specified hunter's action.
        """
        hunter_idx = agent_index - 1
        old_pos = self._hunter_positions[hunter_idx]
        new_pos = Actions.get_successor(old_pos, action)
        self._hunter_positions[hunter_idx] = (int(new_pos[0]), int(new_pos[1]))

    def _check_terminal_conditions(self) -> None:
        """
        Check if the game has been won or lost after a move.
        """
        for hunter_pos in self._hunter_positions:
            if hunter_pos == self._drone_position:
                self._lose = True
                self._score += CAPTURE_SCORE
                return

        if len(self._pending_deliveries) == 0:
            self._win = True

    def is_win(self) -> bool:
        """
        Check if the game has been won (all deliveries completed).
        """
        return self._win

    def is_lose(self) -> bool:
        """
        Check if the game has been lost (drone captured by a hunter).
        """
        return self._lose

    def get_pending_deliveries(self) -> set[tuple[int, int]]:
        """
        Get the set of pending delivery positions.
        """
        return set(self._pending_deliveries)

    def get_num_agents(self) -> int:
        """
        Get the total number of agents in the game (1 drone + number of hunters).
        """
        return self._num_agents

    def get_score(self) -> int:
        """
        Get the current score of the game.
        """
        return self._score

    def get_walls(self) -> Grid | None:
        """
        Get the grid representing wall positions.
        """
        return self._walls

    def get_layout(self) -> DroneLayout | None:
        """
        Get the layout object associated with this GameState.
        """
        return self._layout

    def _get_agent_position(self, agent_index: int) -> tuple[int, int] | None:
        """
        Get the current position of the specified agent.
        Agent 0 is the drone, agents 1..n-1 are hunters.
        """
        if agent_index == 0:
            return self._drone_position
        return self.get_hunter_position(agent_index)

    def __eq__(self, other: object) -> bool:
        if other is None or not isinstance(other, GameState):
            return False
        return (
            self._drone_position == other._drone_position
            and self._hunter_positions == other._hunter_positions
            and self._pending_deliveries == other._pending_deliveries
            and self._score == other._score
            and self._win == other._win
            and self._lose == other._lose
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._drone_position,
                tuple(self._hunter_positions),
                frozenset(self._pending_deliveries),
                self._score,
                self._win,
                self._lose,
            )
        )

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append(f"Drone: {self._drone_position}")
        lines.append(f"Hunters: {self._hunter_positions}")
        lines.append(f"Pending deliveries: {sorted(self._pending_deliveries)}")
        lines.append(f"Score: {self._score}")
        if self._win:
            lines.append("State: WIN")
        elif self._lose:
            lines.append("State: LOSE")
        return "\n".join(lines)

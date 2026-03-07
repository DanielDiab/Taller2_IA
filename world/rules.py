from __future__ import annotations

import random
from typing import TYPE_CHECKING

from algorithms.utils import bfs_distance
from world.game import Actions, Agent, Directions, Game
from world.game_state import GameState

if TYPE_CHECKING:
    from world.layout import DroneLayout
    from world.runner import AdversarialDisplay

MAX_MOVES = 500


class GameRules:
    """
    Rules for the drone vs. hunters adversarial game.
    Used by the Game class to manage game flow.
    """

    @staticmethod
    def new_game(
        layout: DroneLayout,
        drone_agent: Agent,
        hunter_agents: list[Agent],
        display: AdversarialDisplay,
        quiet: bool = False,
    ) -> Game:
        """
        Creates a new Game instance with the given layout, agents, and display.
        """
        agents: list[Agent] = [drone_agent] + hunter_agents
        game = Game(agents, display, GameRules, GameState(layout))
        GameRules._quiet = quiet
        GameRules._move_count = 0
        return game

    @staticmethod
    def process(state: GameState, game: Game) -> None:
        """
        Check terminal conditions after each move.
        """
        GameRules._move_count += 1
        if state.is_win():
            if not GameRules._quiet:
                print("All deliveries completed! Score: %d" % state.get_score())
            game.game_over = True
        elif state.is_lose():
            if not GameRules._quiet:
                print("Drone captured! Score: %d" % state.get_score())
            game.game_over = True
        elif GameRules._move_count >= MAX_MOVES:
            if not GameRules._quiet:
                print("Move limit reached! Score: %d" % state.get_score())
            game.game_over = True


class HunterAgent(Agent):
    """
    Simple hunter agent that chases the drone.
    Uses greedy approach: moves toward the drone position.
    Uses BFS distance to navigate around obstacles instead of getting stuck.
    If drone is unreachable, does not move.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    def get_action(self, state: GameState) -> Directions:
        """
        Get the action for this hunter agent based on the current GameState.
        Uses BFS distance to move towards the drone while navigating around walls.
        If the drone is unreachable, returns STOP.
        """
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        drone_pos = state.get_drone_position()
        assert drone_pos is not None
        hunter_pos = state.get_hunter_position(self.index)
        layout = state.get_layout()

        best_action = Directions.STOP
        best_dist: int | float = float("inf")

        for action in legal_actions:
            successor = Actions.get_successor(hunter_pos, action)
            sx, sy = int(successor[0]), int(successor[1])
            dist = bfs_distance(layout, (sx, sy), drone_pos, hunter_restricted=True)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action


class RandomHunterAgent(Agent):
    """
    Random hunter agent that picks actions uniformly at random.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    def get_action(self, state: GameState) -> Directions:
        """
        Get a random legal action for this hunter agent.
        If no legal actions are available, returns STOP.
        """
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        return random.choice(legal_actions)


class MixedHunterAgent(Agent):
    """
    Hunter agent that acts greedily with probability (1 - p) and randomly
    with probability p, modelling the -p flag at game-play time.
    """

    def __init__(self, index: int, probability: float) -> None:
        self.index = index
        self.probability = probability
        self._greedy = HunterAgent(index)
        self._random = RandomHunterAgent(index)

    def get_action(self, state: GameState) -> Directions:
        """
        Get an action for this hunter agent based on the current GameState.
        With probability p, returns a random action. Otherwise, returns a greedy action.
        """
        if random.random() < self.probability:
            return self._random.get_action(state)
        return self._greedy.get_action(state)

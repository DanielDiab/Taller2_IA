from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from algorithms.utils import bfs_distance
from world.game import Agent, Directions, Actions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        num_agents = state.get_num_agents()
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        def value(s: GameState, depth_left: int, agent_index: int) -> float:
            # Terminal / cutoff
            if s.is_win() or s.is_lose() or depth_left == 0:
                return float(self.evaluation_function(s))

            legal = s.get_legal_actions(agent_index)
            if not legal:
                return float(self.evaluation_function(s))

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth_left - 1 if next_agent == 0 else depth_left

            if agent_index == 0:  # MAX (drone)
                best = float("-inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    best = max(best, value(succ, next_depth, next_agent))
                return best
            else:  # MIN (hunters)
                best = float("inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    best = min(best, value(succ, next_depth, next_agent))
                return best

        # Root: choose action that maximizes minimax value
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = legal_actions[0]
        best_value = float("-inf")
        for a in legal_actions:
            succ = state.generate_successor(0, a)
            v = value(succ, self.depth, 1 if num_agents > 1 else 0)
            if v > best_value:
                best_value = v
                best_action = a

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        return None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        num_agents = state.get_num_agents()
        p = self.prob

        def value(s: GameState, depth_left: int, agent_index: int) -> float:
            # Terminal / cutoff
            if s.is_win() or s.is_lose() or depth_left == 0:
                return float(self.evaluation_function(s))

            legal = s.get_legal_actions(agent_index)
            if not legal:
                return float(self.evaluation_function(s))

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth_left - 1 if next_agent == 0 else depth_left

            # MAX (drone)
            if agent_index == 0:
                best = float("-inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    best = max(best, value(succ, next_depth, next_agent))
                return best

            # CHANCE (hunter): MixedHunterAgent model from world/rules.py
            layout = s.get_layout()
            drone_pos = s.get_drone_position()
            hunter_pos = s.get_hunter_position(agent_index)

            # If positions missing, fallback to uniform expected value
            if drone_pos is None or hunter_pos is None:
                total = 0.0
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    total += value(succ, next_depth, next_agent)
                return total / len(legal)

            # 1) Greedy action = minimize BFS distance to drone (same as HunterAgent)
            best_action = Directions.STOP
            best_dist = float("inf")

            for a in legal:
                successor_pos = Actions.get_successor(hunter_pos, a)
                sx, sy = int(successor_pos[0]), int(successor_pos[1])
                dist = bfs_distance(layout, (sx, sy), drone_pos, hunter_restricted=True)
                if dist < best_dist:
                    best_dist = dist
                    best_action = a

            succ_greedy = s.generate_successor(agent_index, best_action)
            v_greedy = value(succ_greedy, next_depth, next_agent)

            # 2) Random part = average over all legal actions
            total = 0.0
            for a in legal:
                succ = s.generate_successor(agent_index, a)
                total += value(succ, next_depth, next_agent)
            v_random = total / len(legal)

            return (1 - p) * v_greedy + p * v_random

        # Root: choose action that maximizes expected value
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = legal_actions[0]
        best_value = float("-inf")

        first_hunter = 1 if num_agents > 1 else 0
        for a in legal_actions:
            succ = state.generate_successor(0, a)
            v = value(succ, self.depth, first_hunter)
            if v > best_value:
                best_value = v
                best_action = a

        return best_action
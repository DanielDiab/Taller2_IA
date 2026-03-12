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
        numero_agentes = state.get_num_agents() #total de agentes en el juego (1 dron + n cazadores)
        probabilidad_random = self.prob #probabilidad de que cada cazador actúe aleatoriamente (0=greedy, 1=random)

        def v(s: GameState, profundidad: int, iagente: int) -> float:
            if s.is_win() or s.is_lose() or profundidad == 0: #casos base terminal o corte por profundidad
                return float(self.evaluation_function(s))

            acciones_l = s.get_legal_actions(iagente) #acciones legales para el agente actual (dron o cazador)
            if not acciones_l:
                return float(self.evaluation_function(s)) #si no hay acciones legales, evaluar el estado actual

            siguiente_agente = (iagente + 1) % numero_agentes
            profundidad_siguiente = profundidad - 1 if siguiente_agente == 0 else profundidad #disminuir profundidad solo después de que todos los agentes hayan movido (ciclo completo)

            # accion que maximiza el valor esperado para el dron (MAX)
            if iagente == 0:
                m_valor = float("-inf")
                for a in acciones_l: #iterar sobre las acciones legales del dron
                    succ = s.generate_successor(iagente, a)
                    m_valor = max(m_valor, v(succ, profundidad_siguiente, siguiente_agente)) # actualizar el valor máximo encontrado para el dron
                return m_valor

            # cazadores se modela greedy como se especifica en las reglas (rules.py)
            layout = s.get_layout()
            drone_pos = s.get_drone_position()
            cazador_pos = s.get_hunter_position(iagente) #mapa, posicion dron y posicion del cazador actual

            # si falta informacion (cazador o dron no visible), usar solo la parte aleatoria (media de los hijos)
            if drone_pos is None or cazador_pos is None:
                total = 0.0
                for a in acciones_l:
                    succ = s.generate_successor(iagente, a)
                    total += v(succ, profundidad_siguiente, siguiente_agente)
                return total / len(acciones_l)

            # Greedy, esta es la parte de greedy en donde se calcula la accion que minimiza la distancia al dron (minimizar BFS)
            m_accion = Directions.STOP
            m_distancia = float("inf")

            for a in acciones_l: #poscicion siguiente del cazador para cada accion legal
                pos_siguiente = Actions.get_successor(cazador_pos, a)
                sx, sy = int(pos_siguiente[0]), int(pos_siguiente[1])
                distancia = bfs_distance(layout, (sx, sy), drone_pos, hunter_restricted=True) #calcular la distancia BFS desde la posicion siguiente del cazador hasta el dron (con restricciones para cazadores)
                if distancia < m_distancia: #elegir la accion que minimiza la distancia al dron
                    m_distancia = distancia
                    m_accion = a

            estado_greedy = s.generate_successor(iagente, m_accion)
            v_greedy = v(estado_greedy, profundidad_siguiente, siguiente_agente) #valor del estado resultante de la accion greedy para el cazador

            # Parte aleatoria: promedio de los valores de los estados resultantes de cada accion legal
            total = 0.0
            for a in acciones_l:
                succ = s.generate_successor(iagente, a)
                total += v(succ, profundidad_siguiente, siguiente_agente)
            v_random = total / len(acciones_l)

            return (1 - probabilidad_random) * v_greedy + probabilidad_random * v_random #valor esperado para el cazador con el modelo mixto: combinación ponderada del valor greedy y el valor aleatorio

        # Raíz: elegir la acción que maximiza el valor esperado para el dron
        acciones_dron = state.get_legal_actions(0)
        if not acciones_dron:
            return None

        m_accion_dron = acciones_dron[0]
        m_valor_dron = float("-inf")

        cazador_i = 1 if numero_agentes > 1 else 0 #primer cazador (si existe) para la siguiente llamada recursiva después de la acción del dron
        for a in acciones_dron:
            succ = state.generate_successor(0, a)
            valor_succ = v(succ, self.depth, cazador_i)  
            if valor_succ > m_valor_dron:
                m_valor_dron = valor_succ #estados del dron y cazador resultantes de la acción del dron
                m_accion_dron = a

        return m_accion_dron #acción que maximiza el valor esperado para el dron en la raíz del árbol de búsqueda
    

## PRIMERA SOLUCION AUTONOMA
"""
    numero_agentes = state.get_num_agents()
    probabilidad_random = self.prob

    def valor(s: GameState, profundidad: int, iagente: int) -> float:
        if s.is_win() or s.is_lose() or profundidad == 0:
            return float(self.evaluation_function(s))

        acciones_l = s.get_legal_actions(iagente)
        if not acciones_l:
            return float(self.evaluation_function(s))

        siguiente_agente = (iagente + 1) % numero_agentes
        profundidad_siguiente = profundidad - 1 if siguiente_agente == 0 else profundidad

        # MAX (dron)
        if iagente == 0:
            mejor_valor = float("-inf")
            for accion in acciones_l:
                succ = s.generate_successor(iagente, accion)
                mejor_valor = max(mejor_valor, valor(succ, profundidad_siguiente, siguiente_agente))
            return mejor_valor

        # Usa (1-p)*min(valores_hijos) en vez de (1-p)*V(succ_greedyBFS)
        valores_hijos = []
        for accion in acciones_l:
            succ = s.generate_successor(iagente, accion)
            valores_hijos.append(valor(succ, profundidad_siguiente, siguiente_agente))

        valor_greedy_mal = min(valores_hijos)                      
        valor_random = sum(valores_hijos) / len(valores_hijos)     

        return (1 - probabilidad_random) * valor_greedy_mal + probabilidad_random * valor_random

    acciones_dron = state.get_legal_actions(0)
    if not acciones_dron:
        return None

    mejor_accion = acciones_dron[0]
    mejor_valor = float("-inf")
    primer_cazador = 1 if numero_agentes > 1 else 0

    for accion in acciones_dron:
        succ = state.generate_successor(0, accion)
        valor_succ = valor(succ, self.depth, primer_cazador)
        if valor_succ > mejor_valor:
            mejor_valor = valor_succ
            mejor_accion = accion

    return mejor_accion 


Prompt 1
“Implementa Expectimax para el dron (MAX) y cazadores como CHANCE con probabilidad p de acción aleatoria. Debe correr con -p y -d.”

Lo que se tomó/corrigió del intento:
	•	Se construyó la recursión value(s, depth, agent_index) con:
	•	caso MAX: max(...)
	•	caso CHANCE: combinación (1-p)*... + p*mean(...)
	•	Se agregó lógica de next_agent y reducción de depth solo cuando vuelve al dron.

⸻

Prompt 2
“El enunciado dice que el cazador greedy no es minimax: en world/rules.py persigue usando BFS distance. Ajusta Expectimax para que su parte greedy replique exactamente ese comportamiento.”

Lo que se tomó/corrigió del intento:
	•	Se reemplazó min(child_vals) por una sola transición greedy:
	•	calcular best_action minimizando bfs_distance(layout, successor_pos, drone_pos, hunter_restricted=True)
	•	usar v_greedy = value(generate_successor(hunter, best_action), ...)
	•	Se mantuvo el promedio uniforme para la parte random.

⸻

Prompt 3
“Me falla el código por nombres/funciones del framework. Hazlo compatible con el proyecto: usa get_layout(), get_drone_position(), get_hunter_position(), Actions.get_successor(), get_legal_actions() y generate_successor().”

Lo que se tomó/corrigió del intento:
	•	Se alinearon las llamadas a métodos del GameState del proyecto:
	•	s.get_layout(), s.get_drone_position(), s.get_hunter_position(agent_index)
	•	Actions.get_successor(hunter_pos, action)
	•	s.generate_successor(agent_index, action)
	•	Se agregó fallback si faltan posiciones: promedio uniforme.

⸻

Prompt 4
“Haz que el agente retorne la acción óptima desde la raíz (no el valor), y que no se caiga si no hay acciones legales.”

Lo que se tomó/corrigió del intento:
	•	Se implementó selección en la raíz:
	•	iterar acciones del dron, escoger la que maximiza value(...)
	•	if not legal_actions: return None
    """
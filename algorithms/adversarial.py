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
        self.index = 0
        self.depth = int(depth)
        self.prob = float(prob)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        pass


class RandomAgent(MultiAgentSearchAgent):
    def get_action(self, state: GameState) -> Directions | None:
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        num_agents = state.get_num_agents()

        def value(s: GameState, depth_left: int, agent_index: int) -> float:
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

        # -----------------------------------------------------------------------
        # PRIMERA VERSION AUTONOMA
        #
        # Idea: recorrer el árbol de izquierda a derecha. En cada nivel, si es el
        # dron (MAX) elegir el mayor valor; si es un cazador (MIN) elegir el menor.
        # La profundidad se reduce en cada llamada recursiva.
        #
        # Problemas:
        #   1. depth se restaba en cada agente, no solo al completar el turno completo.
        #      Con dos cazadores, un turno consumía 3 niveles de depth en vez de 1.
        #   2. No se manejaba el caso de lista de acciones vacía en la raíz.
        #   3. generate_successor puede lanzar excepción en estado terminal, por lo que
        #      la verificación is_win/is_lose debe ir antes de llamar get_legal_actions.
        # -----------------------------------------------------------------------
        # def minimax(s, depth, agent):
        #     if s.is_win() or s.is_lose() or depth == 0:
        #         return self.evaluation_function(s)
        #
        #     acciones = s.get_legal_actions(agent)
        #     if not acciones:
        #         return self.evaluation_function(s)
        #
        #     siguiente = (agent + 1) % num_agents
        #     nueva_depth = depth - 1  # ERROR: debería bajar solo cuando next_agent == 0
        #
        #     if agent == 0:  # MAX
        #         return max(
        #             minimax(s.generate_successor(agent, a), nueva_depth, siguiente)
        #             for a in acciones
        #         )
        #     else:           # MIN
        #         return min(
        #             minimax(s.generate_successor(agent, a), nueva_depth, siguiente)
        #             for a in acciones
        #         )
        #
        # acciones_dron = state.get_legal_actions(0)
        # # ERROR: no se manejaba lista vacía -> max() de iterable vacío lanza excepción
        # mejor = max(
        #     acciones_dron,
        #     key=lambda a: minimax(state.generate_successor(0, a), self.depth, 1)
        # )
        # return mejor
        #
        # -----------------------------------------------------------------------
        # PROMPTS USADOS
        # -----------------------------------------------------------------------
        # Prompt 1:
        # "Tengo un agente Minimax para un juego dron vs cazadores. El dron es MAX
        #  y cada cazador es MIN. Implementa get_action() con una función recursiva
        #  value(state, depth, agent_index). depth se reduce solo cuando todos los
        #  agentes han movido (cuando next_agent vuelve a 0). Usa get_legal_actions(),
        #  generate_successor(), is_win(), is_lose() y self.evaluation_function()."
        #
        # Lo que se corrigió vs el intento autónomo:
        #   - La reducción de depth se hace solo cuando next_agent == 0 (turno completo),
        #     no en cada llamada recursiva. Esto es: next_depth = depth-1 if next_agent==0
        #     else depth.
        #   - Se añadió el guard 'if not legal_actions: return None' en la raíz.
        #
        # Prompt 2:
        # "El framework lanza excepción si llamo generate_successor en estado terminal.
        #  Asegúrate de verificar is_win()/is_lose() al inicio de value(), antes de
        #  cualquier otra operación."
        #
        # Lo que se corrigió:
        #   - La verificación terminal se colocó como primera línea de value(), antes
        #     de get_legal_actions() y generate_successor().
        # -----------------------------------------------------------------------


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Produce las mismas decisiones que MinimaxAgent
    pero explora menos nodos gracias a la poda.

    Invariantes:
      - alpha: mejor valor que MAX puede garantizar en el camino actual (inicia en -inf)
      - beta:  mejor valor que MIN puede garantizar en el camino actual (inicia en +inf)
      - Nodo MAX poda cuando valor > beta  (desigualdad estricta)
      - Nodo MIN poda cuando valor < alpha (desigualdad estricta)
    """

    def get_action(self, state: GameState) -> Directions | None:
        num_agents = state.get_num_agents()

        def value(
            s: GameState,
            depth_left: int,
            agent_index: int,
            alpha: float,
            beta: float,
        ) -> float:
            if s.is_win() or s.is_lose() or depth_left == 0:
                return float(self.evaluation_function(s))

            legal = s.get_legal_actions(agent_index)
            if not legal:
                return float(self.evaluation_function(s))

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth_left - 1 if next_agent == 0 else depth_left

            if agent_index == 0:
                # Nodo MAX (dron)
                v = float("-inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    v = max(v, value(succ, next_depth, next_agent, alpha, beta))
                    if v > beta:    # poda: MIN nunca permitirá este camino
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                # Nodo MIN (cazador)
                v = float("inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    v = min(v, value(succ, next_depth, next_agent, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = legal_actions[0]
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for a in legal_actions:
            succ = state.generate_successor(0, a)
            v = value(succ, self.depth, 1 if num_agents > 1 else 0, alpha, beta)
            if v > best_value:
                best_value = v
                best_action = a
            alpha = max(alpha, best_value)

        return best_action

        # -----------------------------------------------------------------------
        # PRIMERA VERSION AUTONOMA
        #
        # Idea: igual que Minimax pero con dos variables alpha (-inf) y beta (+inf).
        # En nodos MAX: si el valor supera beta, cortar y actualizar alpha.
        # En nodos MIN: si el valor baja de alpha, cortar y actualizar beta.
        #
        # Problemas:
        #   1. La condición de poda usaba >= y <= (no estricto), causando cortes
        #      incorrectos cuando dos ramas tienen el mismo valor (empates).
        #   2. El mismo error de depth que Minimax: se restaba en cada agente.
        #   3. No se actualizaba alpha en la raíz entre acciones del dron, por lo que
        #      las podas en ramas posteriores de la raíz no eran efectivas.
        # -----------------------------------------------------------------------
        # def alphabeta(s, depth, agent, alpha, beta):
        #     if s.is_win() or s.is_lose() or depth == 0:
        #         return self.evaluation_function(s)
        #
        #     acciones = s.get_legal_actions(agent)
        #     siguiente = (agent + 1) % num_agents
        #     nueva_depth = depth - 1  # ERROR: mismo problema de depth que Minimax
        #
        #     if agent == 0:  # MAX
        #         v = float("-inf")
        #         for a in acciones:
        #             succ = s.generate_successor(agent, a)
        #             v = max(v, alphabeta(succ, nueva_depth, siguiente, alpha, beta))
        #             alpha = max(alpha, v)
        #             if alpha >= beta:  # ERROR: debe ser alpha > beta (estricto)
        #                 break
        #         return v
        #     else:           # MIN
        #         v = float("inf")
        #         for a in acciones:
        #             succ = s.generate_successor(agent, a)
        #             v = min(v, alphabeta(succ, nueva_depth, siguiente, alpha, beta))
        #             beta = min(beta, v)
        #             if beta <= alpha:  # ERROR: debe ser beta < alpha (estricto)
        #                 break
        #         return v
        #
        # acciones_dron = state.get_legal_actions(0)
        # mejor = max(
        #     acciones_dron,
        #     key=lambda a: alphabeta(
        #         state.generate_successor(0, a), self.depth, 1,
        #         float("-inf"), float("inf")
        #     )
        # )
        # return mejor
        # # ERROR: alpha no se actualiza entre iteraciones de la raíz
        #
        # -----------------------------------------------------------------------
        # PROMPTS USADOS
        # -----------------------------------------------------------------------
        # Prompt 1:
        # "Implementa AlphaBetaAgent igual que MinimaxAgent pero con poda alfa-beta.
        #  Alpha empieza en -inf (mejor garantía para MAX), beta en +inf (mejor para MIN).
        #  Nodo MAX poda con desigualdad ESTRICTA v > beta. Nodo MIN poda con v < alpha.
        #  Pasa alpha y beta como parámetros en cada llamada recursiva."
        #
        # Lo que se corrigió vs el intento autónomo:
        #   - La condición de poda usa > y < estrictos (no >= ni <=), evitando
        #     cortes incorrectos en empates entre valores.
        #   - La reducción de depth se corrigió igual que en Minimax.
        #
        # Prompt 2:
        # "En la raíz del árbol también hay que actualizar alpha después de evaluar
        #  cada acción del dron, para que las podas en ramas posteriores sean efectivas.
        #  Ajusta el loop raíz para que haga alpha = max(alpha, best_value) en cada
        #  iteración."
        #
        # Lo que se corrigió:
        #   - Se agregó 'alpha = max(alpha, best_value)' dentro del loop de la raíz,
        #     permitiendo podar ramas restantes cuando ya se encontró un valor alto.
        # -----------------------------------------------------------------------


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent con modelo mixto de cazadores.

    Cada cazador actúa aleatoriamente con probabilidad self.prob y de forma
    greedy (BFS hacia el dron) con probabilidad (1 - self.prob).

    Fórmula nodo de azar:
        value = (1 - p) * v_greedy + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        numero_agentes = state.get_num_agents()
        probabilidad_random = self.prob

        def v(s: GameState, profundidad: int, iagente: int) -> float:
            if s.is_win() or s.is_lose() or profundidad == 0:
                return float(self.evaluation_function(s))

            acciones_l = s.get_legal_actions(iagente)
            if not acciones_l:
                return float(self.evaluation_function(s))

            siguiente_agente = (iagente + 1) % numero_agentes
            profundidad_siguiente = profundidad - 1 if siguiente_agente == 0 else profundidad

            if iagente == 0:  # MAX (dron)
                m_valor = float("-inf")
                for a in acciones_l:
                    succ = s.generate_successor(iagente, a)
                    m_valor = max(m_valor, v(succ, profundidad_siguiente, siguiente_agente))
                return m_valor

            # Cazador: modelo mixto greedy + aleatorio
            layout = s.get_layout()
            drone_pos = s.get_drone_position()
            cazador_pos = s.get_hunter_position(iagente)

            if drone_pos is None or cazador_pos is None:
                total = sum(
                    v(s.generate_successor(iagente, a), profundidad_siguiente, siguiente_agente)
                    for a in acciones_l
                )
                return total / len(acciones_l)

            # Acción greedy: minimiza distancia BFS al dron
            m_accion = Directions.STOP
            m_distancia = float("inf")
            for a in acciones_l:
                pos_siguiente = Actions.get_successor(cazador_pos, a)
                sx, sy = int(pos_siguiente[0]), int(pos_siguiente[1])
                distancia = bfs_distance(layout, (sx, sy), drone_pos, hunter_restricted=True)
                if distancia < m_distancia:
                    m_distancia = distancia
                    m_accion = a

            estado_greedy = s.generate_successor(iagente, m_accion)
            v_greedy = v(estado_greedy, profundidad_siguiente, siguiente_agente)

            # Parte aleatoria: promedio uniforme de todos los sucesores
            total = sum(
                v(s.generate_successor(iagente, a), profundidad_siguiente, siguiente_agente)
                for a in acciones_l
            )
            v_random = total / len(acciones_l)

            return (1 - probabilidad_random) * v_greedy + probabilidad_random * v_random

        acciones_dron = state.get_legal_actions(0)
        if not acciones_dron:
            return None

        m_accion_dron = acciones_dron[0]
        m_valor_dron = float("-inf")
        cazador_i = 1 if numero_agentes > 1 else 0

        for a in acciones_dron:
            succ = state.generate_successor(0, a)
            valor_succ = v(succ, self.depth, cazador_i)
            if valor_succ > m_valor_dron:
                m_valor_dron = valor_succ
                m_accion_dron = a

        return m_accion_dron


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
"Implementa Expectimax para el dron (MAX) y cazadores como CHANCE con probabilidad p de acción aleatoria. Debe correr con -p y -d."

Lo que se tomó/corrigió del intento:
	•	Se construyó la recursión value(s, depth, agent_index) con:
	•	caso MAX: max(...)
	•	caso CHANCE: combinación (1-p)*... + p*mean(...)
	•	Se agregó lógica de next_agent y reducción de depth solo cuando vuelve al dron.

⸻

Prompt 2
"El enunciado dice que el cazador greedy no es minimax: en world/rules.py persigue usando BFS distance. Ajusta Expectimax para que su parte greedy replique exactamente ese comportamiento."

Lo que se tomó/corrigió del intento:
	•	Se reemplazó min(child_vals) por una sola transición greedy:
	•	calcular best_action minimizando bfs_distance(layout, successor_pos, drone_pos, hunter_restricted=True)
	•	usar v_greedy = value(generate_successor(hunter, best_action), ...)
	•	Se mantuvo el promedio uniforme para la parte random.

⸻

Prompt 3
"Me falla el código por nombres/funciones del framework. Hazlo compatible con el proyecto: usa get_layout(), get_drone_position(), get_hunter_position(), Actions.get_successor(), get_legal_actions() y generate_successor()."

Lo que se tomó/corrigió del intento:
	•	Se alinearon las llamadas a métodos del GameState del proyecto:
	•	s.get_layout(), s.get_drone_position(), s.get_hunter_position(agent_index)
	•	Actions.get_successor(hunter_pos, action)
	•	s.generate_successor(agent_index, action)
	•	Se agregó fallback si faltan posiciones: promedio uniforme.

⸻

Prompt 4
"Haz que el agente retorne la acción óptima desde la raíz (no el valor), y que no se caiga si no hay acciones legales."

Lo que se tomó/corrigió del intento:
	•	Se implementó selección en la raíz:
	•	iterar acciones del dron, escoger la que maximiza value(...)
	•	if not legal_actions: return None
    """
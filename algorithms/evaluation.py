from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
  
    from algorithms.utils import bfs_distance

    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = state.get_pending_deliveries()

    if drone_pos is None:
        return 0.0

    score = 0.0

    score += 1.0 * state.get_score()

    score += -50.0 * len(pending_deliveries)

    if pending_deliveries:
        min_delivery_dist = min(
            bfs_distance(layout, drone_pos, ep, hunter_restricted=False)
            for ep in pending_deliveries
        )
        if min_delivery_dist == float("inf"):
            score -= 200.0 
        else:
            score += -15.0 * min_delivery_dist

    if hunter_positions:
        hunter_dists = []
        for hp in hunter_positions:
            d = bfs_distance(layout, hp, drone_pos, hunter_restricted=True)
            if d != float("inf"):
                hunter_dists.append(d)

        if hunter_dists:
            min_hunter_dist = min(hunter_dists)
            capped = min(min_hunter_dist, 6)
            score += 20.0 * capped

    if pending_deliveries and hunter_positions:
        for ep in pending_deliveries:
            drone_to_ep = bfs_distance(layout, drone_pos, ep, hunter_restricted=False)
            if drone_to_ep == float("inf"):
                continue
            min_hunter_to_ep = min(
                bfs_distance(layout, hp, ep, hunter_restricted=True)
                for hp in hunter_positions
            )
            if drone_to_ep < min_hunter_to_ep:
                score += 30.0
    return float(score)

    # -----------------------------------------------------------------------
    # PRIMERA VERSION AUTONOMA
    #
    # Idea inicial: solo usar distancia Manhattan al punto de entrega más cercano
    # y distancia Manhattan al cazador más cercano, con pesos fijos iguales.
    #
    # Problemas:
    #   1. La distancia Manhattan ignora paredes y obstáculos: el dron puede ir hacia
    #      una entrega que en realidad requiere un camino mucho más largo (BFS).
    #   2. Los pesos iguales hacen que el dron no priorice sobrevivir cuando el cazador
    #      está muy cerca: huye y entrega con la misma urgencia, causando bucles.
    #   3. No se penaliza el número de entregas pendientes, por lo que el dron no
    #      tiene incentivo para completar la misión y puede quedarse oscilando.
    #   4. No se trata el caso de cazadores inalcanzables (distancia infinita), lo que
    #      podría restar puntos incorrectamente por cazadores bloqueados por montañas.
    # -----------------------------------------------------------------------
    # from algorithms.utils import manhattan_distance
    #
    # if state.is_win():
    #     return 1000.0
    # if state.is_lose():
    #     return -1000.0
    #
    # drone_pos = state.get_drone_position()
    # hunter_positions = state.get_hunter_positions()
    # pending_deliveries = state.get_pending_deliveries()
    #
    # if drone_pos is None:
    #     return 0.0
    #
    # # Distancia Manhattan al punto de entrega más cercano (ERROR: ignora paredes)
    # if pending_deliveries:
    #     min_ep_dist = min(manhattan_distance(drone_pos, ep) for ep in pending_deliveries)
    # else:
    #     min_ep_dist = 0
    #
    # # Distancia Manhattan al cazador más cercano (ERROR: ignora que cazadores no pasan por niebla/montaña)
    # if hunter_positions:
    #     min_hunter_dist = min(manhattan_distance(drone_pos, hp) for hp in hunter_positions)
    # else:
    #     min_hunter_dist = 999
    #
    # # Pesos iguales: entrega y seguridad tienen la misma importancia (ERROR: debería priorizar supervivencia)
    # score = -10.0 * min_ep_dist + 10.0 * min_hunter_dist
    # return score
    #
    # -----------------------------------------------------------------------
    # PROMPTS USADOS PARA LLEGAR A LA VERSION FINAL
    # -----------------------------------------------------------------------
    # Prompt 1:
    # "Mi evaluation_function solo usa distancia Manhattan y el agente se queda en bucles
    #  o va hacia entregas inalcanzables. ¿Cómo mejoro la función para que use distancias
    #  reales (BFS) y balancee correctamente supervivencia vs progreso de entrega?"
    #
    # Lo que se aprendió/corrigió:
    #   - Reemplazar manhattan_distance por bfs_distance(layout, ..., hunter_restricted=False)
    #     para el dron, y bfs_distance(..., hunter_restricted=True) para los cazadores,
    #     que refleja sus restricciones reales de movimiento.
    #   - Agregar penalización por número de entregas pendientes (-50 por entrega) para
    #     que el dron tenga incentivo constante de completar la misión.
    #
    # Prompt 2:
    # "El dron se paraliza cerca de un cazador aunque podría hacer la entrega de forma
    #  segura. ¿Cómo añado un factor de urgencia que lo motive a comprometerse con una
    #  entrega cuando puede llegar antes que el cazador?"
    #
    # Lo que se aprendió/corrigió:
    #   - Se añadió el factor de urgencia: comparar bfs_distance(dron → entrega) con
    #     min bfs_distance(cazador → entrega). Si el dron llega primero, sumar +30.
    #   - Se aplicó un cap en 6 a la distancia del cazador para evitar que domine
    #     demasiado en mapas grandes cuando el cazador está muy lejos.
    # -----------------------------------------------------------------------
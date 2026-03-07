from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)
        var = csp.get_unassigned_variables(assignment)[0]
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                result = backtrack(assignment)
                if result is not None:
                    return result
                csp.unassign(var, assignment)
        return None

    return backtrack({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    def restaurar_dominios(valores_eliminados: dict[str, list[str]]) -> None:
        for variable, valores in valores_eliminados.items():
            csp.domains[variable].extend(valores)

    def forward_check(
        variable_asignada: str, asignacion: dict[str, str]
    ) -> tuple[bool, dict[str, list[str]]]:
        valores_eliminados: dict[str, list[str]] = {}

        for vecina in csp.get_neighbors(variable_asignada):
            if vecina in asignacion:
                continue

            eliminados_vecina: list[str] = []

            for valor in list(csp.domains[vecina]):
                if not csp.is_consistent(vecina, valor, asignacion):
                    csp.domains[vecina].remove(valor)
                    eliminados_vecina.append(valor)

            if eliminados_vecina:
                valores_eliminados[vecina] = eliminados_vecina

            if len(csp.domains[vecina]) == 0:
                return False, valores_eliminados

        return True, valores_eliminados

    def backtrack(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return dict(asignacion)

        variable = csp.get_unassigned_variables(asignacion)[0]

        for valor in list(csp.domains[variable]):
            if csp.is_consistent(variable, valor, asignacion):
                csp.assign(variable, valor, asignacion)

                se_puede, eliminados = forward_check(variable, asignacion)
                if se_puede:
                    resultado = backtrack(asignacion)
                    if resultado is not None:
                        return resultado

                restaurar_dominios(eliminados)
                csp.unassign(variable, asignacion)

        return None

    return backtrack({})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    def revisar(xi: str, xj: str) -> bool:
        cambio = False

        for valor_xi in list(csp.domains[xi]):
            hay_soporte = False

            for valor_xj in csp.domains[xj]:
                asignacion_temporal = {xi: valor_xi}
                if csp.is_consistent(xj, valor_xj, asignacion_temporal):
                    hay_soporte = True
                    break

            if not hay_soporte:
                csp.domains[xi].remove(valor_xi)
                cambio = True

        return cambio

    def ac3(cola: deque[tuple[str, str]]) -> bool:
        while cola:
            xi, xj = cola.popleft()

            if revisar(xi, xj):
                if len(csp.domains[xi]) == 0:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        cola.append((xk, xi))

        return True

    cola_inicial = deque()
    for xi in csp.variables:
        for xj in csp.get_neighbors(xi):
            cola_inicial.append((xi, xj))

    copia_inicial = {var: list(csp.domains[var]) for var in csp.variables}
    if not ac3(cola_inicial):
        csp.domains = copia_inicial
        return None

    def backtrack(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return dict(asignacion)

        variable = csp.get_unassigned_variables(asignacion)[0]

        for valor in list(csp.domains[variable]):
            if csp.is_consistent(variable, valor, asignacion):
                copia_dominios = {var: list(csp.domains[var]) for var in csp.variables}

                csp.assign(variable, valor, asignacion)
                csp.domains[variable] = [valor]

                cola = deque()
                for vecina in csp.get_neighbors(variable):
                    cola.append((vecina, variable))

                if ac3(cola):
                    resultado = backtrack(asignacion)
                    if resultado is not None:
                        return resultado

                csp.domains = copia_dominios
                csp.unassign(variable, asignacion)

        return None

    return backtrack({})


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    
    def revisar(xi: str, xj: str) -> bool:
        cambio = False

        for valor_xi in list(csp.domains[xi]):
            hay_soporte = False

            for valor_xj in csp.domains[xj]:
                asignacion_temporal = {xi: valor_xi}
                if csp.is_consistent(xj, valor_xj, asignacion_temporal):
                    hay_soporte = True
                    break

            if not hay_soporte:
                csp.domains[xi].remove(valor_xi)
                cambio = True

        return cambio

    def ac3(cola: deque[tuple[str, str]]) -> bool:
        while cola:
            xi, xj = cola.popleft()

            if revisar(xi, xj):
                if len(csp.domains[xi]) == 0:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        cola.append((xk, xi))

        return True

    cola_inicial = deque()
    for xi in csp.variables:
        for xj in csp.get_neighbors(xi):
            cola_inicial.append((xi, xj))

    copia_inicial = {var: list(csp.domains[var]) for var in csp.variables}
    if not ac3(cola_inicial):
        csp.domains = copia_inicial
        return None

    def backtrack(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return dict(asignacion)

        variable = csp.get_unassigned_variables(asignacion)[0]

        for valor in list(csp.domains[variable]):
            if csp.is_consistent(variable, valor, asignacion):
                copia_dominios = {var: list(csp.domains[var]) for var in csp.variables}

                csp.assign(variable, valor, asignacion)
                csp.domains[variable] = [valor]

                cola = deque()
                for vecina in csp.get_neighbors(variable):
                    cola.append((vecina, variable))

                if ac3(cola):
                    resultado = backtrack(asignacion)
                    if resultado is not None:
                        return resultado

                csp.domains = copia_dominios
                csp.unassign(variable, asignacion)

        return None

    return backtrack({})


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking con MRV + LCV + Forward Checking.
    """

    def escoger_variable(asignacion: dict[str, str]) -> str:
        no_asignadas = csp.get_unassigned_variables(asignacion)

        mejor_variable = no_asignadas[0]
        mejor_tamano = len(csp.domains[mejor_variable])

        for variable in no_asignadas[1:]:
            tamano = len(csp.domains[variable])
            if tamano < mejor_tamano:
                mejor_variable = variable
                mejor_tamano = tamano

        return mejor_variable

    def ordenar_valores(variable: str, asignacion: dict[str, str]) -> list[str]:
        return sorted(
            csp.domains[variable],
            key=lambda valor: csp.get_num_conflicts(variable, valor, asignacion),
        )

    def restaurar_dominios(valores_eliminados: dict[str, list[str]]) -> None:
        for variable, valores in valores_eliminados.items():
            csp.domains[variable].extend(valores)

    def forward_check(
        variable_asignada: str, asignacion: dict[str, str]
    ) -> tuple[bool, dict[str, list[str]]]:
        valores_eliminados: dict[str, list[str]] = {}

        for vecina in csp.get_neighbors(variable_asignada):
            if vecina in asignacion:
                continue

            eliminados_vecina: list[str] = []
            for valor in list(csp.domains[vecina]):
                if not csp.is_consistent(vecina, valor, asignacion):
                    csp.domains[vecina].remove(valor)
                    eliminados_vecina.append(valor)

            if eliminados_vecina:
                valores_eliminados[vecina] = eliminados_vecina

            if len(csp.domains[vecina]) == 0:
                return False, valores_eliminados

        return True, valores_eliminados

    def backtrack(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return dict(asignacion)

        variable = escoger_variable(asignacion)

        for valor in ordenar_valores(variable, asignacion):
            if csp.is_consistent(variable, valor, asignacion):
                csp.assign(variable, valor, asignacion)

                se_puede, eliminados = forward_check(variable, asignacion)
                if se_puede:
                    resultado = backtrack(asignacion)
                    if resultado is not None:
                        return resultado

                restaurar_dominios(eliminados)
                csp.unassign(variable, asignacion)

        return None

    return backtrack({})

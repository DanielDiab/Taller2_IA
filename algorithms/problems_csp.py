from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.utils import dijkstra, manhattan_distance

if TYPE_CHECKING:
    from world.layout import DroneLayout
    from world.runner import DeliveryDict, DroneDict


class DroneAssignmentCSP:
    """
    Models assigning n drones to m delivery points as a CSP.

    Variables: A1, A2, ..., Am for each delivery point.
    Domains: D(Ai) = {drone1, drone2, ..., dronen}
    Constraints:
      1. Capacity: total weight of deliveries assigned to a drone <= its capacity.
      2. Range: drone's total route distance <= its battery.
      3. Time windows: drone must arrive within [t_early, t_late] for each delivery.
      4. Sequential: a drone handles multiple deliveries one at a time.
    """

    def __init__(
        self,
        layout: DroneLayout | None,
        drones: list[DroneDict],
        delivery_points: list[DeliveryDict],
    ) -> None:
        self.layout = layout
        self.drones: dict[str, DroneDict] = {d["id"]: d for d in drones}
        self.delivery_points: dict[str, DeliveryDict] = {
            dp["id"]: dp for dp in delivery_points
        }
        self.drone_ids: list[str] = [d["id"] for d in drones]

        self.variables: list[str] = [dp["id"] for dp in delivery_points]
        self.domains: dict[str, list[str]] = {
            var: list(self.drone_ids) for var in self.variables
        }

        self.var_to_delivery: dict[str, DeliveryDict] = {
            dp["id"]: dp for dp in delivery_points
        }

        self._distance_cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        self._precompute_distances()

        self._var_to_index: dict[str, int] = {
            var: i for i, var in enumerate(self.variables)
        }

    def _precompute_distances(self) -> None:
        """
        Precompute shortest-path distances between all relevant points.
        """
        all_positions: list[tuple[int, int]] = []
        for d in self.drones.values():
            all_positions.append(d["position"])
        for dp in self.delivery_points.values():
            all_positions.append(dp["position"])

        unique_positions = list(set(all_positions))
        for i, p1 in enumerate(unique_positions):
            for j, p2 in enumerate(unique_positions):
                if i < j:
                    if self.layout is not None:
                        dist, _ = dijkstra(self.layout, p1, p2)
                    else:
                        dist = float(manhattan_distance(p1, p2))
                    self._distance_cache[(p1, p2)] = dist
                    self._distance_cache[(p2, p1)] = dist
            self._distance_cache[(p1, p1)] = 0.0

    def _get_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
        """
        Get cached distance between two positions.
        """
        if (pos1, pos2) in self._distance_cache:
            return self._distance_cache[(pos1, pos2)]
        if self.layout is not None:
            dist, _ = dijkstra(self.layout, pos1, pos2)
        else:
            dist = float(manhattan_distance(pos1, pos2))
        self._distance_cache[(pos1, pos2)] = dist
        self._distance_cache[(pos2, pos1)] = dist
        return dist

    def _get_drone_deliveries(
        self,
        drone_id: str,
        assignment: dict[str, str],
        extra_var: str | None = None,
        extra_val: str | None = None,
    ) -> list[str]:
        """
        Get list of delivery point IDs assigned to a drone.
        """
        deliveries: list[str] = []
        for var, val in assignment.items():
            if val == drone_id:
                deliveries.append(var)
        if (
            extra_var is not None
            and extra_val == drone_id
            and extra_var not in assignment
        ):
            deliveries.append(extra_var)
        return deliveries

    def _compute_drone_total_weight(
        self,
        drone_id: str,
        assignment: dict[str, str],
        extra_var: str | None = None,
        extra_val: str | None = None,
    ) -> int:
        """
        Compute total weight of deliveries assigned to a drone.
        """
        deliveries = self._get_drone_deliveries(
            drone_id, assignment, extra_var, extra_val
        )
        total = 0
        for dp_id in deliveries:
            total += self.var_to_delivery[dp_id]["weight"]
        return total

    def _compute_drone_route_cost(
        self,
        drone_id: str,
        assignment: dict[str, str],
        extra_var: str | None = None,
        extra_val: str | None = None,
    ) -> float:
        """
        Compute total route distance for a drone visiting all its assigned
        delivery points sequentially, starting from its home position and
        returning after each delivery.
        """
        deliveries = self._get_drone_deliveries(
            drone_id, assignment, extra_var, extra_val
        )
        if not deliveries:
            return 0.0

        drone = self.drones[drone_id]
        drone_pos: tuple[int, int] = drone["position"]
        total_cost = 0.0

        # Sequential: drone goes from home to each delivery and back
        for dp_id in deliveries:
            dp_pos: tuple[int, int] = self.var_to_delivery[dp_id]["position"]
            total_cost += self._get_distance(drone_pos, dp_pos)
            total_cost += self._get_distance(dp_pos, drone_pos)

        return total_cost

    def _check_time_window(
        self, drone_id: str, var: str, assignment: dict[str, str]
    ) -> bool:
        """
        Check if the drone can reach the delivery point within its time window.
        Assumes sequential delivery: compute arrival time based on previous deliveries.
        """
        dp = self.var_to_delivery[var]
        if "time_window" not in dp:
            return True

        t_early, t_late = dp["time_window"]
        drone = self.drones[drone_id]
        drone_pos: tuple[int, int] = drone["position"]
        speed: float = drone.get("speed", 1.0)

        # Compute time to reach this delivery after completing all prior ones
        travel_time = 0.0
        deliveries_before = self._get_drone_deliveries(drone_id, assignment)
        for prev_dp_id in deliveries_before:
            prev_pos: tuple[int, int] = self.var_to_delivery[prev_dp_id]["position"]
            travel_time += self._get_distance(drone_pos, prev_pos) / speed
            travel_time += self._get_distance(prev_pos, drone_pos) / speed

        dp_pos: tuple[int, int] = dp["position"]
        travel_time += self._get_distance(drone_pos, dp_pos) / speed

        return t_early <= travel_time <= t_late

    def is_consistent(self, var: str, value: str, assignment: dict[str, str]) -> bool:
        """
        Check if assigning drone 'value' to delivery point 'var' is consistent.

        Checks:
          1. Capacity: total weight for this drone <= capacity.
          2. Range: total route cost for this drone <= battery.
          3. Time window: drone arrives within the delivery's time window.
        """
        drone_id = value
        drone = self.drones[drone_id]

        # 1. Capacity constraint
        total_weight = self._compute_drone_total_weight(
            drone_id, assignment, var, value
        )
        if total_weight > drone["capacity"]:
            return False

        # 2. Range constraint
        route_cost = self._compute_drone_route_cost(drone_id, assignment, var, value)
        if route_cost > drone["battery"]:
            return False

        # 3. Time window constraint
        if not self._check_time_window(drone_id, var, assignment):
            return False

        return True

    def assign(self, var: str, value: str, assignment: dict[str, str]) -> None:
        """
        Assign drone value to delivery point var.
        """
        assignment[var] = value

    def unassign(self, var: str, assignment: dict[str, str]) -> None:
        """
        Remove var from assignment.
        """
        if var in assignment:
            del assignment[var]

    def get_neighbors(self, var: str) -> list[str]:
        """
        Get variables that share a constraint with var.
        All delivery points share capacity, range, and sequential constraints
        through any drone they might be co-assigned to.
        """
        return [v for v in self.variables if v != var]

    def is_complete(self, assignment: dict[str, str]) -> bool:
        """
        Check if all delivery points are assigned.
        """
        return len(assignment) == len(self.variables)

    def get_unassigned_variables(self, assignment: dict[str, str]) -> list[str]:
        """
        Return list of unassigned delivery point variables.
        """
        return [v for v in self.variables if v not in assignment]

    def get_num_conflicts(
        self, var: str, value: str, assignment: dict[str, str]
    ) -> int:
        """
        Count how many values in neighboring variables' domains would be
        eliminated by assigning var=value (drone to delivery point).
        """
        conflicts = 0
        drone_id = value
        drone = self.drones[drone_id]
        temp_assignment = dict(assignment)
        temp_assignment[var] = value

        for neighbor in self.get_neighbors(var):
            if neighbor in assignment:
                continue
            for domain_val in self.domains[neighbor]:
                if domain_val != drone_id:
                    continue
                # Check if assigning the same drone to neighbor would violate constraints
                total_weight = self._compute_drone_total_weight(
                    drone_id, temp_assignment, neighbor, drone_id
                )
                if total_weight > drone["capacity"]:
                    conflicts += 1
                    continue

                route_cost = self._compute_drone_route_cost(
                    drone_id, temp_assignment, neighbor, drone_id
                )
                if route_cost > drone["battery"]:
                    conflicts += 1
                    continue

        return conflicts

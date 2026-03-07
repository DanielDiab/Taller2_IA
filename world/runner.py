from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict
import algorithms.adversarial as adversarial_module
import algorithms.csp as csp_module
from algorithms.problems_csp import DroneAssignmentCSP
from algorithms.utils import dijkstra
from world.game import Agent
from world.rules import GameRules, HunterAgent, MixedHunterAgent, RandomHunterAgent
from view.display import AdversarialDisplay, CspDisplay

if TYPE_CHECKING:
    from world.layout import DroneLayout


CSP_ALGORITHM_MAP: dict[str, str] = {
    "backtracking": "backtracking_search",
    "backtracking_fc": "backtracking_fc",
    "backtracking_ac3": "backtracking_ac3",
    "backtracking_mrv_lcv": "backtracking_mrv_lcv",
}


class DroneDict(TypedDict):
    id: str
    capacity: int
    battery: int
    position: tuple[int, int]
    speed: float


class DeliveryDict(TypedDict):
    id: str
    position: tuple[int, int]
    weight: int
    time_window: tuple[int, int]


class DroneState(TypedDict):
    position: tuple[int, int]
    base: tuple[int, int]
    target: str | None
    status: str
    path: list[tuple[int, int]]
    path_index: int
    delivery_queue: list[DeliveryDict]
    delivery_index: int
    returning: bool
    battery: int
    max_battery: int


class DeliveryStatus(TypedDict):
    position: tuple[int, int]
    status: str
    time_window: tuple[int, int]


def run_csp_mode(layout: "DroneLayout", display: CspDisplay, algorithm: str) -> None:
    """
    Run CSP planning mode with grid-based simulation visualization.
    """
    fn_name = CSP_ALGORITHM_MAP.get(algorithm, algorithm)

    print("=" * 60)
    print("DRONE ASSIGNMENT CSP")
    print("=" * 60)

    bases = layout.bases if layout.bases else [layout.agent_positions[0]]
    drones: list[DroneDict] = []
    for i, pos in enumerate(bases):
        lp = layout.drone_params.get(i + 1, {})
        if "capacity" not in lp:
            raise Exception(
                f"Drone {i + 1} missing 'capacity' in layout params. "
                "Add 'drone:<index>:capacity=<int>,battery=<int>' after '---' in the layout file."
            )
        if "battery" not in lp:
            raise Exception(
                f"Drone {i + 1} missing 'battery' in layout params. "
                "Add 'drone:<index>:capacity=<int>,battery=<int>' after '---' in the layout file."
            )
        drones.append(
            DroneDict(
                id=f"drone{i + 1}",
                capacity=lp["capacity"],
                battery=lp["battery"],
                position=pos,
                speed=1.0,
            )
        )

    delivery_points: list[DeliveryDict] = []
    for i, pos in enumerate(layout.delivery_positions):
        lp = layout.delivery_params.get(i + 1, {})
        if "weight" not in lp:
            raise Exception(
                f"Delivery E{i + 1} missing 'weight' in layout params. "
                "Add 'delivery:<index>:weight=<int>,window=<int>-<int>' after '---' in the layout file."
            )
        if "time_window" not in lp:
            raise Exception(
                f"Delivery E{i + 1} missing 'window' in layout params. "
                "Add 'delivery:<index>:weight=<int>,window=<int>-<int>' after '---' in the layout file."
            )
        delivery_points.append(
            DeliveryDict(
                id=f"E{i + 1}",
                position=pos,
                weight=lp["weight"],
                time_window=lp["time_window"],
            )
        )

    print(f"  Drones: {len(drones)}")
    for d in drones:
        print(
            f"    {d['id']}: pos={d['position']}, capacity={d['capacity']}kg, battery={d['battery']}"
        )
    print(f"  Deliveries: {len(delivery_points)}")
    for dp in delivery_points:
        print(
            f"    {dp['id']}: pos={dp['position']}, weight={dp['weight']}kg, window={dp['time_window']}"
        )
    print(f"  Algorithm: {algorithm}")
    print("-" * 60)

    csp = DroneAssignmentCSP(layout, drones, delivery_points)

    if not hasattr(csp_module, fn_name):
        print(f"Error: Algorithm '{algorithm}' not found in algorithms/csp.py")
        return

    algorithm_fn = getattr(csp_module, fn_name)

    start_time = time.time()
    result = algorithm_fn(csp)
    elapsed = time.time() - start_time

    if result is not None:
        print("\n\u2705 Solution found!")
        print("-" * 60)
        _simulate_csp_solution(layout, display, result, csp, drones, delivery_points)
    else:
        print("\n\u274c No solution found.")
        print("-" * 60)

    print(f"\n  Time: {elapsed:.4f}s")
    print("=" * 60)


def _simulate_csp_solution(
    layout: DroneLayout,
    display: CspDisplay,
    result: dict[str, str],
    csp: DroneAssignmentCSP,
    drones_list: list[DroneDict],
    delivery_points_list: list[DeliveryDict],
) -> None:
    """
    Simulate the CSP solution step by step on the grid.
    """
    # Group deliveries by drone, sorted by time_window start
    drone_deliveries: dict[str, list[DeliveryDict]] = {}
    for var, drone_id in result.items():
        dp = csp.var_to_delivery[var]
        if drone_id not in drone_deliveries:
            drone_deliveries[drone_id] = []
        drone_deliveries[drone_id].append(dp)

    for drone_id in drone_deliveries:
        drone_deliveries[drone_id].sort(
            key=lambda dp: dp.get("time_window", (0, 9999))[0]
        )

    # Initialize drone states
    drone_states: dict[str, DroneState] = {}
    for d in drones_list:
        drone_id = d["id"]
        deliveries = drone_deliveries.get(drone_id, [])
        first_target = deliveries[0] if deliveries else None
        path: list[tuple[int, int]] = []
        if first_target:
            _, path = dijkstra(layout, d["position"], first_target["position"])
        drone_states[drone_id] = DroneState(
            position=d["position"],
            base=d["position"],
            target=first_target["id"] if first_target else None,
            status="en_route" if first_target else "idle",
            path=path,
            path_index=1,  # Skip start position (already there)
            delivery_queue=deliveries,
            delivery_index=0,
            returning=False,
            battery=d["battery"],
            max_battery=d["battery"],
        )

    # Initialize delivery statuses
    delivery_statuses: dict[str, DeliveryStatus] = {}
    for dp in delivery_points_list:
        delivery_statuses[dp["id"]] = DeliveryStatus(
            position=dp["position"], status="pending", time_window=dp["time_window"]
        )

    current_time = 0
    display.initialize(layout, drone_states, delivery_statuses, current_time)

    max_time = 500
    while current_time < max_time:
        current_time += 1

        for drone_id, ds in drone_states.items():
            if ds["status"] == "idle":
                continue

            if ds["status"] == "waiting":
                target_dp = ds["delivery_queue"][ds["delivery_index"]]
                tw = target_dp.get("time_window", (0, 9999))
                if current_time >= tw[0]:
                    delivery_statuses[target_dp["id"]]["status"] = "delivered"
                    ds["delivery_index"] += 1
                    ds["returning"] = True
                    ds["status"] = "returning"
                    _, ds["path"] = dijkstra(layout, ds["position"], ds["base"])
                    ds["path_index"] = 1  # Skip start position
                continue

            if ds["path_index"] < len(ds["path"]):
                new_pos = ds["path"][ds["path_index"]]
                cost = layout.get_terrain_cost(new_pos[0], new_pos[1])
                ds["position"] = new_pos
                ds["battery"] = max(0, ds["battery"] - cost)
                ds["path_index"] += 1

            if ds["path_index"] >= len(ds["path"]):
                if ds["returning"]:
                    ds["returning"] = False
                    if ds["delivery_index"] < len(ds["delivery_queue"]):
                        next_dp = ds["delivery_queue"][ds["delivery_index"]]
                        ds["target"] = next_dp["id"]
                        ds["status"] = "en_route"
                        _, ds["path"] = dijkstra(
                            layout, ds["position"], next_dp["position"]
                        )
                        ds["path_index"] = 1  # Skip start position
                        delivery_statuses[next_dp["id"]]["status"] = "in_progress"
                    else:
                        ds["target"] = None
                        ds["status"] = "idle"
                else:
                    target_dp = ds["delivery_queue"][ds["delivery_index"]]
                    tw = target_dp.get("time_window", (0, 9999))
                    if current_time >= tw[0]:
                        if current_time > tw[1]:
                            print(
                                f"  \u26a0 {drone_id}: delivery {target_dp['id']} is LATE "
                                f"(t={current_time}, window=[{tw[0]},{tw[1]}])"
                            )
                        delivery_statuses[target_dp["id"]]["status"] = "delivered"
                        ds["delivery_index"] += 1
                        ds["returning"] = True
                        ds["status"] = "returning"
                        _, ds["path"] = dijkstra(layout, ds["position"], ds["base"])
                        ds["path_index"] = 1  # Skip start position
                    else:
                        ds["status"] = "waiting"
                        delivery_statuses[target_dp["id"]]["status"] = "waiting"

        display.update(layout, drone_states, delivery_statuses, current_time)

        if all(ds["status"] == "idle" for ds in drone_states.values()):
            break

    display.finish()


def run_adversarial_mode(
    layout: DroneLayout,
    display: AdversarialDisplay,
    agent_type: str,
    depth: int,
    random_probability: float,
    num_games: int = 1,
) -> None:
    """
    Run adversarial game mode.
    """
    if not hasattr(adversarial_module, agent_type):
        raise Exception(
            f"Agent type '{agent_type}' not found in algorithms/adversarial.py"
        )

    AgentClass = getattr(adversarial_module, agent_type)
    drone_agent = AgentClass(depth=str(depth), prob=str(random_probability))

    num_hunters = len(layout.hunter_positions)
    hunter_agents: list[Agent] = []
    for i in range(num_hunters):
        if random_probability == 0.0:
            hunter_agents.append(HunterAgent(i + 1))
        elif random_probability == 1.0:
            hunter_agents.append(RandomHunterAgent(i + 1))
        else:
            hunter_agents.append(MixedHunterAgent(i + 1, random_probability))

    wins = 0
    losses = 0
    total_score = 0

    for i in range(num_games):
        game = GameRules.new_game(
            layout,
            drone_agent,
            hunter_agents,
            display,
            quiet=(num_games > 1),
        )
        game.run()

        if game.state.is_win():
            wins += 1
        elif game.state.is_lose():
            losses += 1
        total_score += game.state.get_score()

    if num_games > 1:
        print("\n" + "=" * 60)
        print("RESULTS (%d games)" % num_games)
        print("=" * 60)
        print(f"  Wins:   {wins}/{num_games}")
        print(f"  Losses: {losses}/{num_games}")
        print(f"  Avg Score: {total_score / num_games:.1f}")
        print("=" * 60)

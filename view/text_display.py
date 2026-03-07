import time

from world.game_state import GameState
from world.layout import DroneLayout
from world.runner import DeliveryStatus, DroneState
from view.display import AdversarialDisplay, CspDisplay

DRAW_EVERY = 1
sleep_time: float = 0
DISPLAY_MOVES = False
QUIET = False


class CspNullGraphics(CspDisplay):
    """
    Null graphics for quiet mode (no visual output). Used when running with -q flag.
    """

    def initialize(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        pass

    def update(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        pass

    def finish(self):
        pass


class AdversarialNullGraphics(AdversarialDisplay):
    """
    Null graphics for quiet mode (no visual output). Used when running with -q flag.
    """

    def initialize(self, state: GameState):
        pass

    def update(self, state: GameState):
        pass

    def finish(self):
        pass


class TextAdversarialGraphics(AdversarialDisplay):
    """
    Text-based graphics for the adversarial game domain.
    """

    def __init__(self, speed: float | None = None):
        if speed is not None:
            global sleep_time
            sleep_time = speed

    def initialize(self, state: GameState):
        """
        Initialize the display with the initial game state.
        """
        self.draw(state)
        self.pause()
        self.turn = 0
        self.agent_counter = 0

    def update(self, state: GameState):
        """
        Update the display to reflect the new game state after a move.
        """
        num_agents = state.get_num_agents()
        self.agent_counter = (self.agent_counter + 1) % num_agents
        if self.agent_counter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                drone_pos = state.get_drone_position()
                hunter_pos = state.get_hunter_positions()
                score = state.get_score()
                pending = len(state.get_pending_deliveries())
                print(
                    "%4d) Drone: %-8s" % (self.turn, str(drone_pos)),
                    "| Hunters: %s" % str(hunter_pos),
                    "| Score: %-5d" % score,
                    "| Pending: %d" % pending,
                )
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state.is_win() or state.is_lose():
            self.draw(state)

    def pause(self):
        """
        Pause for a moment to allow the user to see the current state.
        """
        if sleep_time < 0:
            input("Press Enter to continue...")
        else:
            time.sleep(sleep_time)

    def draw(self, state: GameState):
        """
        Draw the current game state.
        """
        _draw_board(state)

    def finish(self):
        pass


class CspGraphics(CspDisplay):
    """
    Text-based graphics for CSP drone delivery simulation.
    Shows the grid map with multiple drones moving toward delivery points.
    """

    def __init__(self, speed: float | None = None):
        if speed is not None:
            global sleep_time
            sleep_time = speed

    def initialize(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        """
        Initialize the CSP display.

        Args:
            layout: The DroneLayout object
            drone_states: dict of drone_id -> {"position": (x,y), "target": delivery_id or None, "status": str}
            delivery_statuses: dict of delivery_id -> {"position": (x,y), "status": str, "time_window": (t_early, t_late)}
                status can be: "pending", "in_progress", "waiting", "delivered"
            current_time: current simulation time step
        """
        self.layout = layout
        self._draw_csp_board(layout, drone_states, delivery_statuses, current_time)
        self._pause()

    def update(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        """
        Update the CSP display.
        """
        self._draw_csp_board(layout, drone_states, delivery_statuses, current_time)
        self._pause()

    def finish(self):
        pass

    def _pause(self):
        if sleep_time < 0:
            input("Press Enter to continue...")
        else:
            time.sleep(sleep_time)

    def _draw_csp_board(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        """
        Render the CSP simulation board as text.
        """
        width = layout.width
        height = layout.height
        walls = layout.walls

        # Build position lookup maps
        drone_positions: dict[tuple[int, int], str] = {}  # (x,y) -> drone_id
        for drone_id, ds in drone_states.items():
            pos = ds["position"]
            drone_positions[pos] = drone_id

        delivery_positions: dict[
            tuple[int, int], tuple[str, str]
        ] = {}  # (x,y) -> (delivery_id, status)
        for dp_id, ds in delivery_statuses.items():
            pos = ds["position"]
            delivery_positions[pos] = (dp_id, ds["status"])

        base_positions: set[tuple[int, int]] = (
            set(layout.bases) if layout.bases else set()
        )

        # Build grid top-to-bottom
        print(f"\n{'=' * 40}")
        print(f"  Time: {current_time}")
        print(f"{'=' * 40}")

        lines: list[str] = []
        for y in range(height - 1, -1, -1):
            row: list[str] = []
            for x in range(width):
                pos = (x, y)
                if pos in drone_positions:
                    # Show drone number (extract digit from drone_id like "drone1" -> "1")
                    drone_id = drone_positions[pos]
                    num = "".join(c for c in drone_id if c.isdigit()) or "D"
                    row.append(num[-1])  # Use last digit
                elif pos in delivery_positions:
                    dp_id, status = delivery_positions[pos]
                    if status == "delivered":
                        row.append("✓")
                    elif status == "waiting":
                        row.append("W")
                    else:
                        row.append("E")
                elif pos in base_positions:
                    row.append("B")
                elif walls[x][y]:
                    row.append("%")
                else:
                    terrain = layout.get_terrain(x, y)
                    if terrain in ("~", "^", "*"):
                        row.append(terrain)
                    else:
                        row.append(".")
            lines.append("".join(row))

        print("\n".join(lines))

        # Status info
        print("-" * 40)
        for drone_id, ds in sorted(drone_states.items()):
            target = ds.get("target", None)
            status = ds.get("status", "idle")
            pos = ds["position"]
            target_str = f"-> {target}" if target else "(idle)"
            battery = ds.get("battery", None)
            max_battery = ds.get("max_battery", None)
            if max_battery:
                pct = int(100 * battery / max_battery)
                bar_len = 10
                filled = int(bar_len * battery / max_battery)
                bar = "█" * filled + "░" * (bar_len - filled)
                bat_str = f"  🔋 [{bar}] {battery}/{max_battery} ({pct}%)"
            else:
                bat_str = ""
            print(f"  {drone_id}: {pos} {target_str} [{status}]{bat_str}")

        # Delivery statuses
        for dp_id, ds in sorted(delivery_statuses.items()):
            tw = ds.get("time_window", (0, 9999))
            status = ds["status"]
            available = (
                "✓ available" if tw[0] <= current_time <= tw[1] else "✗ not available"
            )
            if status == "delivered":
                available = "✓ delivered"
            print(
                f"  {dp_id}: window=[{tw[0]},{tw[1]}] | t={current_time} | {available} | {status}"
            )
        print()


def _draw_board(state: GameState):
    """
    Render the adversarial game board as a text grid showing walls, terrain, drone, hunters, and deliveries.
    """
    layout = state.get_layout()
    if layout is None:
        print(state)
        return

    width = layout.width
    height = layout.height
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = state.get_pending_deliveries()
    all_deliveries = set(layout.delivery_positions)
    walls = layout.walls

    # Build grid top-to-bottom
    lines: list[str] = []
    for y in range(height - 1, -1, -1):
        row: list[str] = []
        for x in range(width):
            pos = (x, y)
            if pos == drone_pos:
                row.append("D")
            elif pos in hunter_positions:
                row.append("C")
            elif pos in pending_deliveries:
                row.append("E")
            elif pos in all_deliveries:
                row.append("✓")
            elif walls[x][y]:
                row.append("%")
            else:
                terrain = layout.get_terrain(x, y)
                if terrain in ("~", "^", "*"):
                    row.append(terrain)
                else:
                    row.append(".")
        lines.append("".join(row))

    print("\n".join(lines))
    print(
        f"Score: {state.get_score()}  |  Pending: {len(pending_deliveries)}  |  "
        f"Drone: {drone_pos}  |  Hunters: {hunter_positions}"
    )

import os
from typing import Any, TypedDict

from world.game import Grid


class DroneParameters(TypedDict):
    capacity: int
    battery: int


class DeliveryParameters(TypedDict):
    weight: int
    time_window: tuple[int, int]


class DroneLayout:
    """
    A DroneLayout manages the static information about the drone delivery grid.
    """

    def __init__(self, layout_text: list[str], name: str = "") -> None:
        self.name: str = name
        # Separate grid from parameters (split on '---' separator)
        grid_lines: list[str] = []
        self.param_lines: list[str] = []
        in_params = False
        for line in layout_text:
            if line.strip() == "---":
                in_params = True
                continue
            if in_params:
                self.param_lines.append(line)
            else:
                grid_lines.append(line)

        self.width = len(grid_lines[0])
        self.height = len(grid_lines)
        self.walls = Grid(self.width, self.height, False)
        self.deliveries = Grid(self.width, self.height, False)

        self.agent_positions: list[tuple[int, int]] = []
        self.hunter_positions: list[tuple[int, int]] = []
        self.bases: list[tuple[int, int]] = []
        self.delivery_positions: list[tuple[int, int]] = []
        self._drone_positions: list[tuple[int, int]] = []
        self.terrain: dict[tuple[int, int], str] = {}
        self.process_layout_text(grid_lines)
        self.layout_text = grid_lines

        # Parse CSP parameters from layout file
        self.drone_params: dict[int, DroneParameters] = {}
        self.delivery_params: dict[int, DeliveryParameters] = {}
        self._parse_params()

    def get_terrain(self, x: int, y: int) -> str:
        """
        Get the terrain character at position (x, y).
        Returns the terrain character or '.' for normal floor.
        """
        return self.terrain.get((x, y), ".")

    def get_terrain_cost(self, x: int, y: int) -> int:
        """
        Get the movement cost for terrain at position (x, y).

        Terrain costs:
        - Normal floor (' ' or '.'): 1
        - Fog zone ('~'): 2
        - Mountain terrain ('^'): 3
        - Electric storm ('*'): 5
        """
        terrain_char = self.get_terrain(x, y)
        TERRAIN_COSTS: dict[str, int] = {
            ".": 1,  # Normal floor
            " ": 1,  # Empty space
            "~": 2,  # Fog zone
            "^": 3,  # Mountain terrain
            "*": 5,  # Electric storm
        }
        return TERRAIN_COSTS.get(terrain_char, 1)

    def __str__(self) -> str:
        return "\n".join(self.layout_text)

    def process_layout_text(self, layout_text: list[str]) -> None:
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the grid. Each character represents a different type of object:
         % - Wall/Obstacle (impassable)
         D - Drone (starting position)
         E - Delivery point (destination for medical supplies, e.g. E1, E2)
         C - Hunter (adversary agent, e.g. C1, C2)
         . - Normal floor (movement cost 1)
         ~ - Fog zone (movement cost 2)
         ^ - Mountain terrain (movement cost 3)
         * - Electric storm (movement cost 5)

        Other characters are treated as normal floor (cost 1).
        """
        max_y = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layout_char = layout_text[max_y - y][x]
                self.process_layout_char(x, y, layout_char)

        # Ensure drone (D) is first (index 0), hunters (C) follow
        self._drone_positions.sort()
        self.hunter_positions.sort()
        self.agent_positions = self._drone_positions + self.hunter_positions

    def process_layout_char(self, x: int, y: int, layout_char: str) -> None:
        """
        Process a single layout character and update the appropriate data structure.
        """
        # Walls
        if layout_char == "%":
            self.walls[x][y] = True

        # Drone starting position
        elif layout_char == "D":
            self._drone_positions.append((x, y))
            self.bases.append((x, y))

        # Drone base position (CSP layouts)
        elif layout_char == "B":
            self.bases.append((x, y))

        # Hunter (adversary agent)
        elif layout_char == "C":
            self.hunter_positions.append((x, y))

        # Delivery point
        elif layout_char == "E":
            self.deliveries[x][y] = True
            self.delivery_positions.append((x, y))

        # Terrain types for variable movement costs
        elif layout_char in ("~", "^", "*"):
            self.terrain[(x, y)] = layout_char

        # All other characters (including ' ' and '.') are treated as normal floor

    def _parse_params(self) -> None:
        """
        Parse CSP parameters from layout file.
        Lines after '---' separator with format:
          drone:<index>:capacity=<int>,battery=<int>
          delivery:<index>:weight=<int>,window=<int>-<int>
        """
        for pline in self.param_lines:
            pline = pline.strip()
            if not pline:
                continue
            parts = pline.split(":")
            if len(parts) < 3:
                continue
            kind = parts[0].strip()
            idx = int(parts[1].strip())
            kvs = parts[2].strip()
            params: dict[str, Any] = {}
            for kv in kvs.split(","):
                kv = kv.strip()
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "window":
                    lo, hi = v.split("-")
                    params[k] = (int(lo), int(hi))
                else:
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
            if kind == "drone":
                capacity = params.get("capacity", 0)
                battery = params.get("battery", 0)

                assert isinstance(capacity, int)
                assert isinstance(battery, int)

                self.drone_params[idx] = DroneParameters(
                    capacity=capacity,
                    battery=battery,
                )
            elif kind == "delivery":
                weight = params.get("weight", 0)
                time_window_raw = params.get("window", (0, 9999))
                time_window: tuple[int, int] = tuple(time_window_raw)  # type: ignore

                assert isinstance(weight, int)
                assert isinstance(time_window, tuple)
                assert len(time_window) == 2
                assert all(isinstance(x, int) for x in time_window)

                self.delivery_params[idx] = DeliveryParameters(
                    weight=weight,
                    time_window=time_window,
                )


def get_layout(name: str) -> DroneLayout | None:
    """
    Load a layout file by name.

    Searches recursively inside the layouts/ directory for a matching .lay file.
    """
    filename = name if name.endswith(".lay") else name + ".lay"
    for root, _dirs, files in os.walk("layouts"):
        if filename in files:
            return try_to_load(os.path.join(root, filename), name)
    return None


def try_to_load(fullname: str, name: str = "") -> DroneLayout | None:
    """
    Attempt to load a layout from a file.
    Returns None if file doesn't exist.
    """
    if not os.path.exists(fullname):
        return None
    with open(fullname) as f:
        return DroneLayout([line.strip() for line in f], name=name)

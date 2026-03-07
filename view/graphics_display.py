from view.graphics_utils import (
    begin_graphics,
    begin_graphics_scrollable,
    changeText,
    circle,
    edit,
    end_graphics,
    formatColor,
    line,
    refresh,
    remove_from_screen,
    sleep,
    square,
    text,
    wait_for_keys,
)
from world.game import Grid
from world.game_state import GameState
from world.layout import DroneLayout
from world.runner import DeliveryStatus, DroneState
from view.display import AdversarialDisplay, CspDisplay


DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 52
INFO_PANE_PADDING = 12
MAX_WINDOW_WIDTH = 1400
MAX_WINDOW_HEIGHT = 900
VIEWPORT_MAX_WIDTH = 1280
VIEWPORT_MAX_HEIGHT = 720

# Environment (jungle)
MA_BACKGROUND_COLOR = formatColor(0.18, 0.30, 0.15)  # Dark forest green
MA_GRID_LINE_COLOR = formatColor(0.15, 0.25, 0.12)  # Darker green grid

# Walls (dense vegetation / brown rocks)
MA_WALL_FILL = formatColor(0.30, 0.22, 0.14)  # Dark brown
MA_WALL_OUTLINE = formatColor(0.20, 0.14, 0.08)  # Darker outline

# Terrain (jungle hazards)
MA_FOG_BASE = formatColor(0.75, 0.78, 0.80)  # Light gray fog
MA_FOG_SWIRL = formatColor(0.88, 0.90, 0.92)  # White swirl
MA_FOG_TEXT = formatColor(0.3, 0.3, 0.3)  # Dark text

MA_MOUNTAIN_BASE = formatColor(0.50, 0.45, 0.38)  # Gray-brown rock
MA_MOUNTAIN_PEAK = formatColor(0.62, 0.58, 0.50)  # Lighter peaks
MA_MOUNTAIN_TEXT = formatColor(1.0, 1.0, 1.0)  # White text

MA_STORM_CORE = formatColor(0.55, 0.20, 0.70)  # Purple core
MA_STORM_GLOW = formatColor(0.90, 0.85, 0.20)  # Yellow lightning
MA_STORM_TEXT = formatColor(0.1, 0.1, 0.1)  # Dark text

# Drone (bright cyan/blue)
MA_DRONE_BODY = formatColor(0.10, 0.70, 0.90)  # Bright cyan
MA_DRONE_ACCENT = formatColor(0.20, 0.85, 1.0)  # Light cyan
MA_DRONE_ROTOR = formatColor(0.60, 0.60, 0.65)  # Metallic gray
MA_DRONE_OUTLINE = formatColor(0.05, 0.30, 0.45)  # Dark teal

# Hunters (red / dark red)
MA_HUNTER_BODY = formatColor(0.80, 0.15, 0.15)  # Bright red
MA_HUNTER_ACCENT = formatColor(0.55, 0.08, 0.08)  # Dark red
MA_HUNTER_OUTLINE = formatColor(0.35, 0.05, 0.05)  # Very dark red

# Delivery points
MA_DELIVERY_PENDING = formatColor(1.0, 0.85, 0.0)  # Bright gold
MA_DELIVERY_OUTLINE = formatColor(0.85, 0.65, 0.0)  # Darker gold
MA_DELIVERY_TEXT = formatColor(0.1, 0.1, 0.1)  # Dark text
MA_DELIVERY_DONE = formatColor(0.3, 0.8, 0.3)  # Green (completed)
MA_DELIVERY_DONE_OUTLINE = formatColor(0.2, 0.6, 0.2)  # Dark green

# Bases
MA_BASE_FILL = formatColor(0.25, 0.45, 0.75)  # Blue
MA_BASE_OUTLINE = formatColor(0.15, 0.30, 0.55)  # Darker blue

# Info Pane (multi-agent)
MA_SCORE_COLOR = formatColor(0.9, 0.9, 0.8)  # Light cream text
MA_TITLE_COLOR = formatColor(0.7, 0.85, 0.6)  # Soft green
MA_DELIVERY_COUNT_COLOR = formatColor(1.0, 0.85, 0.0)  # Gold

# CSP-specific colors
MA_DELIVERY_NOT_AVAILABLE = formatColor(0.85, 0.35, 0.15)  # Orange-red (not in window)
MA_DELIVERY_NOT_AVAILABLE_OUTLINE = formatColor(0.65, 0.25, 0.10)
MA_DELIVERY_WAITING = formatColor(0.90, 0.60, 0.10)  # Amber (waiting)
MA_DELIVERY_WAITING_OUTLINE = formatColor(0.70, 0.45, 0.05)

# Drone colors for CSP (multiple drones)
CSP_DRONE_COLORS = [
    (
        formatColor(0.10, 0.70, 0.90),
        formatColor(0.20, 0.85, 1.0),
        formatColor(0.05, 0.30, 0.45),
    ),  # Cyan
    (
        formatColor(0.30, 0.80, 0.30),
        formatColor(0.40, 0.95, 0.40),
        formatColor(0.15, 0.40, 0.15),
    ),  # Green
    (
        formatColor(0.90, 0.60, 0.10),
        formatColor(1.00, 0.75, 0.20),
        formatColor(0.45, 0.30, 0.05),
    ),  # Orange
    (
        formatColor(0.80, 0.20, 0.80),
        formatColor(0.95, 0.35, 0.95),
        formatColor(0.40, 0.10, 0.40),
    ),  # Purple
    (
        formatColor(0.90, 0.30, 0.30),
        formatColor(1.00, 0.45, 0.45),
        formatColor(0.45, 0.15, 0.15),
    ),  # Red
    (
        formatColor(0.20, 0.50, 0.90),
        formatColor(0.35, 0.65, 1.00),
        formatColor(0.10, 0.25, 0.45),
    ),  # Blue
]


class MultiAgentInfoPane:
    """
    Info pane for multi-agent (drone vs hunters) mode.
    """

    def __init__(self, layout: DroneLayout, grid_size: float, total_deliveries: int):
        self.grid_size = grid_size
        self.width = layout.width * grid_size
        self.base = (layout.height + 1) * grid_size
        self.total_deliveries = total_deliveries

        if self.width < 300:
            self.font_size = 14
            self.title_size = 10
        elif self.width < 500:
            self.font_size = 16
            self.title_size = 11
        else:
            self.font_size = 18
            self.title_size = 12

        self.draw_pane()

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert info pane coordinates to screen coordinates.
        """
        return (self.grid_size + x, self.base + y)

    def draw_pane(self):
        """
        Draw the info pane background and static text elements.
        """
        title_x = self.width // 2
        self.title_text = text(
            self.to_screen(title_x, 4),
            MA_TITLE_COLOR,
            "DRONE MISSION",
            "Arial",
            self.title_size,
            "bold",
            anchor="n",
        )

        self.score_text = text(
            self.to_screen(INFO_PANE_PADDING, 24),
            MA_SCORE_COLOR,
            "SCORE: 0",
            "Arial",
            self.font_size,
            "bold",
            anchor="nw",
        )

        self.delivery_text = text(
            self.to_screen(self.width - INFO_PANE_PADDING, 24),
            MA_DELIVERY_COUNT_COLOR,
            f"DELIVERIES: 0/{self.total_deliveries}",
            "Arial",
            self.font_size,
            "bold",
            anchor="ne",
        )

    def update_score(self, score: int):
        """
        Update the score display.
        """
        changeText(self.score_text, f"SCORE: {score}")

    def update_deliveries(self, completed: int, total: int):
        """
        Update the deliveries count display.
        """
        changeText(self.delivery_text, f"DELIVERIES: {completed}/{total}")


class VisualAdversarialGraphics(AdversarialDisplay):
    def __init__(
        self, zoom: float = 1.0, frame_time: float = 0.1, capture: bool = False
    ):
        self.zoom = zoom
        self.grid_size = DEFAULT_GRID_SIZE * zoom
        self.frame_time = frame_time
        self.capture = capture
        self._step_mode_message_shown = False

        self.grid_lines: list[int] = []
        self.terrain_tiles: list[int] = []
        self.terrain_labels: list[int] = []

        # Multi-agent state
        self._drone_images: list[int] = []
        self._hunter_images: list[list[int]] = []
        self._delivery_images: dict[tuple[int, int], list[int]] = {}
        self._base_images: list[list[int]] = []
        self._total_deliveries = 0
        self._pending_deliveries: set[tuple[int, int]] = set()

    def initialize(self, state: GameState):
        """
        Initialize display with mission state (multi-agent drone vs hunters).
        """
        self._initialize_multi_agent(state)

    def _initialize_multi_agent(self, state: GameState):
        """
        Initialize display for multi-agent (drone vs hunters) mode.
        """
        self.layout = state.get_layout()
        assert self.layout is not None

        self.width = self.layout.width
        self.height = self.layout.height

        self._pending_deliveries = set(state.get_pending_deliveries())
        self._total_deliveries = len(self.layout.delivery_positions)

        screen_width = 2 * self.grid_size + (self.width - 1) * self.grid_size
        screen_height = (
            2 * self.grid_size + (self.height - 1) * self.grid_size + INFO_PANE_HEIGHT
        )
        self._use_scroll = (
            screen_width > VIEWPORT_MAX_WIDTH or screen_height > VIEWPORT_MAX_HEIGHT
        )
        if self._use_scroll:
            self._content_width = int(screen_width)
            self._content_height = int(screen_height)
        else:
            if screen_width > MAX_WINDOW_WIDTH or screen_height > MAX_WINDOW_HEIGHT:
                scale = min(
                    MAX_WINDOW_WIDTH / screen_width, MAX_WINDOW_HEIGHT / screen_height
                )
                self.grid_size *= scale
                screen_width = 2 * self.grid_size + (self.width - 1) * self.grid_size
                screen_height = (
                    2 * self.grid_size
                    + (self.height - 1) * self.grid_size
                    + INFO_PANE_HEIGHT
                )
            self._content_width = int(screen_width)
            self._content_height = int(screen_height)

        self._make_window_multi_agent()
        self.info_pane = MultiAgentInfoPane(
            self.layout, self.grid_size, self._total_deliveries
        )

        self._draw_static_multi_agent(state)
        self._draw_agents_multi_agent(state)
        self.previous_state = state

    def finish(self):
        """
        Clean up graphics resources and close window.
        """
        end_graphics()

    def _make_window_multi_agent(self):
        """
        Create window with jungle theme for multi-agent mode.
        """
        if getattr(self, "_use_scroll", False):
            viewport_w = min(VIEWPORT_MAX_WIDTH, self._content_width)
            viewport_h = min(VIEWPORT_MAX_HEIGHT, self._content_height)
            begin_graphics_scrollable(
                viewport_w,
                viewport_h,
                self._content_width,
                self._content_height,
                MA_BACKGROUND_COLOR,
                "Drone Mission Control",
            )
        else:
            begin_graphics(
                self._content_width,
                self._content_height,
                MA_BACKGROUND_COLOR,
                "Drone Mission Control",
            )

    def to_screen(self, point: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates to screen coordinates.
        """
        x, y = point
        x = (x + 1) * self.grid_size
        y = (self.height - y) * self.grid_size
        return (x, y)

    def _draw_static_multi_agent(self, state: GameState):
        """
        Draw all static elements for multi-agent mode.
        """
        walls = state.get_walls()
        assert walls is not None
        assert self.layout is not None
        self._draw_background_grid_multi_agent()
        self._draw_terrain_multi_agent(state)
        self._draw_walls_multi_agent(walls)
        self._draw_bases(self.layout.bases)
        self._draw_delivery_points(state)
        refresh()

    def _draw_background_grid_multi_agent(self):
        """
        Draw subtle dark green grid lines.
        """
        for x in range(self.width + 1):
            x_screen = (x + 1) * self.grid_size
            line_obj = line(
                (x_screen, self.grid_size),
                (x_screen, (self.height + 1) * self.grid_size),
                MA_GRID_LINE_COLOR,
                width=1,
            )
            self.grid_lines.append(line_obj)

        for y in range(self.height + 1):
            y_screen = (self.height - y + 1) * self.grid_size
            line_obj = line(
                (self.grid_size, y_screen),
                ((self.width + 1) * self.grid_size, y_screen),
                MA_GRID_LINE_COLOR,
                width=1,
            )
            self.grid_lines.append(line_obj)

    def _draw_walls_multi_agent(self, wall_matrix: Grid):
        """
        Draw walls as dense vegetation / brown rocks.
        """
        for x in range(wall_matrix.width):
            for y in range(wall_matrix.height):
                if not wall_matrix[x][y]:
                    continue
                screen = self.to_screen((x, y))
                square(screen, 0.48 * self.grid_size, color=MA_WALL_FILL, filled=1)
                square(
                    screen,
                    0.48 * self.grid_size,
                    color=MA_WALL_OUTLINE,
                    filled=0,
                    behind=0,
                )

    def _draw_terrain_multi_agent(self, state: GameState):
        """
        Draw terrain with jungle-themed styling.
        """
        for obj in self.terrain_tiles + self.terrain_labels:
            remove_from_screen(obj)
        self.terrain_tiles = []
        self.terrain_labels = []

        layout = state.get_layout()
        assert layout is not None
        walls = layout.walls

        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                terrain_char = (
                    layout.get_terrain(x, y) if hasattr(layout, "get_terrain") else "."
                )
                if terrain_char == "~":
                    self._draw_fog(x, y)
                elif terrain_char == "^":
                    self._draw_mountain(x, y)
                elif terrain_char == "*":
                    self._draw_storm(x, y)

    def _draw_fog(self, x: int, y: int):
        """
        Draw fog zone with swirl pattern.
        """
        screen = self.to_screen((x, y))
        tile = square(screen, 0.5 * self.grid_size, color=MA_FOG_BASE, filled=1)
        self.terrain_tiles.append(tile)

        swirl1 = line(
            (screen[0] - 0.30 * self.grid_size, screen[1] - 0.10 * self.grid_size),
            (screen[0] + 0.30 * self.grid_size, screen[1] - 0.10 * self.grid_size),
            MA_FOG_SWIRL,
            width=2,
        )
        swirl2 = line(
            (screen[0] - 0.20 * self.grid_size, screen[1] + 0.12 * self.grid_size),
            (screen[0] + 0.20 * self.grid_size, screen[1] + 0.12 * self.grid_size),
            MA_FOG_SWIRL,
            width=2,
        )
        self.terrain_tiles.extend([swirl1, swirl2])

        label = text(screen, MA_FOG_TEXT, "2", "Arial", 14, "bold")
        self.terrain_labels.append(label)

    def _draw_mountain(self, x: int, y: int):
        """
        Draw mountain terrain with peak texture.
        """
        screen = self.to_screen((x, y))
        tile = square(screen, 0.5 * self.grid_size, color=MA_MOUNTAIN_BASE, filled=1)
        self.terrain_tiles.append(tile)

        peak = circle(
            (screen[0], screen[1] - 0.10 * self.grid_size),
            0.12 * self.grid_size,
            MA_MOUNTAIN_PEAK,
            MA_MOUNTAIN_PEAK,
        )
        self.terrain_tiles.append(peak)

        label = text(screen, MA_MOUNTAIN_TEXT, "3", "Arial", 14, "bold")
        self.terrain_labels.append(label)

    def _draw_storm(self, x: int, y: int):
        """
        Draw electric storm with lightning glow.
        """
        screen = self.to_screen((x, y))

        glow = circle(screen, 0.45 * self.grid_size, MA_STORM_GLOW, MA_STORM_GLOW)
        self.terrain_tiles.append(glow)

        core = circle(screen, 0.30 * self.grid_size, MA_STORM_CORE, MA_STORM_CORE)
        self.terrain_tiles.append(core)

        label = text(screen, MA_STORM_TEXT, "5", "Arial", 16, "bold")
        self.terrain_labels.append(label)

    def _draw_bases(self, base_positions: list[tuple[int, int]]):
        """
        Draw base/depot locations as blue squares.
        """
        self._base_images = []
        for pos in base_positions:
            screen = self.to_screen(pos)
            outer = square(
                screen, 0.40 * self.grid_size, color=MA_BASE_OUTLINE, filled=1
            )
            inner = square(screen, 0.32 * self.grid_size, color=MA_BASE_FILL, filled=1)
            label = text(screen, formatColor(1.0, 1.0, 1.0), "B", "Arial", 12, "bold")
            self._base_images.append([outer, inner, label])

    def _draw_delivery_points(self, state: GameState):
        """
        Draw delivery point markers (pending = gold, completed = green).
        """
        self._delivery_images = {}
        pending = state.get_pending_deliveries()
        assert self.layout is not None
        for pos in self.layout.delivery_positions:
            screen = self.to_screen(pos)
            if pos in pending:
                outer = circle(
                    screen,
                    0.32 * self.grid_size,
                    MA_DELIVERY_OUTLINE,
                    MA_DELIVERY_OUTLINE,
                )
                inner = circle(
                    screen,
                    0.26 * self.grid_size,
                    MA_DELIVERY_PENDING,
                    MA_DELIVERY_PENDING,
                )
                label = text(screen, MA_DELIVERY_TEXT, "E", "Arial", 11, "bold")
            else:
                outer = circle(
                    screen,
                    0.32 * self.grid_size,
                    MA_DELIVERY_DONE_OUTLINE,
                    MA_DELIVERY_DONE_OUTLINE,
                )
                inner = circle(
                    screen, 0.26 * self.grid_size, MA_DELIVERY_DONE, MA_DELIVERY_DONE
                )
                label = text(screen, MA_DELIVERY_TEXT, "\u2713", "Arial", 13, "bold")
            self._delivery_images[pos] = [outer, inner, label]

    def _mark_delivery_completed(self, pos: tuple[int, int]):
        """
        Change a delivery point from pending (gold) to completed (green).
        """
        if pos not in self._delivery_images:
            return
        outer, inner, label = self._delivery_images[pos]
        edit(
            outer,
            ("fill", MA_DELIVERY_DONE_OUTLINE),
            ("outline", MA_DELIVERY_DONE_OUTLINE),
        )
        edit(inner, ("fill", MA_DELIVERY_DONE), ("outline", MA_DELIVERY_DONE))
        changeText(label, "\u2713")

    def _draw_agents_multi_agent(self, state: GameState):
        """
        Draw drone and hunters for multi-agent mode.
        """
        drone_pos = state.get_drone_position()
        assert drone_pos is not None
        self._drone_images = self._draw_drone_at_position(drone_pos)

        self._hunter_images = []
        for hunter_pos in state.get_hunter_positions():
            parts = self._draw_hunter_at_position(hunter_pos)
            self._hunter_images.append(parts)
        refresh()

    def _draw_drone_at_position(self, pos: tuple[int, int]) -> list[int]:
        """
        Draw drone as bright cyan quadcopter icon.
        """
        screen = self.to_screen(pos)
        parts: list[int] = []

        body = circle(screen, 0.28 * self.grid_size, MA_DRONE_BODY, MA_DRONE_OUTLINE)
        parts.append(body)

        accent = circle(screen, 0.18 * self.grid_size, MA_DRONE_ACCENT, MA_DRONE_ACCENT)
        parts.append(accent)

        # Rotor arms (4 small circles at corners)
        offsets = [(-0.30, -0.30), (0.30, -0.30), (-0.30, 0.30), (0.30, 0.30)]
        for dx, dy in offsets:
            rx = screen[0] + dx * self.grid_size
            ry = screen[1] + dy * self.grid_size
            rotor = circle(
                (rx, ry), 0.08 * self.grid_size, MA_DRONE_ROTOR, MA_DRONE_ROTOR
            )
            parts.append(rotor)

        return parts

    def _draw_hunter_at_position(self, pos: tuple[int, int]):
        """
        Draw hunter as red triangle-ish icon.
        """
        screen = self.to_screen(pos)
        parts: list[int] = []

        outer = square(screen, 0.36 * self.grid_size, color=MA_HUNTER_BODY, filled=1)
        parts.append(outer)

        outline = square(
            screen, 0.36 * self.grid_size, color=MA_HUNTER_OUTLINE, filled=0, behind=0
        )
        parts.append(outline)

        inner = circle(
            screen, 0.14 * self.grid_size, MA_HUNTER_ACCENT, MA_HUNTER_ACCENT
        )
        parts.append(inner)

        label = text(screen, formatColor(1.0, 1.0, 1.0), "H", "Arial", 11, "bold")
        parts.append(label)

        return parts

    def _move_drone(self, pos: tuple[int, int]):
        """
        Move drone by redrawing at new position.
        """
        for p in self._drone_images:
            remove_from_screen(p)
        self._drone_images = self._draw_drone_at_position(pos)

    def _move_hunter(self, index: int, pos: tuple[int, int]):
        """
        Move a hunter by redrawing at new position.
        """
        if index < len(self._hunter_images):
            for p in self._hunter_images[index]:
                remove_from_screen(p)
            self._hunter_images[index] = self._draw_hunter_at_position(pos)

    def update(self, state: GameState):
        """
        Update display with new mission state (multi-agent mode).
        """
        self._update_multi_agent(state)

    def _update_multi_agent(self, new_state: GameState):
        """
        Update display for multi-agent (drone vs hunters) mode.
        """
        # Update drone position
        drone_position = new_state.get_drone_position()
        assert drone_position is not None
        self._move_drone(drone_position)

        # Update hunter positions
        hunter_positions = new_state.get_hunter_positions()
        for i, hunter_pos in enumerate(hunter_positions):
            self._move_hunter(i, hunter_pos)

        # Check for completed deliveries
        new_pending = new_state.get_pending_deliveries()
        for pos in self._pending_deliveries - new_pending:
            self._mark_delivery_completed(pos)
        self._pending_deliveries = set(new_pending)

        # Update info pane
        score = new_state.get_score()
        self.info_pane.update_score(score)
        completed = self._total_deliveries - len(self._pending_deliveries)
        self.info_pane.update_deliveries(completed, self._total_deliveries)

        refresh()
        if self.frame_time < 0:
            if not self._step_mode_message_shown:
                print(
                    "STEP-BY-STEP MODE: press any key in the graphics window to advance."
                )
                self._step_mode_message_shown = True
            wait_for_keys()
        else:
            sleep(self.frame_time)

    def draw(self, state: GameState) -> None:
        """
        Draw the current game state (multi-agent mode).
        """
        self.update(state)

    def pause(self) -> None:
        """
        Pause the display (wait for user input).
        """
        wait_for_keys()


class CspInfoPane:
    """
    Info pane for CSP drone delivery simulation.
    """

    def __init__(
        self,
        layout: DroneLayout,
        grid_size: float,
        total_deliveries: int,
        current_time: int,
    ):
        self.grid_size = grid_size
        self.width = layout.width * grid_size
        self.base = (layout.height + 1) * grid_size
        self.total_deliveries = total_deliveries

        if self.width < 300:
            self.font_size = 14
            self.title_size = 10
        elif self.width < 500:
            self.font_size = 16
            self.title_size = 11
        else:
            self.font_size = 18
            self.title_size = 12

        self._current_time = current_time
        self.draw_pane()

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (self.grid_size + x, self.base + y)

    def draw_pane(self):
        title_x = self.width // 2
        self.title_text = text(
            self.to_screen(title_x, 4),
            MA_TITLE_COLOR,
            "CSP DRONE DELIVERY",
            "Arial",
            self.title_size,
            "bold",
            anchor="n",
        )

        self.time_text = text(
            self.to_screen(INFO_PANE_PADDING, 24),
            MA_SCORE_COLOR,
            f"TIME: {self._current_time}",
            "Arial",
            self.font_size,
            "bold",
            anchor="nw",
        )

        self.delivery_text = text(
            self.to_screen(self.width - INFO_PANE_PADDING, 24),
            MA_DELIVERY_COUNT_COLOR,
            f"DELIVERIES: 0/{self.total_deliveries}",
            "Arial",
            self.font_size,
            "bold",
            anchor="ne",
        )

    def update_time(self, current_time: int):
        self._current_time = current_time
        changeText(self.time_text, f"TIME: {current_time}")

    def update_deliveries(self, completed: int, total: int):
        changeText(self.delivery_text, f"DELIVERIES: {completed}/{total}")


class CspGraphics(CspDisplay):
    """
    Graphics display for CSP drone delivery simulation.
    """

    def __init__(self, zoom: float = 1.0, frame_time: float = 0.0):
        self.zoom = zoom
        self.grid_size = DEFAULT_GRID_SIZE * zoom
        self.frame_time = frame_time
        self._step_mode_message_shown = False

        self.grid_lines: list[int] = []
        self.terrain_tiles: list[int] = []
        self.terrain_labels: list[int] = []

        self._drone_images: dict[str, tuple[list[int], int]] = {}
        self._delivery_images: dict[str, tuple[int, int, int, tuple[int, int]]] = {}
        self._base_images: list[list[int]] = []

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
            layout: DroneLayout object
            drone_states: dict of drone_id -> {"position": (x,y), ...}
            delivery_statuses: dict of delivery_id -> {
                "position": (x,y), "status": str,
                "time_window": (early, late)}
            current_time: current simulation time
        """
        self.layout = layout
        self.width = layout.width
        self.height = layout.height

        screen_width = 2 * self.grid_size + (self.width - 1) * self.grid_size
        screen_height = (
            2 * self.grid_size + (self.height - 1) * self.grid_size + INFO_PANE_HEIGHT
        )
        self._use_scroll = (
            screen_width > VIEWPORT_MAX_WIDTH or screen_height > VIEWPORT_MAX_HEIGHT
        )
        if self._use_scroll:
            self._content_width = int(screen_width)
            self._content_height = int(screen_height)
        else:
            if screen_width > MAX_WINDOW_WIDTH or screen_height > MAX_WINDOW_HEIGHT:
                scale = min(
                    MAX_WINDOW_WIDTH / screen_width, MAX_WINDOW_HEIGHT / screen_height
                )
                self.grid_size *= scale
                screen_width = 2 * self.grid_size + (self.width - 1) * self.grid_size
                screen_height = (
                    2 * self.grid_size
                    + (self.height - 1) * self.grid_size
                    + INFO_PANE_HEIGHT
                )
            self._content_width = int(screen_width)
            self._content_height = int(screen_height)

        self._make_window()

        total_deliveries = len(delivery_statuses)
        self.info_pane = CspInfoPane(
            layout, self.grid_size, total_deliveries, current_time
        )

        self._draw_static(layout)
        self._draw_delivery_points(delivery_statuses, current_time)
        self._draw_bases(layout.bases)
        self._draw_all_drones(drone_states)
        refresh()

    def _make_window(self):
        """
        Create window for CSP mode.
        """
        if getattr(self, "_use_scroll", False):
            viewport_w = min(VIEWPORT_MAX_WIDTH, self._content_width)
            viewport_h = min(VIEWPORT_MAX_HEIGHT, self._content_height)
            begin_graphics_scrollable(
                viewport_w,
                viewport_h,
                self._content_width,
                self._content_height,
                MA_BACKGROUND_COLOR,
                "CSP Drone Delivery",
            )
        else:
            begin_graphics(
                self._content_width,
                self._content_height,
                MA_BACKGROUND_COLOR,
                "CSP Drone Delivery",
            )

    def to_screen(self, point: tuple[int, int]) -> tuple[float, float]:
        """
        Convert grid coordinates to screen coordinates.
        """
        x, y = point
        x = (x + 1) * self.grid_size
        y = (self.height - y) * self.grid_size
        return (x, y)

    def _draw_static(self, layout: DroneLayout):
        """
        Draw static background elements.
        """
        self._draw_background_grid()
        self._draw_terrain(layout)
        self._draw_walls(layout.walls)

    def _draw_background_grid(self):
        """
        Draw subtle dark green grid lines.
        """
        for x in range(self.width + 1):
            x_screen = (x + 1) * self.grid_size
            line_obj = line(
                (x_screen, self.grid_size),
                (x_screen, (self.height + 1) * self.grid_size),
                MA_GRID_LINE_COLOR,
                width=1,
            )
            self.grid_lines.append(line_obj)

        for y in range(self.height + 1):
            y_screen = (self.height - y + 1) * self.grid_size
            line_obj = line(
                (self.grid_size, y_screen),
                ((self.width + 1) * self.grid_size, y_screen),
                MA_GRID_LINE_COLOR,
                width=1,
            )
            self.grid_lines.append(line_obj)

    def _draw_walls(self, wall_matrix: Grid):
        """
        Draw walls as dense vegetation / brown rocks.
        """
        for x in range(wall_matrix.width):
            for y in range(wall_matrix.height):
                if not wall_matrix[x][y]:
                    continue
                screen = self.to_screen((x, y))
                square(screen, 0.48 * self.grid_size, color=MA_WALL_FILL, filled=1)
                square(
                    screen,
                    0.48 * self.grid_size,
                    color=MA_WALL_OUTLINE,
                    filled=0,
                    behind=0,
                )

    def _draw_terrain(self, layout: DroneLayout):
        """
        Draw terrain with jungle-themed styling.
        """
        for obj in self.terrain_tiles + self.terrain_labels:
            remove_from_screen(obj)
        self.terrain_tiles = []
        self.terrain_labels = []

        walls = layout.walls
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                terrain_char = (
                    layout.get_terrain(x, y) if hasattr(layout, "get_terrain") else "."
                )
                if terrain_char == "~":
                    self._draw_fog(x, y)
                elif terrain_char == "^":
                    self._draw_mountain(x, y)
                elif terrain_char == "*":
                    self._draw_storm(x, y)

    def _draw_fog(self, x: int, y: int):
        """
        Draw fog zone with swirl pattern.
        """
        screen = self.to_screen((x, y))
        tile = square(screen, 0.5 * self.grid_size, color=MA_FOG_BASE, filled=1)
        self.terrain_tiles.append(tile)

        swirl1 = line(
            (screen[0] - 0.30 * self.grid_size, screen[1] - 0.10 * self.grid_size),
            (screen[0] + 0.30 * self.grid_size, screen[1] - 0.10 * self.grid_size),
            MA_FOG_SWIRL,
            width=2,
        )
        swirl2 = line(
            (screen[0] - 0.20 * self.grid_size, screen[1] + 0.12 * self.grid_size),
            (screen[0] + 0.20 * self.grid_size, screen[1] + 0.12 * self.grid_size),
            MA_FOG_SWIRL,
            width=2,
        )
        self.terrain_tiles.extend([swirl1, swirl2])

        label = text(screen, MA_FOG_TEXT, "2", "Arial", 14, "bold")
        self.terrain_labels.append(label)

    def _draw_mountain(self, x: int, y: int):
        """
        Draw mountain terrain with peak texture.
        """
        screen = self.to_screen((x, y))
        tile = square(screen, 0.5 * self.grid_size, color=MA_MOUNTAIN_BASE, filled=1)
        self.terrain_tiles.append(tile)

        peak = circle(
            (screen[0], screen[1] - 0.10 * self.grid_size),
            0.12 * self.grid_size,
            MA_MOUNTAIN_PEAK,
            MA_MOUNTAIN_PEAK,
        )
        self.terrain_tiles.append(peak)

        label = text(screen, MA_MOUNTAIN_TEXT, "3", "Arial", 14, "bold")
        self.terrain_labels.append(label)

    def _draw_storm(self, x: int, y: int):
        """
        Draw electric storm with lightning glow.
        """
        screen = self.to_screen((x, y))

        glow = circle(screen, 0.45 * self.grid_size, MA_STORM_GLOW, MA_STORM_GLOW)
        self.terrain_tiles.append(glow)

        core = circle(screen, 0.30 * self.grid_size, MA_STORM_CORE, MA_STORM_CORE)
        self.terrain_tiles.append(core)

        label = text(screen, MA_STORM_TEXT, "5", "Arial", 16, "bold")
        self.terrain_labels.append(label)

    def _draw_bases(self, base_positions: list[tuple[int, int]]):
        """
        Draw base/depot locations as blue squares.
        """
        self._base_images = []
        for pos in base_positions:
            screen = self.to_screen(pos)
            outer = square(
                screen, 0.40 * self.grid_size, color=MA_BASE_OUTLINE, filled=1
            )
            inner = square(screen, 0.32 * self.grid_size, color=MA_BASE_FILL, filled=1)
            label = text(screen, formatColor(1.0, 1.0, 1.0), "B", "Arial", 12, "bold")
            self._base_images.append([outer, inner, label])

    def _get_delivery_color(
        self, status: str, time_window: tuple[int, int], current_time: int
    ) -> tuple[str, str]:
        """
        Return (fill, outline) colors for a delivery point.
        """
        if status == "delivered":
            return MA_DELIVERY_DONE, MA_DELIVERY_DONE_OUTLINE
        early, late = time_window
        if current_time < early:
            return MA_DELIVERY_NOT_AVAILABLE, MA_DELIVERY_NOT_AVAILABLE_OUTLINE
        if early <= current_time <= late:
            if status == "waiting":
                return MA_DELIVERY_WAITING, MA_DELIVERY_WAITING_OUTLINE
            return MA_DELIVERY_PENDING, MA_DELIVERY_OUTLINE
        return MA_DELIVERY_NOT_AVAILABLE, MA_DELIVERY_NOT_AVAILABLE_OUTLINE

    def _draw_delivery_points(
        self, delivery_statuses: dict[str, DeliveryStatus], current_time: int
    ):
        """
        Draw delivery point markers with time-window coloring.
        """
        self._delivery_images = {}
        for d_id, info in delivery_statuses.items():
            pos = info["position"]
            status = info.get("status", "pending")
            time_window = info.get("time_window", (0, float("inf")))
            fill_color, outline_color = self._get_delivery_color(
                status, time_window, current_time
            )
            screen = self.to_screen(pos)
            if status == "delivered":
                lbl = "\u2713"
            else:
                lbl = "E"
            outer = circle(screen, 0.32 * self.grid_size, outline_color, outline_color)
            inner = circle(screen, 0.26 * self.grid_size, fill_color, fill_color)
            label = text(screen, MA_DELIVERY_TEXT, lbl, "Arial", 11, "bold")
            self._delivery_images[d_id] = (outer, inner, label, pos)

    def _update_delivery_points(
        self, delivery_statuses: dict[str, DeliveryStatus], current_time: int
    ):
        """
        Update delivery point colors based on time windows and status.
        """
        for d_id, info in delivery_statuses.items():
            status = info.get("status", "pending")
            time_window = info.get("time_window", (0, float("inf")))
            fill_color, outline_color = self._get_delivery_color(
                status, time_window, current_time
            )
            if d_id in self._delivery_images:
                outer, inner, label, _pos = self._delivery_images[d_id]
                edit(outer, ("fill", outline_color), ("outline", outline_color))
                edit(inner, ("fill", fill_color), ("outline", fill_color))
                if status == "delivered":
                    changeText(label, "\u2713")

    def _draw_drone_at_position(
        self, pos: tuple[int, int], color_index: int = 0
    ) -> list[int]:
        """
        Draw a drone with the given color index.
        """
        colors = CSP_DRONE_COLORS[color_index % len(CSP_DRONE_COLORS)]
        body_color, accent_color, outline_color = colors
        screen = self.to_screen(pos)
        parts: list[int] = []

        body = circle(screen, 0.28 * self.grid_size, body_color, outline_color)
        parts.append(body)

        accent = circle(screen, 0.18 * self.grid_size, accent_color, accent_color)
        parts.append(accent)

        offsets = [(-0.30, -0.30), (0.30, -0.30), (-0.30, 0.30), (0.30, 0.30)]
        for dx, dy in offsets:
            rx = screen[0] + dx * self.grid_size
            ry = screen[1] + dy * self.grid_size
            rotor = circle(
                (rx, ry), 0.08 * self.grid_size, MA_DRONE_ROTOR, MA_DRONE_ROTOR
            )
            parts.append(rotor)

        return parts

    def _draw_all_drones(self, drone_states: dict[str, DroneState]):
        """
        Draw all drones with different colors.
        """
        self._drone_images = {}
        for i, (drone_id, info) in enumerate(sorted(drone_states.items())):
            pos = info["position"]
            parts = self._draw_drone_at_position(pos, color_index=i)
            self._drone_images[drone_id] = (parts, i)

    def _move_drone(self, drone_id: str, pos: tuple[int, int], color_index: int):
        """
        Move a drone by redrawing at new position.
        """
        if drone_id in self._drone_images:
            parts, _ = self._drone_images[drone_id]
            for p in parts:
                remove_from_screen(p)
        new_parts = self._draw_drone_at_position(pos, color_index=color_index)
        self._drone_images[drone_id] = (new_parts, color_index)

    def update(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ):
        """
        Update display with new positions.
        """
        for i, (drone_id, info) in enumerate(sorted(drone_states.items())):
            pos = info["position"]
            color_index = i
            if drone_id in self._drone_images:
                _, color_index = self._drone_images[drone_id]
            self._move_drone(drone_id, pos, color_index)

        self._update_delivery_points(delivery_statuses, current_time)

        self.info_pane.update_time(current_time)
        delivered = sum(
            1
            for info in delivery_statuses.values()
            if info.get("status") == "delivered"
        )
        self.info_pane.update_deliveries(delivered, len(delivery_statuses))

        refresh()
        if self.frame_time < 0:
            if not self._step_mode_message_shown:
                print(
                    "STEP-BY-STEP MODE: press any key in the graphics window to advance."
                )
                self._step_mode_message_shown = True
            wait_for_keys()
        else:
            sleep(self.frame_time)

    def finish(self):
        """
        Clean up graphics resources and close window.
        """
        end_graphics()

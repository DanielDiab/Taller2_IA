from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from world.game_state import GameState
from world.layout import DroneLayout

if TYPE_CHECKING:
    from world.runner import DeliveryStatus, DroneState


class AdversarialDisplay(ABC):
    """
    Abstract base class for adversarial game displays.
    """

    @abstractmethod
    def initialize(self, state: GameState) -> None:
        """
        Initialize the display
        """
        pass

    @abstractmethod
    def update(self, state: GameState) -> None:
        """
        Update the display to reflect the current game state.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Perform any cleanup necessary when the game is finished.
        """
        pass


class CspDisplay(ABC):
    """
    Abstract base class for CSP problem displays.
    """

    @abstractmethod
    def initialize(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ) -> None:
        """
        Initialize the display.
        """
        pass

    @abstractmethod
    def update(
        self,
        layout: DroneLayout,
        drone_states: dict[str, DroneState],
        delivery_statuses: dict[str, DeliveryStatus],
        current_time: int,
    ) -> None:
        """
        Update the display to reflect the current variable assignment.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Perform any cleanup necessary when the CSP solving is finished.
        """
        pass

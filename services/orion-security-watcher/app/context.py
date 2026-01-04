import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import OrionCodec

from .notifications import Notifier
from .settings import get_settings
from .state_store import SecurityStateStore
from .guard import VisionGuard

logger = logging.getLogger("orion-security-watcher.context")


class AppContext:
    """
    Single place to construct shared singletons.
    Keeps main.py clean and prevents import spaghetti.
    """

    def __init__(self):
        self.settings = get_settings()

        self.bus = OrionBusAsync(
            url=self.settings.ORION_BUS_URL,
            enabled=self.settings.ORION_BUS_ENABLED,
            codec=OrionCodec(),
        )
        self.state_store = SecurityStateStore(self.settings)
        # self.visit_manager = VisitManager(self.settings) # Replaced by Guard
        self.guard = VisionGuard(self.settings)
        self.notifier = Notifier(self.settings)


ctx = AppContext()

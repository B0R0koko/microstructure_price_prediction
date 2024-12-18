from enum import Enum, auto


class CollectMode(Enum):
    MONTHLY = auto()
    DAILY = auto()

    def __str__(self):
        return self.name.lower()

    def lower(self) -> str:
        return self.name.lower()

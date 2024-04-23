from enum import Enum


class VRepMode(Enum):
    BLOCKING = 1
    STREAMING = 2
    BUFFER = 3
    ONESHOT = 4


class VRepReturn(Enum):
    OK = 1
    ERROR = 2

from enum import Enum, auto

class BackendType(Enum):
    AUTO = auto()
    OPENCL = auto()
    CUDA = auto()
    CPU = auto()

class ColoringMode(Enum):
    EXTERIOR = auto()
    INTERIOR = auto()
    HYBRID = auto()

class EngineMode(Enum):
    FULL_FRAME = auto()
    TILED = auto()

class Tools(Enum):
    Drag = auto()
    Click_zoom = auto()
    Wheel_zoom = auto()
    Set_center = auto()

class PrecisionMode(Enum):
    Single = auto()
    Double = auto()
    Arbitrary = auto()

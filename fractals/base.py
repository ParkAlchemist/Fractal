from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Viewport:
    """
    Holds the viewport parameters for rendering a fractal.
    X and Y limits determine the area of the fractal to render.
    Width and Height determine the size of the resulting image in pixels.
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int


@dataclass(frozen=True)
class OperationConfig:
    """
    Configuration for a specific operation in the fractal rendering process.
    Enabled indicates whether the operation is active.
    Parameters holds any additional settings required for the operation.
    """
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    fractal: Optional[str] = None


@dataclass(frozen=True)
class LODPass:
    """
    Level of Detail (LOD) pass configuration.
    Each pass has a name and scaling factors for resolution, maximum iterations,
    and samples.
    """
    name: str
    resolution_scale: float = 1.0
    max_iter_scale: float = 1.0
    samples_scale: float = 1.0


@dataclass
class RenderSettings:
    """
    Holds the rendering settings for a fractal.
    Max_iter determines the maximum number of iterations for the fractal calculation.
    Samples controls the number of samples per pixel for antialiasing.
    Precision specifies the floating-point precision for calculations.
    Base_res and target_res specify the resolution settings for the fractal.
    """
    max_iter: int
    samples: int = 1
    precision: np.dtype = np.float32

    operations: List[OperationConfig] = field(default_factory=lambda:[
        OperationConfig("iter"),
        OperationConfig("smooth"),
        OperationConfig("normalize")
    ])
    lods: List[LODPass] = field(default_factory=lambda:[LODPass("full")])
    backend: Optional[str] = None


# --------------- Program-spec primitives ----------------------

ArgRole = Literal["scalar", "buffer_out", "buffer_in"]
ArgSource = Literal["viewport", "settings", "constant", "runtime"]


@dataclass(frozen=True)
class ArgSpec:
    """
    Describes a kernel argument.
    - role: semantic role
    - dtype: dtype to cast to
    - shape_expr: textual expression for shapes (e.g. "H, W") for buffers
    - source: where the value comes from (viewport, settings, constant, runtime)
    """
    name: str
    role: ArgRole
    dtype: Any
    shape_expr: Optional[str] = None
    source: ArgSource = "runtime"


@dataclass(frozen=True)
class KernelStep:
    """
    A single kernel launch:
    - func: the kernel function / program object
    - args: ordered list of ArgSpec names for this kernel
    - meta: optional launch metadata
    """
    name: str
    func: Any
    args: List[str]
    meta: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ProgramSpec:
    """
    Full program for a backend:
    - backend: backend name
    - precision: data type for calculations (e.g., np.float32)
    - args: dict of ArgSpec (by name)
    - steps: list of KernelStep (execution order)
    - output_arg: name of the buffer that holds the final result
    """
    backend: str
    precision: Any
    args: Dict[str, ArgSpec]
    steps: List[KernelStep]
    output_arg: str


class Fractal(ABC):
    """
    An abstract base class for fractal types.
    """
    name: str

    @abstractmethod
    def build_arg_values(self, vp: Viewport, st: RenderSettings) -> Dict[
        str, Any]:
        ...

    @abstractmethod
    def get_program_spec(self, st: RenderSettings,
                         backend_name: str) -> ProgramSpec:
        ...

    @abstractmethod
    def output_semantics(self) -> str:
        ...

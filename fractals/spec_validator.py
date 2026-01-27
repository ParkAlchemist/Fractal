from __future__ import annotations
from typing import List, Optional

from fractals.base import ProgramSpec, KernelStep


class SpecError(Exception):
    """Aggregated ProgramSpec validation error(s)."""


def validate_program_spec(spec: ProgramSpec, *, backend_hint: Optional[str] = None) -> None:
    """
    Validates structural correctness of a ProgramSpec. Raises SpecError on failure.
    """
    errors: List[str] = []
    args = spec.args or {}
    steps = spec.steps or []

    # --- output_arg ---
    if not spec.output_arg or spec.output_arg not in args:
        errors.append(f"output_arg '{spec.output_arg}' is not present in spec.args.")
    else:
        out = args[spec.output_arg]
        if out.role != "buffer_out":
            errors.append(f"output_arg '{spec.output_arg}' must have role='buffer_out'.")

    # --- arg sanity ---
    for name, a in args.items():
        if a.role not in {"scalar", "buffer_out", "buffer_in"}:
            errors.append(f"Arg '{name}': invalid role='{a.role}'.")
        if a.dtype is None:
            errors.append(f"Arg '{name}': dtype is None.")
        if a.role in {"buffer_out", "buffer_in"} and not a.shape_expr:
            errors.append(f"Arg '{name}': buffer role but no shape_expr provided.")

    # --- steps sanity ---
    for idx, step in enumerate(steps):
        if not isinstance(step, KernelStep):
            errors.append(f"Step[{idx}] is not a KernelStep.")
            continue

        # step.args defined
        if not step.args or not isinstance(step.args, (list, tuple)):
            errors.append(f"Step[{idx}] '{step.name}': args must be a non-empty list.")
            continue

        # all step.args exist in spec.args
        for a in step.args:
            if a not in args:
                errors.append(f"Step[{idx}] '{step.name}': arg '{a}' not found in spec.args.")

        # duplicate args guard
        if len(set(step.args)) != len(step.args):
            errors.append(f"Step[{idx}] '{step.name}': args contain duplicates.")

        # backend-specific function checks
        fn = step.func
        target_backend = (backend_hint or spec.backend or "").upper()

        if isinstance(fn, dict):
            # Only treat dict as OpenCL meta when target backend is OPENCL
            if target_backend == "OPENCL":
                if "src" not in fn or "kernel_name" not in fn:
                    errors.append(f"Step[{idx}] '{step.name}': OpenCL meta must include 'src' and 'kernel_name'.")
                if "build_options" in fn and not isinstance(fn["build_options"], (list, tuple)):
                    errors.append(f"Step[{idx}] '{step.name}': 'build_options' must be a list[str] if provided.")
            else:
                if "func" not in fn or not callable(fn["func"]):
                    errors.append(f"Step[{idx}] '{step.name}': 'func' must be callable for backend '{target_backend}'.")
        elif isinstance(fn, str):
            # allow raw OpenCL source string (kernel_name defaults to step.name in your backend)
            pass
        else:
            # CUDA/CPU callable expected
            if not callable(fn):
                if target_backend == "OPENCL":
                    errors.append(f"Step[{idx}] '{step.name}': expected OpenCL dict/str meta; got {type(fn).__name__}.")
                else:
                    errors.append(f"Step[{idx}] '{step.name}': 'func' must be callable.")

    if errors:
        raise SpecError("ProgramSpec validation failed:\n- " + "\n- ".join(errors))


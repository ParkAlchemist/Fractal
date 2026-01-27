from dataclasses import dataclass
from typing import Dict, Any, List, Set
import numpy as np

from fractals.base import (
    Fractal, Viewport, RenderSettings, ProgramSpec, ArgSpec, KernelStep
)
from kernel_sources import load_kernel, \
    get_op_descriptor  # re-exported entry points


@dataclass
class MandelbrotFractal(Fractal):
    name: str = "mandelbrot"

    # Scalar map remains mostly as you had it
    def build_arg_values(self, vp: Viewport, st: RenderSettings) -> Dict[
        str, Any]:
        return {
            "min_x": vp.min_x, "max_x": vp.max_x,
            "min_y": vp.min_y, "max_y": vp.max_y,
            "width": vp.width, "height": vp.height,
            "max_iter": st.max_iter, "samples": st.samples,
            # common defaults; can be overridden per-op via OperationConfig.params
            "bailout": getattr(st, "bailout", 4.0),
            "min_val": getattr(st, "min_val", 0.0),
            "max_val": getattr(st, "max_val", float(st.max_iter)),
        }

    def _resolve_pipeline(self, st: RenderSettings) -> List[Dict[str, Any]]:
        """
        Return a topo-sorted list of operation dicts:
        { name, fractal, params, depends_on }
        """
        requested = [op for op in st.operations if op.enabled]
        # Attach dependency info from registry descriptors
        ops = []
        for op in requested:
            # If no fractal set, default to this fractal
            op_fractal = op.fractal or self.name
            desc = {}
            try:
                desc = get_op_descriptor(op_fractal, op.name)  # registry call
            except KeyError:
                # Allow ops without descriptors (no deps/defaults)
                desc = {"depends_on": [], "default_params": {}}
            ops.append({
                "name": op.name,
                "fractal": op_fractal,
                "params": {**desc.get("default_params", {}), **op.parameters},
                "depends_on": desc.get("depends_on", []),
            })

        # Topological sort on names (within the chosen ops)
        name_to_op = {o["name"]: o for o in ops}
        visited, order = set(), []

        def dfs(n: str, stack: Set[str]):
            if n in visited:
                return
            if n in stack:
                raise ValueError(f"Cycle in op dependencies at {n}")
            stack.add(n)
            for d in name_to_op.get(n, {}).get("depends_on", []):
                if d not in name_to_op:
                    raise ValueError(f"Missing dependency '{d}' for '{n}'")
                dfs(d, stack)
            stack.remove(n)
            visited.add(n)
            order.append(n)

        for o in ops:
            dfs(o["name"], set())

        return [name_to_op[n] for n in order]

    @staticmethod
    def _synthesize_argspecs(cast, pipeline_meta: List[Dict[str, Any]]) -> \
    Dict[str, ArgSpec]:
        args: Dict[str, ArgSpec] = {}
        # viewport/settings scalars
        for k in ("min_x", "max_x", "min_y", "max_y"):
            args[k] = ArgSpec(k, role="scalar", dtype=cast, source="viewport")
        for k in ("width", "height"):
            args[k] = ArgSpec(k, role="scalar", dtype=np.int32,
                              source="viewport")
        args["max_iter"] = ArgSpec("max_iter", role="scalar", dtype=np.int32,
                                   source="settings")
        args["samples"] = ArgSpec("samples", role="scalar", dtype=np.int32,
                                  source="settings")

        # Scan kernel meta
        for km in pipeline_meta:
            for s in km.get("scalars", []):
                if s not in args:
                    dt = cast
                    if s in ("width", "height", "samples", "max_iter"):
                        # already declared (with precise dtypes)
                        continue
                    args[s] = ArgSpec(s, role="scalar", dtype=dt,
                                      source="runtime")
            for b in km.get("produces", []):
                # default shape: "H,W"; override per-op later if you add shape hints
                args[b] = ArgSpec(b, role="buffer_out", dtype=cast,
                                  shape_expr="H,W")
            for b in km.get("consumes", []):
                if b not in args:
                    args[b] = ArgSpec(b, role="buffer_in", dtype=cast,
                                      shape_expr="H,W")
        return args

    def get_program_spec(self, st: RenderSettings,
                         backend_name: str) -> ProgramSpec:
        cast = st.precision
        backend = backend_name.upper()
        prec = "f64" if cast == np.float64 else "f32"

        # 1) Resolve ops + dependencies
        op_plan = self._resolve_pipeline(st)

        # 2) Load kernel meta for each op (backend+precision)
        pipeline_meta: List[Dict[str, Any]] = []
        steps: List[KernelStep] = []

        for op in op_plan:
            km = load_kernel(backend, op["fractal"], op["name"],
                             prec)  # loader with registry fallback
            km = {**km, "params": op["params"], "fractal": op["fractal"]}
            pipeline_meta.append(km)

        # 3) Build ArgSpecs
        args = self._synthesize_argspecs(cast, pipeline_meta)

        # 4) Steps from arg_order
        def to_step(op_name: str, meta: Dict[str, Any]) -> KernelStep:
            blk = meta.get("block", None)
            return KernelStep(
                name=f"{meta.get('fractal', self.name)}_{op_name}",
                func=meta if backend == "OPENCL" else meta["func"],
                args=meta["arg_order"],
                meta={"block": blk} if blk else None
            )

        for op, meta in zip(op_plan, pipeline_meta):
            steps.append(to_step(op["name"], meta))

        # 5) Output arg: last op's last produced buffer (fallback to "iter_norm")
        last_prod = pipeline_meta[-1].get("produces",
                                          []) if pipeline_meta else []
        output_arg = last_prod[-1] if last_prod else "iter_norm"

        return ProgramSpec(
            backend=backend,
            precision=cast,
            args=args,
            steps=steps,
            output_arg=output_arg
        )

    def output_semantics(self) -> str:
        return "iterations"

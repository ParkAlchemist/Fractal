from kernel_sources.registry import register_op_descriptor

# "iter" produces raw iteration counts and |z| at escape; no deps.
register_op_descriptor(
    "mandelbrot", "iter",
    depends_on=[],
    default_params={"bailout": 4.0}
)

# "smooth" consumes iter_raw/z_mag, produces iter_smooth; depends on iter.
register_op_descriptor(
    "mandelbrot", "smooth",
    depends_on=["iter"],
    default_params={}
)

# "normalize" consumes iter_smooth, produces iter_norm; depends on smooth
register_op_descriptor(
    "mandelbrot", "normalize",
    depends_on=["smooth"],
    default_params={"min_val": 0.0, "max_val": None}
)

import os


def auto_detect_mujoco_gl() -> str:
    """Auto-detect best MuJoCo GL backend.

    Priority:
        1) egl
        2) osmesa

    Returns:
        Selected backend string.

    Prints a warning if neither backend appears available.
    """
    # 1) Prefer EGL if render node exists (typical GPU system)

    graphics_backend: str

    if os.access(path="/dev/dri/renderD128", mode=os.R_OK | os.W_OK):
        graphics_backend = "egl"
    elif any(
        os.path.exists(path=p)
        for p in (
            "/usr/lib/libOSMesa.so",
            "/usr/lib64/libOSMesa.so",
            "/usr/lib/x86_64-linux-gnu/libOSMesa.so",
        )
    ):
        graphics_backend = "osmesa"
    else:
        print(
            "[train] WARNING: EGL not accessible and OSMesa not explicitly detected. "
            "Falling back to 'osmesa'.",
            flush=True,
        )
        graphics_backend = "osmesa"

    return graphics_backend

def __getattr__(name):
    if name in ("run_debug", "run_debug_month"):
        from .reports import run_debug, run_debug_month
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["run_debug", "run_debug_month"]

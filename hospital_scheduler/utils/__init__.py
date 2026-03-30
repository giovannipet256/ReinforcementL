def __getattr__(name):
    if name in ("resolve_model_path", "load_model", "get_latest_checkpoint"):
        from .io import resolve_model_path, load_model, get_latest_checkpoint
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["resolve_model_path", "load_model", "get_latest_checkpoint"]

from .ferie import (
    FERIE_LEVEL_1, FERIE_LEVEL_2, FERIE_LEVEL_3, FERIE_LEVEL_4, FERIE_LEVEL_5,
    FERIE_DEFAULTS, DEFAULT_PREFERENCES,
    generate_random_ferie, generate_annual_ferie, get_ferie_for_month,
    generate_random_preferences,
)

# callbacks and curriculum require gymnasium / stable_baselines3 at import time;
# import them lazily so the pure-Python submodules remain usable without RL deps.
def __getattr__(name):
    if name in ("KPITrainingCallback", "PeriodicCheckpointCallback"):
        from .callbacks import KPITrainingCallback, PeriodicCheckpointCallback
        return locals()[name]
    if name in ("CURRICULUM", "LEVEL_GATES", "make_env", "build_model", "train_curriculum"):
        from . import curriculum as _c
        return getattr(_c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FERIE_LEVEL_1", "FERIE_LEVEL_2", "FERIE_LEVEL_3", "FERIE_LEVEL_4", "FERIE_LEVEL_5",
    "FERIE_DEFAULTS", "DEFAULT_PREFERENCES",
    "generate_random_ferie", "generate_annual_ferie", "get_ferie_for_month",
    "generate_random_preferences",
    "KPITrainingCallback", "PeriodicCheckpointCallback",
    "CURRICULUM", "LEVEL_GATES", "make_env", "build_model", "train_curriculum",
]

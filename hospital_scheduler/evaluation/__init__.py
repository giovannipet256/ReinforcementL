from .metrics import (
    DEFAULT_WEIGHTS,
    compute_shift_distribution_score,
    compute_doctor_coverage_score,
    compute_episode_score,
)

# evaluator requires gymnasium at import time; import lazily
def __getattr__(name):
    if name in ("evaluate_model", "run_single_episode"):
        from .evaluator import evaluate_model, run_single_episode
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_WEIGHTS",
    "compute_shift_distribution_score",
    "compute_doctor_coverage_score",
    "compute_episode_score",
    "evaluate_model",
    "run_single_episode",
]

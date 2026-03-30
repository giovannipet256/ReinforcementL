# env submodules require gymnasium; import lazily so config/training/metrics
# remain importable in environments without gymnasium installed.
def __getattr__(name):
    if name in ("HospitalSchedulingEnv", "HospitalActionMasker"):
        from .environment import HospitalSchedulingEnv
        from .masking import HospitalActionMasker
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["HospitalSchedulingEnv", "HospitalActionMasker"]

from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolves a model path, appending .zip if needed."""
    p = Path(model_path)
    candidates = [p] if p.suffix == '.zip' else [p, p.with_suffix('.zip')]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'Modello non trovato: {candidates}')


def load_model(model_path: str | Path):
    """Loads a MaskablePPO or PPO model from the given path."""
    path = resolve_model_path(model_path)
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(str(path))
        except Exception:
            pass
    if PPO is not None:
        return PPO.load(str(path))
    raise ImportError('Nessuna libreria RL disponibile (stable_baselines3 / sb3_contrib)')


def get_latest_checkpoint() -> Tuple[Path | None, int]:
    """
    Scans all checkpoint directories (models/checkpoints/level_1-5) and returns
    the latest checkpoint by timestep number.

    Returns:
        (checkpoint_path, timesteps) or (None, 0) if no checkpoints found.
    """
    # Try both relative and absolute paths
    models_dir = Path("models/checkpoints")
    if not models_dir.exists():
        # Try absolute path based on this file's location
        models_dir = Path(__file__).parent.parent.parent / "models" / "checkpoints"
    
    if not models_dir.exists():
        return None, 0

    latest_path: Path | None = None
    latest_timesteps = 0

    for level in range(1, 6):
        level_dir = models_dir / f"level_{level}"
        if not level_dir.exists():
            continue
        for checkpoint_file in level_dir.glob("checkpoint_*.zip"):
            try:
                timesteps = int(checkpoint_file.stem.replace("checkpoint_", ""))
                if timesteps > latest_timesteps:
                    latest_timesteps = timesteps
                    latest_path = checkpoint_file
            except (ValueError, AttributeError):
                continue

    return latest_path, latest_timesteps

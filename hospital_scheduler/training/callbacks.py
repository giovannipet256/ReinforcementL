from __future__ import annotations

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object

from hospital_scheduler.training.ferie import generate_random_ferie, generate_random_preferences


class KPITrainingCallback(BaseCallback):
    """Logs KPI metrics at each rollout end and randomises ferie/preferences at rollout start."""

    def __init__(self, curriculum_level: int = 1):
        super().__init__()
        self.curriculum_level = curriculum_level
        self.episode_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        vals = getattr(self.model.logger, "name_to_value", {})
        try:
            payload = {
                "ep_rew_mean":          round(float(vals.get("rollout/ep_rew_mean",          0.0)), 4),
                "ep_len_mean":          round(float(vals.get("rollout/ep_len_mean",          0.0)), 4),
                "approx_kl":            round(float(vals.get("train/approx_kl",             0.0)), 6),
                "clip_fraction":        round(float(vals.get("train/clip_fraction",         0.0)), 6),
                "entropy_loss":         round(float(vals.get("train/entropy_loss",          0.0)), 6),
                "value_loss":           round(float(vals.get("train/value_loss",            0.0)), 6),
                "policy_gradient_loss": round(float(vals.get("train/policy_gradient_loss", 0.0)), 6),
            }
            print(payload)
        except Exception:
            pass

    def _on_rollout_start(self) -> None:
        """Generates new ferie and preferences for each rollout."""
        self.episode_count += 1
        new_ferie = generate_random_ferie(self.curriculum_level, seed=None)
        new_prefs = generate_random_preferences(self.curriculum_level, seed=None)

        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                env.unwrapped.ferie_table = new_ferie
                env.unwrapped.preferences_table = new_prefs
        else:
            self.training_env.unwrapped.ferie_table = new_ferie
            self.training_env.unwrapped.preferences_table = new_prefs


class PeriodicCheckpointCallback(BaseCallback):
    """Saves model checkpoints at regular intervals (every N timesteps)."""

    def __init__(self, save_freq: int, save_path: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % (self.save_freq // self.model.get_env().num_envs) == 0:
            self.model.save(f"{self.save_path}/checkpoint_{self.num_timesteps}")
        return True

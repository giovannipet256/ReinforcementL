from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hospital_scheduler.config.rewards import REWARD_WEIGHTS_BASE
from hospital_scheduler.env.environment import HospitalSchedulingEnv
from hospital_scheduler.env.masking import HospitalActionMasker
from hospital_scheduler.training.ferie import (
    FERIE_DEFAULTS, DEFAULT_PREFERENCES,
    generate_random_ferie, generate_random_preferences,
)
from hospital_scheduler.training.callbacks import KPITrainingCallback, PeriodicCheckpointCallback

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CallbackList
    SB3_AVAILABLE = True
except ImportError:
    PPO = Monitor = CallbackList = None
    SB3_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    MASKABLE_AVAILABLE = True
except ImportError:
    MaskablePPO = MaskableEvalCallback = None
    MASKABLE_AVAILABLE = False

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================
# Global training budget
# ============================================================
MAX_GLOBAL_TIMESTEPS = 2_000_010

# ============================================================
# Curriculum stages
# ============================================================
CURRICULUM = [
    {
        "level": 1,
        "name": "Base",
        "max_timesteps": 200_000,
        "chunk_timesteps": 50_000,
        "ferie_table": FERIE_DEFAULTS[1],
        "ap_prob": 0, #questo parametro non controlla nulla
        "reward_weights": {**REWARD_WEIGHTS_BASE, "coverage": 1.5, "legality": 1.5, "fairness": 0.7, "calendar": 0.5},
    },
    {
        "level": 2,
        "name": "Ferie semplici",
        "max_timesteps": 300_000,
        "chunk_timesteps": 75_000,
        "ferie_table": FERIE_DEFAULTS[2],
        "ap_prob": 0,
        "reward_weights": {**REWARD_WEIGHTS_BASE, "coverage": 1.4, "legality": 1.4, "fairness": 0.8, "calendar": 0.7},
    },
    {
        "level": 3,
        "name": "Ferie critiche",
        "max_timesteps": 400_000,
        "chunk_timesteps": 100_000,
        "ferie_table": FERIE_DEFAULTS[3],
        "ap_prob": 0,
        "reward_weights": {**REWARD_WEIGHTS_BASE, "coverage": 1.3, "legality": 1.4, "fairness": 1.0, "calendar": 0.9},
    },
    {
        "level": 4,
        "name": "Weekend/Festivi/Fairness",
        "max_timesteps": 500_000,
        "chunk_timesteps": 125_000,
        "ferie_table": FERIE_DEFAULTS[4],
        "ap_prob": 0,
        "reward_weights": {
            **REWARD_WEIGHTS_BASE,
            "coverage": 1.2, "legality": 1.3, "fairness": 1.3, "calendar": 1.3, "weekly": 1.1,
        },
    },
    {
        "level": 5,
        "name": "Stress realistico",
        "max_timesteps": 400_000,
        "chunk_timesteps": 150_000,
        "ferie_table": FERIE_DEFAULTS[5],
        "ap_prob": 0,
        "reward_weights": {
            **REWARD_WEIGHTS_BASE,
            "coverage": 1.2, "legality": 1.2, "fairness": 1.5, "calendar": 1.4,
            "weekly": 1.3, "monthly": 1.2, "efficiency": 1.1,
        },
    },
]

# ============================================================
# Level advancement gates
# ============================================================
LEVEL_GATES = {
    1: {"avg_reward": 5.0,  "max_hard_overrides": 20, "max_jolly": 12},
    2: {"avg_reward": 7.0,  "max_hard_overrides": 16, "max_jolly": 10},
    3: {"avg_reward": 8.0,  "max_hard_overrides": 12, "max_jolly": 8},
    4: {"avg_reward": 9.0,  "max_hard_overrides": 10, "max_jolly": 6},
    5: {"avg_reward": 10.0, "max_hard_overrides": 8,  "max_jolly": 5},
}

# ============================================================
# Environment factory
# ============================================================
def make_env(
    level: int,
    reward_weights: Dict[str, float],
    ap_prob: float,
    monitor_filename: Optional[str] = None,
    fixed_ferie: bool = False,
    fixed_preferences: bool = False,
):
    """
    Creates an environment with random or fixed ferie/preferences.

    Args:
        level: Curriculum level (1-5)
        reward_weights: Reward component weights
        ap_prob: AP assignment probability
        monitor_filename: CSV file for monitoring
        fixed_ferie: If True, uses default ferie (stable eval)
        fixed_preferences: If True, uses default preferences (stable eval)
    """
    ferie_table = FERIE_DEFAULTS.get(level, {}) if fixed_ferie else generate_random_ferie(level, seed=None)
    preferences_table = DEFAULT_PREFERENCES.get(level, {}) if fixed_preferences else generate_random_preferences(level, seed=None)

    env = HospitalSchedulingEnv(
        ferie_table=ferie_table,
        reward_weights=reward_weights,
        ap_probability_dirigenza=ap_prob,
        preferences_table=preferences_table,
    )
    env = HospitalActionMasker(env)
    if Monitor is not None:
        env = Monitor(env, filename=monitor_filename)
    return env


# ============================================================
# Model builder
# ============================================================
def build_model(env, use_masking: bool = True):
    if not SB3_AVAILABLE:
        raise ImportError("stable_baselines3 non installato nell'environment attivo.")

    from torch import nn  # lazy import — torch only needed at training time
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    )

    common_kwargs: Dict[str, Any] = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(LOGS_DIR),
    )

    if use_masking:
        if not MASKABLE_AVAILABLE:
            raise ImportError("MaskablePPO richiesto ma sb3_contrib non è installato.")
        return MaskablePPO(**common_kwargs)
    return PPO(**common_kwargs)


# ============================================================
# Curriculum trainer
# ============================================================
def train_curriculum(use_masking: bool = True, pretrained_model_path: Optional[str] = None, start_level: int = 1):
    from hospital_scheduler.utils.io import load_model  # avoid circular at module level

    if not SB3_AVAILABLE:
        raise ImportError("stable_baselines3 non installato")

    if start_level < 1 or start_level > 5:
        raise ValueError(f"start_level must be between 1 and 5, got {start_level}")

    model = None
    total_global_timesteps = 0

    for stage in CURRICULUM:
        if stage['level'] < start_level:
            continue
        if total_global_timesteps >= MAX_GLOBAL_TIMESTEPS:
            print(f"[STOP] Raggiunto budget globale massimo: {MAX_GLOBAL_TIMESTEPS}")
            break

        level = stage["level"]
        level_name = stage["name"]
        max_timesteps = stage["max_timesteps"]
        chunk_timesteps = stage["chunk_timesteps"]

        print(f"\n=== LIVELLO {level} | {level_name} ===")

        env = make_env(
            level=level,
            reward_weights=stage["reward_weights"],
            ap_prob=stage["ap_prob"],
            monitor_filename=str(LOGS_DIR / f"train_monitor_level_{level}.csv"),
            fixed_ferie=False,
            fixed_preferences=False,
        )
        eval_env = make_env(
            level=level,
            reward_weights=stage["reward_weights"],
            ap_prob=stage["ap_prob"],
            monitor_filename=str(LOGS_DIR / f"eval_monitor_level_{level}.csv"),
            fixed_ferie=True,
            fixed_preferences=True,
        )

        if model is None:
            if pretrained_model_path:
                model = load_model(pretrained_model_path)
                print(f"[LOAD] Modello caricato da: {pretrained_model_path}")
            else:
                model = build_model(env, use_masking=use_masking)
                print("[NEW] Nuovo modello creato da zero")

        model.set_env(env)

        callbacks = [KPITrainingCallback(curriculum_level=level)]

        checkpoint_dir = MODELS_DIR / "checkpoints" / f"level_{level}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(PeriodicCheckpointCallback(save_freq=25000, save_path=str(checkpoint_dir)))

        if use_masking and MASKABLE_AVAILABLE and MaskableEvalCallback is not None:
            callbacks.append(
                MaskableEvalCallback(
                    eval_env,
                    best_model_save_path=str(MODELS_DIR / f"best_level_{level}"),
                    log_path=str(LOGS_DIR / f"eval_level_{level}"),
                    eval_freq=5_000,
                    deterministic=True,
                    render=False,
                )
            )

        callback = CallbackList(callbacks) if CallbackList is not None else callbacks[0]

        gate = LEVEL_GATES[level]
        trained_on_level = 0
        gate_passed = False

        while trained_on_level < max_timesteps:
            if total_global_timesteps >= MAX_GLOBAL_TIMESTEPS:
                print(f"[STOP] Raggiunto budget globale massimo: {MAX_GLOBAL_TIMESTEPS}")
                break

            level_remaining = max_timesteps - trained_on_level
            global_remaining = MAX_GLOBAL_TIMESTEPS - total_global_timesteps
            current_chunk = min(chunk_timesteps, level_remaining, global_remaining)

            if current_chunk <= 0:
                break

            print(
                f"[TRAIN] L{level} | chunk={current_chunk} | "
                f"trained_on_level={trained_on_level}/{max_timesteps} | "
                f"global={total_global_timesteps}/{MAX_GLOBAL_TIMESTEPS}"
            )

            model.learn(
                total_timesteps=current_chunk,
                reset_num_timesteps=False,
                tb_log_name=f"curriculum_level_{level}",
                callback=callback,
                progress_bar=False,
            )

            trained_on_level += current_chunk
            total_global_timesteps += current_chunk

            from hospital_scheduler.evaluation.evaluator import evaluate_model
            metrics = evaluate_model(model, eval_env, n_episodes=2)
            print(f"[EVAL L{level}] {metrics}")

            gate_passed = (
                metrics["avg_reward"] >= gate["avg_reward"]
                and metrics["max_hard_overrides"] <= gate["max_hard_overrides"]
                and metrics["max_jolly"] <= gate["max_jolly"]
            )

            if gate_passed:
                print(f"[GATE] Livello {level} superato dopo {trained_on_level} timesteps")
                break
            else:
                print(f"[GATE] Livello {level} non ancora superato, continuo...")

        if not gate_passed and total_global_timesteps < MAX_GLOBAL_TIMESTEPS:
            print(
                f"[WARN] Livello {level} non superato, ma budget massimo del livello raggiunto "
                f"({trained_on_level}/{max_timesteps}). Passo al livello successivo."
            )

        model.save(str(MODELS_DIR / f"ppo_hospital_level_{level}"))
        print(f"[SAVE] Salvato livello {level} in models/ppo_hospital_level_{level}.zip")

        if total_global_timesteps >= MAX_GLOBAL_TIMESTEPS:
            print("[STOP] Interruzione curriculum per budget globale massimo raggiunto.")
            break

    if model is not None:
        model.save(str(MODELS_DIR / "ppo_hospital_final"))
        print("\n[SAVE] Modello finale salvato in models/ppo_hospital_final.zip")
        print(f"[TOTAL] Timesteps globali eseguiti: {total_global_timesteps}")
    else:
        print("[WARN] Nessun modello addestrato.")

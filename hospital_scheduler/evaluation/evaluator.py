from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from hospital_scheduler.config.settings import (
    AGENT_SHIFTS, ALL_SHIFTS, EMPLOYEES, CONTRACT_TYPES,
    PREFERENCE_TYPES, SHIFT_NAME_TO_ID,
)


def _extract_info_or_env_metric(info: Dict[str, Any], env, key: str, default: Any):
    if isinstance(info, dict) and key in info:
        return info.get(key, default)
    try:
        return getattr(env.unwrapped, key)
    except Exception:
        return default


def evaluate_model(model, env, n_episodes: int = 3) -> Dict[str, float]:
    """Quick evaluation: returns avg_reward, max_hard_overrides, max_jolly."""
    rewards = []
    hard_overrides = []
    jolly = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        last_info: Dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = info if isinstance(info, dict) else {}

        rewards.append(ep_reward)
        hard_overrides.append(_extract_info_or_env_metric(last_info, env, "total_hard_overrides", 999))
        jolly.append(_extract_info_or_env_metric(last_info, env, "total_jolly", 999))

    return {
        "avg_reward":          float(sum(rewards) / max(len(rewards), 1)),
        "max_hard_overrides":  float(max(hard_overrides) if hard_overrides else 999),
        "max_jolly":           float(max(jolly) if jolly else 999),
    }


def run_single_episode(env, model) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs a single episode and returns per-day rows + episode-level stats."""
    obs, _ = env.reset()
    done = False
    rows: List[Dict[str, Any]] = []

    # Extract preferences_table from environment
    preferences_table = {}
    if hasattr(env, 'env') and hasattr(env.env, 'unwrapped'):
        preferences_table = getattr(env.env.unwrapped, 'preferences_table', {})

    episode_stats: Dict[str, Any] = {
        'morning_counts': {},
        'afternoon_counts': {},
        'night_counts': {},
        'rest_counts': {},
        'total_rests': 0,
        'total_nights': 0,
        'hours_total': {},
        'overtime_by_employee': {},
        'total_reward': 0.0,
    }

    while not done:
        mask_mat = env.action_mask_matrix()
        action, _ = model.predict(obs, deterministic=False)
        action = list(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_stats['total_reward'] += reward

        for i, emp in enumerate(EMPLOYEES):
            final_shift_name = info['final_shifts'][emp['name']]

            pref_key = (emp['id'], info['day'])
            preference_type = None
            preference_rispettata = None
            if pref_key in preferences_table:
                preference_type = preferences_table[pref_key]
                blocked_shifts = PREFERENCE_TYPES.get(preference_type, set())
                shift_id = SHIFT_NAME_TO_ID.get(final_shift_name, 3)
                preference_rispettata = shift_id not in blocked_shifts

            rows.append({
                'giorno': info['day'] + 1,
                'date': info['date'],
                'date_label': info['date_label'],
                'weekday': info['weekday_idx'],
                'weekday_label': info['weekday_label'],
                'dipendente': emp['name'],
                'ruolo': 'Medico' if emp['type'] == 0 else 'Infermiere',
                'azione_agente': AGENT_SHIFTS[int(action[i])],
                'turno_finale': final_shift_name,
                'mask_M': bool(mask_mat[i, 0]),
                'mask_P': bool(mask_mat[i, 1]),
                'mask_N': bool(mask_mat[i, 2]),
                'mask_R': bool(mask_mat[i, 3]),
                'violazioni': ' | '.join(
                    v for v in info.get('violations', [])
                    if emp['name'].split()[-1] in v or emp['name'] in v
                ),
                'soft_violations': ' | '.join(info.get('soft_violations', [])),
                'hard_overrides': ' | '.join(
                    v for v in info.get('hard_overrides', []) if emp['name'] in v
                ),
                'is_weekend':    info['is_weekend'],
                'is_holiday':    info['is_holiday'],
                'is_prefestive': info['is_prefestive'],
                'is_special_day': info['is_special_day'],
                'preference_type': preference_type,
                'preference_rispettata': preference_rispettata,
                'reward_total': reward,
                **{f'reward_{k}': v for k, v in info['reward_components'].items()},
            })

            # Track counts
            if final_shift_name == 'M':
                episode_stats['morning_counts'][emp['name']] = episode_stats['morning_counts'].get(emp['name'], 0) + 1
            elif final_shift_name == 'P':
                episode_stats['afternoon_counts'][emp['name']] = episode_stats['afternoon_counts'].get(emp['name'], 0) + 1
            elif final_shift_name == 'N':
                episode_stats['night_counts'][emp['name']] = episode_stats['night_counts'].get(emp['name'], 0) + 1
                episode_stats['total_nights'] += 1
            elif final_shift_name == 'R':
                episode_stats['rest_counts'][emp['name']] = episode_stats['rest_counts'].get(emp['name'], 0) + 1
                episode_stats['total_rests'] += 1

        # Jolly Medico row
        if info.get('jolly_night_active', False):
            rows.append({
                'giorno': info['day'] + 1,
                'date': info['date'],
                'date_label': info['date_label'],
                'weekday': info['weekday_idx'],
                'weekday_label': info['weekday_label'],
                'dipendente': 'Jolly Medico',
                'ruolo': 'Medico',
                'azione_agente': '-',
                'turno_finale': 'N',
                'mask_M': False, 'mask_P': False, 'mask_N': False, 'mask_R': False,
                'violazioni': '', 'soft_violations': '', 'hard_overrides': '',
                'is_weekend': info['is_weekend'],
                'is_holiday': info['is_holiday'],
                'is_prefestive': info['is_prefestive'],
                'is_special_day': info['is_special_day'],
                'reward_total': reward,
                **{f'reward_{k}': v for k, v in info['reward_components'].items()},
            })

        done = terminated or truncated

    # Compute total hours and overtime
    shift_to_hours = {'M': 6, 'P': 6, 'N': 12, 'R': 0, 'MP': 12, 'J': 12, 'AP': 6}
    for emp in EMPLOYEES:
        emp_rows = [r for r in rows if r['dipendente'] == emp['name']]
        total_hours = sum(shift_to_hours.get(r['turno_finale'], 0) for r in emp_rows)
        episode_stats['hours_total'][emp['name']] = total_hours
        contract = CONTRACT_TYPES[emp['contract']]
        target_hours = contract['weekly_hours'] * 4
        episode_stats['overtime_by_employee'][emp['name']] = max(0.0, total_hours - target_hours)

    return rows, episode_stats

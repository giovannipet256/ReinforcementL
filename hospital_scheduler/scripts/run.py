"""
scripts/run.py
==============
Runs a single deterministic episode and saves the daily KPI log to CSV.

Usage:
    python hospital_scheduler/scripts/run.py
    python hospital_scheduler/scripts/run.py --model models/ppo_hospital_final --out outputs/run.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add the parent of hospital_scheduler to path (the refactored directory)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hospital_scheduler.config.settings import ALL_SHIFTS
from hospital_scheduler.env.environment import HospitalSchedulingEnv
from hospital_scheduler.env.masking import HospitalActionMasker
from hospital_scheduler.training.ferie import FERIE_LEVEL_5
from hospital_scheduler.utils.io import load_model


def run_episode(
    model_path: str = "models/ppo_hospital_final",
    output_csv_path: str = "outputs/run_base_daily_log.csv",
) -> pd.DataFrame:
    env = HospitalActionMasker(HospitalSchedulingEnv(ferie_table=FERIE_LEVEL_5))
    model = load_model(model_path)
    obs, info = env.reset()
    done = False
    rows = []
    day = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "day_index":                  day,
            "date":                       info["date"],
            "date_label":                 info["date_label"],
            "weekday_idx":                info["weekday_idx"],
            "weekday_label":              info["weekday_label"],
            "reward_total":               reward,
            "hard_overrides_today":       info["hard_overrides_today"],
            "soft_violations_today":      info["soft_violations_today"],
            "coverage_violations_today":  info["coverage_violations_today"],
            "jolly_used_today":           info["jolly_used_today"],
            "mp_used_today":              info["mp_used_today"],
            **{f"reward_{k}": v for k, v in info["reward_components"].items()},
            **{f"weighted_{k}": v for k, v in info["reward_components_weighted"].items()},
        })
        day += 1
        done = terminated or truncated

    df = pd.DataFrame(rows)
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run a single episode and save KPI CSV")
    parser.add_argument("--model", default="models/ppo_hospital_final",
                        help="Path to trained model")
    parser.add_argument("--out", default="outputs/run_base_daily_log.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    run_episode(args.model, args.out)


if __name__ == "__main__":
    main()

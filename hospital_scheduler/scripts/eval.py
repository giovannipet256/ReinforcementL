"""
scripts/eval.py
===============
Runs multiple episodes, selects the best by composite score, and saves to CSV.
Optionally runs for a specific month with annual ferie.

Usage:
    # Quick debug run (Aprile, level 5 ferie)
    python hospital_scheduler/scripts/eval.py

    # Specific month
    python hospital_scheduler/scripts/eval.py --year 2026 --month 6 --episodes 50

    # With a specific model
    python hospital_scheduler/scripts/eval.py --model models/best_level_5/best_model.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the parent of hospital_scheduler to path (the refactored directory)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hospital_scheduler.training.ferie import generate_annual_ferie
from hospital_scheduler.visualization.reports import run_debug, run_debug_month


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model over multiple episodes")
    parser.add_argument("--model", default="models/ppo_hospital_final",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--year", type=int, default=None,
                        help="Year for month-specific prediction (requires --month)")
    parser.add_argument("--month", type=int, default=None,
                        help="Month (1-12) for month-specific prediction (requires --year)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for annual ferie generation")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (auto-generated if omitted)")
    args = parser.parse_args()

    if args.year and args.month:
        annual_ferie = generate_annual_ferie(args.year, seed=args.seed)
        run_debug_month(
            year=args.year,
            month=args.month,
            annual_ferie=annual_ferie,
            model_path=args.model,
            output_path=args.out,
            n_episodes=args.episodes,
        )
    else:
        run_debug(
            model_path=args.model,
            output_path=args.out or "logs/debug_run.csv",
            n_episodes=args.episodes,
        )


if __name__ == "__main__":
    main()

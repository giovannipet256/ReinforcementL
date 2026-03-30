"""
scripts/train.py
================
Entry point for curriculum training.

Usage examples:
    # Train from scratch (from refactored directory)
    python hospital_scheduler/scripts/train.py

    # Resume from final trained model
    python hospital_scheduler/scripts/train.py --model models/ppo_hospital_final.zip

    # Resume from level 3 checkpoint
    python hospital_scheduler/scripts/train.py --model models/checkpoints/level_3/checkpoint_50000.zip --start-level 3

    # Start from level 4 with a model from level 3
    hs-train --model models/checkpoints/level_3/checkpoint_75000.zip --start-level 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the parent of hospital_scheduler to path (the refactored directory)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hospital_scheduler.training.curriculum import train_curriculum


def main():
    parser = argparse.ArgumentParser(description="Hospital scheduler curriculum training")
    parser.add_argument(
        "--model", default=None,
        help="Path to a pretrained model to resume from (optional)",
    )
    parser.add_argument(
        "--start-level", type=int, default=1,
        help="Starting curriculum level (1-5). Skips earlier levels. Default: 1",
    )
    parser.add_argument(
        "--no-masking", action="store_true",
        help="Use standard PPO instead of MaskablePPO",
    )
    args = parser.parse_args()

    train_curriculum(
        use_masking=not args.no_masking,
        pretrained_model_path=args.model,
        start_level=args.start_level,
    )


if __name__ == "__main__":
    main()

"""
hospital_scheduler
==================
RL-based hospital shift scheduling system.

Package layout:
    config/         - Settings, rewards, holidays
    env/            - Gymnasium environment and action masking
    training/       - Curriculum, callbacks, ferie generation
    evaluation/     - Episode runner and scoring metrics
    visualization/  - Streamlit dashboard and CSV reports
    utils/          - Model I/O and checkpoint utilities
    scripts/        - CLI entry points (train, run, eval)
"""

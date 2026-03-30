# Hospital Scheduler — RL Shift Planner

Reinforcement learning-based hospital staff scheduler. A MaskablePPO agent trained via curriculum learning assigns daily shifts to doctors and nurses over a 28-day episode, respecting labour law, coverage requirements, fairness, and employee preferences.

---

## Package structure

```
hospital_scheduler/
├── config/
│   ├── settings.py       # Employees, shifts, contracts, calendar
│   ├── rewards.py        # All reward / penalty tables and weights
│   └── holidays.py       # Italian holiday computation
│
├── env/
│   ├── environment.py    # HospitalSchedulingEnv (Gymnasium)
│   └── masking.py        # HospitalActionMasker wrapper
│
├── training/
│   ├── ferie.py          # Ferie / preference generation (random + annual)
│   ├── callbacks.py      # KPITrainingCallback, PeriodicCheckpointCallback
│   └── curriculum.py     # CURRICULUM config, make_env(), train_curriculum()
│
├── evaluation/
│   ├── evaluator.py      # evaluate_model(), run_single_episode()
│   └── metrics.py        # compute_episode_score() and sub-scores
│
├── visualization/
│   ├── reports.py        # run_debug(), run_debug_month() → CSV
│   └── dashboard.py      # Streamlit dashboard (streamlit run ...)
│
├── utils/
│   └── io.py             # load_model(), resolve_model_path(), get_latest_checkpoint()
│
└── scripts/
    ├── train.py          # CLI: curriculum training
    ├── run.py            # CLI: single episode → daily KPI CSV
    └── eval.py           # CLI: multi-episode evaluation → best CSV
```

---

## Installation

```bash
pip install -e .
# or without installing:
cd hospital_scheduler && pip install -r requirements.txt
```

---

## Usage

### Train from scratch
```bash
python -m hospital_scheduler.scripts.train
# or after pip install -e .:
hs-train
```

### Resume training from a checkpoint
```bash
hs-train --model models/ppo_hospital_final.zip
hs-train --model models/best_level_3/best_model.zip
hs-train --model models/checkpoints/level_2/checkpoint_75000.zip
```

### Run a single episode and save KPI CSV
```bash
hs-run
hs-run --model models/ppo_hospital_final --out outputs/my_run.csv
```

### Evaluate over multiple episodes (picks best)
```bash
hs-eval --episodes 100
hs-eval --year 2026 --month 6 --episodes 50   # specific month
```

### Launch the Streamlit dashboard
```bash
streamlit run hospital_scheduler/visualization/dashboard.py
```

---

## Key design decisions

| Concern | Location |
|---|---|
| All reward/penalty numbers | `config/rewards.py` |
| Employee roster, shift rules, contracts | `config/settings.py` |
| Italian holiday logic | `config/holidays.py` |
| Gym environment (step, reset, obs, masks) | `env/environment.py` |
| Action masking wrapper | `env/masking.py` |
| Ferie/preference generation | `training/ferie.py` |
| Curriculum stages and gates | `training/curriculum.py` |
| Training callbacks | `training/callbacks.py` |
| Evaluation and scoring | `evaluation/` |
| Dashboard and CSV reports | `visualization/` |
| Single source for model loading | `utils/io.py` |
| CLI entry points | `scripts/` |

---

## Adding new functionality

**New reward component** → add key to `config/rewards.py`, add handling in `env/environment.py:step()`.

**New employee or contract type** → edit `config/settings.py`.

**New curriculum level** → append to `CURRICULUM` list in `training/curriculum.py`.

**New evaluation metric** → add to `evaluation/metrics.py`.

**New CLI command** → add script in `scripts/`, register in `setup.py` entry_points.

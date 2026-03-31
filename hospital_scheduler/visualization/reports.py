from __future__ import annotations

import calendar as _cal
from datetime import date
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from hospital_scheduler.config.settings import (
    EMPLOYEES, CONTRACT_TYPES, MESI_IT,
    get_holidays_for_month,
)
from hospital_scheduler.env.environment import HospitalSchedulingEnv
from hospital_scheduler.env.masking import HospitalActionMasker
from hospital_scheduler.training.ferie import (
    FERIE_LEVEL_5, DEFAULT_PREFERENCES,
    generate_annual_ferie, get_ferie_for_month, generate_random_preferences,
)
from hospital_scheduler.evaluation.evaluator import run_single_episode
from hospital_scheduler.evaluation.metrics import DEFAULT_WEIGHTS, compute_episode_score
from hospital_scheduler.utils.io import load_model, get_latest_checkpoint


# Shift overlap map: shifts that conflict with each other
SHIFT_OVERLAPS = {
    'M': {'M', 'MP'},    # Morning overlaps with Morning and Morning-Afternoon
    'P': {'P', 'MP'},    # Afternoon overlaps with Afternoon and Morning-Afternoon
    'N': {'N'},          # Night only overlaps with Night
    'MP': {'M', 'P', 'MP'},  # Morning-Afternoon overlaps with all
    'R': set(),          # Rest doesn't conflict
    'AP': set(),         # Absence doesn't conflict
    'J': set(),          # Jolly (external) doesn't count
}


def is_schedule_valid(rows: List[Dict[str, Any]]) -> bool:
    """
    Check if a schedule is valid: both dirigenza (medici) and comparto (infermieri) 
    must have full daily coverage.
    
    Every day must have for BOTH roles:
    - Morning coverage (M or MP)
    - Afternoon coverage (P or MP)
    - Night coverage (N)
    
    MP counts as covering both morning and afternoon.
    
    Returns True if valid (full coverage every day for both roles), False otherwise.
    """
    df = pd.DataFrame(rows)
    
    # Check coverage for both Medici and Infermieri
    for role in ['Medico', 'Infermiere']:
        # Filter for role, excluding Jolly employees
        role_staff = df[(df['ruolo'] == role) & (~df['dipendente'].str.contains('Jolly', na=False))].copy()
        
        if role_staff.empty:
            return False  # No staff for this role = invalid
        
        # Check every day
        for day, group in role_staff.groupby('giorno'):
            shifts = group['turno_finale'].tolist()
            
            # Check morning coverage (need M or MP)
            has_morning = any(shift in ('M', 'MP') for shift in shifts)
            
            # Check afternoon coverage (need P or MP)
            has_afternoon = any(shift in ('P', 'MP') for shift in shifts)
            
            # Check night coverage (need N)
            has_night = any(shift == 'N' for shift in shifts)
            
            # If any part of the day is uncovered for this role, invalid
            if not (has_morning and has_afternoon and has_night):
                return False  # Missing coverage on this day for {role}
    
    return True  # Full coverage found for both roles


def run_debug(
    model_path: str = 'models/ppo_hospital_final',
    output_path: str = 'logs/debug_run.csv',
    n_episodes: int = 5,
    weights: Dict[str, float] | None = None,
) -> Path:
    """
    Runs n_episodes and saves the best one (by composite score) to CSV.
    Falls back to latest checkpoint if available.
    """
    env = HospitalActionMasker(HospitalSchedulingEnv(
        ferie_table=FERIE_LEVEL_5,
        preferences_table=DEFAULT_PREFERENCES.get(5, {}),
    ))

    latest_checkpoint, latest_timesteps = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"[CHECKPOINT] Caricando: {latest_checkpoint.name} ({latest_timesteps} timesteps)")
        model = load_model(str(latest_checkpoint))
    else:
        print(f"[MODEL] Carico il modello: {model_path}")
        model = load_model(model_path)

    if weights is None:
        weights = DEFAULT_WEIGHTS

    print(f"Eseguendo {n_episodes} episodi per selezionare il migliore...")
    best_score = -1.0
    best_rows: List[Dict[str, Any]] = []
    best_stats: Dict[str, Any] = {}
    scores_log = []

    for episode in range(n_episodes):
        print(f"  Episodio {episode + 1}/{n_episodes}...", end=" ")
        rows, stats = run_single_episode(env, model)
        score, detail = compute_episode_score(stats, rows, weights)
        scores_log.append({'episode': episode + 1, 'score': score, **detail})
        print(
            f"Score: {score:.3f} | Doc_cov: {detail['doctor_coverage_score']:.2f} "
            f"| OT_avg: {detail['avg_overtime']:.1f}h | Ore_ok: {detail['dipendenti_ore_ok']}/13"
        )
        if score > best_score:
            best_score = score
            best_rows = rows
            best_stats = stats

    print(f"\nMigliore: score {best_score:.3f}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best_rows).to_csv(out, index=False)

    scores_file = out.parent / f"{out.stem}_scores_log.csv"
    pd.DataFrame(scores_log).to_csv(scores_file, index=False)
    print(f"Salvato: {out}")
    print(f"Salvato log: {scores_file}")
    return out


def run_debug_month(
    year: int,
    month: int,
    annual_ferie: dict,
    model_path: str = 'models/ppo_hospital_final',
    output_path: str | None = None,
    n_episodes: int = 100,
    weights: Dict[str, float] | None = None,
    filter_valid_only: bool = True,
) -> Path:
    """
    Runs prediction for a specific month with the correct annual ferie and calendar.
    
    Args:
        filter_valid_only: If True, only keep episodes with no overlapping dirigenti on weekends.
                          If False, randomly select from all episodes.
    """
    days_in_month = _cal.monthrange(year, month)[1]
    start_date_val = date(year, month, 1)
    holidays = get_holidays_for_month(year, month)
    ferie_month = get_ferie_for_month(annual_ferie, year, month)
    prefs_month = generate_random_preferences(level=5, days_in_episode=days_in_month)

    env = HospitalActionMasker(HospitalSchedulingEnv(
        ferie_table=ferie_month,
        preferences_table=prefs_month,
        start_date=start_date_val,
        days_in_episode=days_in_month,
        holidays=holidays,
    ))

    latest_checkpoint, latest_timesteps = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"[CHECKPOINT] Caricando: {latest_checkpoint.name} ({latest_timesteps} timesteps)")
        model = load_model(str(latest_checkpoint))
    else:
        print(f"[MODEL] Carico il modello: {model_path}")
        model = load_model(model_path)

    if weights is None:
        weights = DEFAULT_WEIGHTS

    if output_path is None:
        output_path = f'logs/debug_run_{year}_{month:02d}.csv'

    filter_mode = "filtro attivo (copertura dirigenza completa)" if filter_valid_only else "selezione casuale"
    print(
        f"Predict {MESI_IT[month]} {year} "
        f"({days_in_month}gg, {len(holidays)} festivi, {len(ferie_month)} ferie) — {filter_mode}"
    )
    print(f"Eseguendo {n_episodes} episodi...")

    best_score = -1.0
    best_rows: List[Dict[str, Any]] = []
    all_rows_with_episode: List[Dict[str, Any]] = []
    scores_log = []
    valid_episodes = []
    all_episodes = []

    for episode in range(n_episodes):
        rows, stats = run_single_episode(env, model)
        score, detail = compute_episode_score(stats, rows, weights, n_days=days_in_month)
        
        is_valid = is_schedule_valid(rows)
        
        scores_log.append({
            'episode': episode + 1,
            'score': score,
            'valid': is_valid,
            **detail
        })
        
        all_episodes.append({'episode': episode + 1, 'score': score, 'rows': rows, 'valid': is_valid})
        
        if is_valid:
            valid_episodes.append(episode + 1)
        
        if (episode + 1) % 10 == 0 or episode == n_episodes - 1:
            status_str = f"| Validi: {len(valid_episodes)}" if filter_valid_only else ""
            print(f"  {episode + 1}/{n_episodes} {status_str}")

    # Select episodes based on filter mode
    if filter_valid_only:
        if valid_episodes:
            # Filter mode: use only valid episodes
            selected_episodes = [ep for ep in all_episodes if ep['valid']]
            print(f"\n✓ Modalità filtro: {len(selected_episodes)} episodi validi su {n_episodes}")
        else:
            # Fallback: no valid episodes, use best overall
            selected_episodes = sorted(all_episodes, key=lambda x: x['score'], reverse=True)[:1]
            print(f"⚠️  Nessun episodio valido trovato! Uso il migliore:")
    else:
        # Random mode: use all episodes
        selected_episodes = all_episodes
        print(f"\n✓ Modalità casuale: selezionando da {n_episodes} episodi")
    
    # Consolidate selected episodes into CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    for ep in selected_episodes:
        for row in ep['rows']:
            row['episode_id'] = ep['episode']
            all_rows_with_episode.append(row)
    
    if all_rows_with_episode:
        pd.DataFrame(all_rows_with_episode).to_csv(out, index=False)

    # Save valid episodes index
    scores_file = out.parent / f"{out.stem}_scores_log.csv"
    pd.DataFrame(scores_log).to_csv(scores_file, index=False)
    
    # Save list of valid episode IDs
    valid_episodes_file = out.parent / f"{out.stem}_valid_episodes.txt"
    with open(valid_episodes_file, 'w') as f:
        f.write(','.join(map(str, valid_episodes)))
    
    print(f"Salvato: {out}")
    print(f"Salvato log: {scores_file}")
    print(f"Salvato episodi validi: {valid_episodes_file}")
    return out

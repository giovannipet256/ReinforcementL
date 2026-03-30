from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from hospital_scheduler.config.settings import EMPLOYEES, CONTRACT_TYPES

DEFAULT_WEIGHTS = {
    'w_mattina': 0.10,
    'w_pomeriggio': 0.10,
    'w_notti': 0.10,
    'w_riposi': 0.10,
    'w_riposi_count': 0.05,
    'w_nights_count': 0.05,
    'w_ore_target': 0.25,
    'w_straordinari': 0.15,
    'w_copertura_medici': 0.30,
}


def compute_shift_distribution_score(shift_counts: Dict[str, int]) -> Tuple[float, float]:
    """
    Calculates uniformity score for a shift type across employees.
    Returns (score 0-1, std_value).
    """
    counts = list(shift_counts.values())
    if not counts:
        return 1.0, 0.0
    std_value = float(np.std(counts))
    if std_value == 0:
        return 1.0, std_value
    max_possible_std = max(counts) if counts else 1
    score = max(0.0, 1.0 - (std_value / max_possible_std))
    return score, std_value


def compute_doctor_coverage_score(
    rows: List[Dict[str, Any]], n_days: int = 30
) -> Tuple[float, int, int]:
    """
    Calculates whether each day has doctor coverage for M, P, N.
    Returns (score 0-1, slots_covered, total_slots).
    """
    coverage: Dict[Tuple[int, str], List[str]] = {}
    for row in rows:
        if row['ruolo'] != 'Medico':
            continue
        day = row['giorno']
        shift = row['turno_finale']
        if shift in ('M', 'P', 'N'):
            slots = [shift]
        elif shift == 'MP':
            slots = ['M', 'P']
        elif shift == 'J':
            slots = ['N']
        else:
            slots = []
        for s in slots:
            coverage.setdefault((day, s), []).append(row['dipendente'])

    total_slots = n_days * 3
    covered_slots = sum(
        1
        for day in range(1, n_days + 1)
        for shift in ('M', 'P', 'N')
        if coverage.get((day, shift))
    )
    return covered_slots / max(1, total_slots), covered_slots, total_slots


def compute_episode_score(
    episode_stats: Dict[str, Any],
    rows: List[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
    n_days: int = 30,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes a composite quality score for an episode based on:
    1. Uniform distribution of morning/afternoon/night/rest shifts
    2. Absolute night and rest counts
    3. Hours on target
    4. Overtime (max 8h average)
    5. Doctor coverage every day for M, P, N
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    mattina_score,    std_mattine   = compute_shift_distribution_score(episode_stats['morning_counts'])
    pomeriggio_score, std_pomeriggi = compute_shift_distribution_score(episode_stats['afternoon_counts'])
    notti_score,      std_notti     = compute_shift_distribution_score(episode_stats['night_counts'])
    riposi_score,     std_riposi    = compute_shift_distribution_score(episode_stats['rest_counts'])

    n_weeks = n_days / 7

    total_rests = episode_stats.get('total_rests', 0)
    avg_rests = total_rests / len(EMPLOYEES) if EMPLOYEES else 0
    target_rests_max = max(1, round(n_weeks) + 1)
    rests_count_score = max(0.0, 1.0 - (avg_rests / target_rests_max))

    total_nights = episode_stats.get('total_nights', 0)
    avg_nights = total_nights / len(EMPLOYEES) if EMPLOYEES else 0
    target_nights_min = max(1, round(n_weeks))
    nights_count_score = min(1.0, avg_nights / target_nights_min)

    hours_total = episode_stats['hours_total']
    dipendenti_ok = 0
    for emp in EMPLOYEES:
        contract = CONTRACT_TYPES[emp['contract']]
        target_hours = contract['weekly_hours'] * n_weeks
        if hours_total.get(emp['name'], 0) >= target_hours * 0.9:
            dipendenti_ok += 1
    hours_score = dipendenti_ok / max(1, len(EMPLOYEES))

    overtime_values = list(episode_stats['overtime_by_employee'].values())
    avg_overtime = sum(overtime_values) / len(EMPLOYEES) if EMPLOYEES else 0.0
    total_overtime = sum(overtime_values)
    if avg_overtime <= 8.0:
        overtime_score = max(0.0, 1.0 - (avg_overtime / 8.0))
    else:
        overtime_score = max(0.0, 1.0 - ((avg_overtime - 8.0) / 8.0))

    doctor_coverage_score, covered_slots, total_slots = compute_doctor_coverage_score(rows, n_days)

    final_score = (
        mattina_score    * weights['w_mattina']
        + pomeriggio_score * weights['w_pomeriggio']
        + notti_score      * weights['w_notti']
        + riposi_score     * weights['w_riposi']
        + rests_count_score  * weights['w_riposi_count']
        + nights_count_score * weights['w_nights_count']
        + hours_score      * weights['w_ore_target']
        + overtime_score   * weights['w_straordinari']
        + doctor_coverage_score * weights['w_copertura_medici']
    )

    detail = {
        'mattina_score': mattina_score,
        'pomeriggio_score': pomeriggio_score,
        'notti_score': notti_score,
        'riposi_score': riposi_score,
        'rests_count_score': rests_count_score,
        'nights_count_score': nights_count_score,
        'hours_score': hours_score,
        'overtime_score': overtime_score,
        'doctor_coverage_score': doctor_coverage_score,
        'std_mattine': std_mattine,
        'std_pomeriggi': std_pomeriggi,
        'std_notti': std_notti,
        'std_riposi': std_riposi,
        'total_overtime': total_overtime,
        'avg_overtime': avg_overtime,
        'dipendenti_ore_ok': dipendenti_ok,
        'doctor_coverage_slots': f"{covered_slots}/{total_slots}",
        'reward_totale': episode_stats['total_reward'],
    }

    return final_score, detail

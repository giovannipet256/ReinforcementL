from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np

from hospital_scheduler.config.settings import (
    EMPLOYEES, N_EMPLOYEES, DAYS_IN_EPISODE,
    PREFERENCE_TYPES, MAX_PREFERENCES_PER_EMPLOYEE,
    MAX_ANNUAL_FERIE, SUMMER_BLOCK_DAYS, SUMMER_MONTHS,
)

# ============================================================
# Static ferie tables (curriculum defaults)
# ============================================================
FERIE_LEVEL_1: Dict[tuple, bool] = {}
FERIE_LEVEL_2: Dict[tuple, bool] = {(3, 4): True, (7, 11): True}
FERIE_LEVEL_3: Dict[tuple, bool] = {(0, 2): True, (1, 2): True, (3, 4): True, (4, 9): True, (6, 17): True}
FERIE_LEVEL_4: Dict[tuple, bool] = {**FERIE_LEVEL_3, (2, 12): True, (10, 19): True}
FERIE_LEVEL_5: Dict[tuple, bool] = {**FERIE_LEVEL_4, (5, 6): True, (8, 22): True, (11, 24): True}

FERIE_DEFAULTS: Dict[int, Dict[tuple, bool]] = {
    1: FERIE_LEVEL_1,
    2: FERIE_LEVEL_2,
    3: FERIE_LEVEL_3,
    4: FERIE_LEVEL_4,
    5: FERIE_LEVEL_5,
}

# ============================================================
# Fixed default preferences for evaluation (one per level)
# ============================================================
DEFAULT_PREFERENCES: Dict[int, Dict[tuple, str]] = {
    1: {},
    2: {},
    3: {(0, 10): 'no_notte', (5, 15): 'no_mattina'},
    4: {(0, 10): 'no_notte', (1, 5): 'no_pomeriggio', (5, 15): 'no_mattina', (8, 20): 'no_MP'},
    5: {
        (0, 10): 'no_notte', (1, 5): 'no_pomeriggio', (2, 18): 'no_pomeriggio_notte',
        (3, 7): 'no_mattina', (5, 15): 'no_mattina', (8, 20): 'no_MP',
        (9, 3): 'no_notte', (11, 25): 'no_pomeriggio',
    },
}


def generate_random_ferie(level: int, seed: Optional[int] = None) -> Dict[tuple, bool]:
    """
    Generates a random ferie table consistent with curriculum difficulty.
    Level 1: 0-2 days  |  Level 2: 2-5  |  Level 3: 5-10
    Level 4: 10-15     |  Level 5: 15-25
    """
    rng = np.random.RandomState(seed)
    ferie_ranges = {1: (0, 2), 2: (2, 5), 3: (5, 10), 4: (10, 15), 5: (15, 25)}
    min_days, max_days = ferie_ranges.get(level, (0, 2))
    total_days = rng.randint(min_days, max_days + 1)

    ferie_table: Dict[tuple, bool] = {}
    assigned = 0
    while assigned < total_days:
        emp_id = rng.randint(0, N_EMPLOYEES)
        day = rng.randint(0, DAYS_IN_EPISODE)
        key = (emp_id, day)
        if key not in ferie_table:
            ferie_table[key] = True
            assigned += 1
    return ferie_table


def generate_annual_ferie(year: int, seed: Optional[int] = None) -> dict:
    """
    Generates the annual ferie plan for all employees.
    - Each employee gets SUMMER_BLOCK_DAYS consecutive days in a summer month,
      distributed evenly (~3-4 per month).
    - Remaining days (up to MAX_ANNUAL_FERIE=30) are scattered randomly.

    Returns: {(emp_id, date): True}
    """
    rng = np.random.RandomState(seed)
    annual_ferie: dict = {}
    emp_ferie_count = {emp['id']: 0 for emp in EMPLOYEES}
    emp_ferie_dates: dict = {emp['id']: set() for emp in EMPLOYEES}

    # Step 1: 15 consecutive summer days for each employee
    emp_ids = [emp['id'] for emp in EMPLOYEES]
    rng.shuffle(emp_ids)

    per_month = len(emp_ids) // len(SUMMER_MONTHS)
    remainder = len(emp_ids) % len(SUMMER_MONTHS)
    month_assignments = []
    idx = 0
    for i, month in enumerate(SUMMER_MONTHS):
        count = per_month + (1 if i < remainder else 0)
        for _ in range(count):
            month_assignments.append((emp_ids[idx], month))
            idx += 1

    for emp_id, month in month_assignments:
        days_in_month = _cal.monthrange(year, month)[1]
        latest_start = max(1, days_in_month - 4)
        start_day = rng.randint(1, latest_start + 1)
        block_start = date(year, month, start_day)
        for offset in range(SUMMER_BLOCK_DAYS):
            d = block_start + timedelta(days=offset)
            if d.year != year:
                break
            annual_ferie[(emp_id, d)] = True
            emp_ferie_dates[emp_id].add(d)
            emp_ferie_count[emp_id] += 1

    # Step 2: Scattered random days (up to MAX_ANNUAL_FERIE)
    for emp in EMPLOYEES:
        emp_id = emp['id']
        remaining = MAX_ANNUAL_FERIE - emp_ferie_count[emp_id]
        extra_days = rng.randint(max(0, remaining - 5), remaining + 1)
        assigned = 0
        attempts = 0
        while assigned < extra_days and attempts < extra_days * 50:
            attempts += 1
            month = rng.randint(1, 13)
            dim = _cal.monthrange(year, month)[1]
            day = rng.randint(1, dim + 1)
            d = date(year, month, day)
            if d in emp_ferie_dates[emp_id]:
                continue
            annual_ferie[(emp_id, d)] = True
            emp_ferie_dates[emp_id].add(d)
            emp_ferie_count[emp_id] += 1
            assigned += 1

    return annual_ferie


def get_ferie_for_month(annual_ferie: dict, year: int, month: int) -> dict:
    """
    Extracts ferie for a specific month from the annual plan.
    Converts from (emp_id, date) to (emp_id, day_in_month) 0-based.
    """
    first = date(year, month, 1)
    result = {}
    for (emp_id, d), v in annual_ferie.items():
        if isinstance(d, date) and d.year == year and d.month == month:
            result[(emp_id, (d - first).days)] = v
    return result


def generate_random_preferences(
    level: int,
    seed: Optional[int] = None,
    days_in_episode: Optional[int] = None,
) -> dict:
    """
    Generates random employee preferences consistent with curriculum level.
    Level 1: 0  |  Level 2: 0-1  |  Level 3: 1-3
    Level 4: 3-6  |  Level 5: 6-10
    Max 2 preferences per employee.

    Returns: {(emp_id, day): pref_type_str}
    """
    n_days = days_in_episode if days_in_episode is not None else DAYS_IN_EPISODE
    rng = np.random.RandomState(seed)
    pref_ranges = {1: (0, 0), 2: (0, 1), 3: (1, 3), 4: (3, 6), 5: (6, 10)}
    min_prefs, max_prefs = pref_ranges.get(level, (0, 0))
    total_prefs = rng.randint(min_prefs, max_prefs + 1)

    pref_type_names = list(PREFERENCE_TYPES.keys())
    preferences: dict = {}
    emp_pref_count = {emp['id']: 0 for emp in EMPLOYEES}
    assigned = 0
    attempts = 0
    max_attempts = total_prefs * 20

    while assigned < total_prefs and attempts < max_attempts:
        attempts += 1
        emp_id = rng.randint(0, N_EMPLOYEES)
        if emp_pref_count[emp_id] >= MAX_PREFERENCES_PER_EMPLOYEE:
            continue
        day = rng.randint(0, n_days)
        key = (emp_id, day)
        if key in preferences:
            continue
        pref_type = pref_type_names[rng.randint(0, len(pref_type_names))]
        preferences[key] = pref_type
        emp_pref_count[emp_id] += 1
        assigned += 1

    return preferences

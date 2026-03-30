from __future__ import annotations

from datetime import date, timedelta

from .holidays import get_holidays_for_month

# ============================================================
# Shifts
# ============================================================
AGENT_SHIFTS = {0: 'M', 1: 'P', 2: 'N', 3: 'R'}
ALL_SHIFTS = {0: 'M', 1: 'P', 2: 'N', 3: 'R', 4: 'MP', 5: 'J', 6: 'AP'}
SHIFT_NAME_TO_ID = {v: k for k, v in ALL_SHIFTS.items()}

# Real working windows (hours from day start). Night ends next day.
SHIFT_TIME_WINDOWS = {
    0: (8, 14),   # M
    1: (14, 20),  # P
    2: (20, 32),  # N = 20:00 -> 08:00 next day
    3: None,      # R
    4: (8, 20),   # MP
    5: (20, 32),  # Jolly night coverage
    6: (8, 14),   # AP (formazione 6h)
}
SHIFT_HOURS = {0: 6, 1: 6, 2: 12, 3: 0, 4: 12, 5: 12, 6: 6}
SHIFT_COVERS = {0: ['M'], 1: ['P'], 2: ['N'], 3: [], 4: ['M', 'P'], 5: ['N'], 6: []}
COVERAGE_SLOTS = ['M', 'P', 'N']

# ============================================================
# Employee types and contracts
# ============================================================
EMPLOYEE_TYPE_MEDICO = 0
EMPLOYEE_TYPE_INFERMIERE = 1
ROLE_NAMES = {0: 'Medico', 1: 'Infermiere'}

CONTRACT_TYPES = {
    'DIR': {
        'name': 'Dirigenza',
        'weekly_hours': 38,
        'max_weekly_overtime': 10,
        'employee_type': EMPLOYEE_TYPE_MEDICO,
        'min_daily_rest_hours': 11,
        'max_daily_hours': 12,
    },
    'COMP': {
        'name': 'Comparto',
        'weekly_hours': 36,
        'max_weekly_overtime': 12,
        'employee_type': EMPLOYEE_TYPE_INFERMIERE,
        'min_daily_rest_hours': 11,
        'max_daily_hours': 12,
    },
}

MAX_MONTHLY_OVERTIME_PROXY = 32
AP_HOURS = 6
AP_PROBABILITY_BASE = 0.25
WEEKLY_REST_ENABLED = True

# ============================================================
# Employees
# ============================================================
# EMPLOYEES = [
#     {'id': 0,  'name': 'Dr. Rossi',   'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 1,  'name': 'Dr. Bianchi', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 2,  'name': 'Dr. Ferrari', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 3,  'name': 'Dr. Greco',   'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 4,  'name': 'Dr. Mancini', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 5,  'name': 'Dr. Romano',  'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 6,  'name': 'Dr. Colombo', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
#     {'id': 7,  'name': 'Inf. Verdi',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 8,  'name': 'Inf. Neri',   'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 9,  'name': 'Inf. Marino', 'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 10, 'name': 'Inf. Russo',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 11, 'name': 'Inf. Conti',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 12, 'name': 'Inf. Bruno',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 13, 'name': 'Inf. Serra',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 14, 'name': 'Inf. Longo',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 15, 'name': 'Inf. Gallo',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
#     {'id': 16, 'name': 'Inf. Costa',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
# ]

EMPLOYEES = [
    {'id': 0,  'name': 'Dr. Rossi',   'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
    {'id': 1,  'name': 'Dr. Bianchi', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
    {'id': 2,  'name': 'Dr. Ferrari', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
    {'id': 3,  'name': 'Dr. Greco',   'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
    {'id': 4,  'name': 'Dr. Mancini', 'type': EMPLOYEE_TYPE_MEDICO,     'contract': 'DIR'},
    {'id': 5,  'name': 'Inf. Verdi',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 6,  'name': 'Inf. Neri',   'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 7,  'name': 'Inf. Marino', 'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 8,  'name': 'Inf. Russo',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 9,  'name': 'Inf. Conti',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 10, 'name': 'Inf. Bruno',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 11, 'name': 'Inf. Serra',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
    {'id': 12, 'name': 'Inf. Longo',  'type': EMPLOYEE_TYPE_INFERMIERE, 'contract': 'COMP'},
]
N_EMPLOYEES = len(EMPLOYEES)

JOLLY = [
    {'id': 98, 'name': 'Jolly Medico', 'type': EMPLOYEE_TYPE_MEDICO},
    {'id': 99, 'name': 'Jolly Infermiere', 'type': EMPLOYEE_TYPE_INFERMIERE},
]

# ============================================================
# Schedule / calendar
# ============================================================
DAYS_IN_EPISODE = 30
WEEK_LEN = 7
START_DATE = date(2026, 4, 1)
ITALIAN_WEEKDAY_LABELS = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']

# Configurable holiday set for the simulated month.
HOLIDAYS = get_holidays_for_month(START_DATE.year, START_DATE.month)

MESI_IT = {
    1: 'Gennaio', 2: 'Febbraio', 3: 'Marzo', 4: 'Aprile',
    5: 'Maggio', 6: 'Giugno', 7: 'Luglio', 8: 'Agosto',
    9: 'Settembre', 10: 'Ottobre', 11: 'Novembre', 12: 'Dicembre',
}

MAX_ANNUAL_FERIE = 30
SUMMER_BLOCK_DAYS = 15
SUMMER_MONTHS = [6, 7, 8, 9]

# ============================================================
# Coverage and rotation rules
# ============================================================
COVERAGE_REQUIREMENTS = {
    'M': {'Medico': 1, 'Infermiere': 2},
    'P': {'Medico': 1, 'Infermiere': 2},
    'N': {'Medico': 1, 'Infermiere': 1},
}

CONSECUTIVE_LIMITS = {
    'Medico':     {'M': 2, 'P': 2, 'N': 1, 'R': 2, 'MP': 1},
    'Infermiere': {'M': 3, 'P': 3, 'N': 2, 'R': 2, 'MP': 1},
}

SPECIAL_DAY_TYPES = ('SAT', 'SUN', 'HOL', 'PRE')
SPECIAL_DAY_ALTERNATION = {
    'enabled': True,
    'apply_to_roles': ['Medico', 'Infermiere'],
    'apply_to_shift_types': ['M', 'P', 'N'],
    'hard_if_alternative_exists': True,
    'soft_penalty_if_inevitable': -4.0,
    'soft_penalty_if_avoidable': -12.0,
    'bonus_if_respected': 0.20,
}

# ============================================================
# Employee preferences (soft constraints)
# ============================================================
PREFERENCE_TYPES = {
    'no_mattina':          {0},
    'no_pomeriggio':       {1},
    'no_notte':            {2},
    'no_pomeriggio_notte': {1, 2},
    'no_MP':               {0, 1, 4},
}
MAX_PREFERENCES_PER_EMPLOYEE = 2
MONTHLY_HOURS_TOLERANCE = 8.0

# ============================================================
# Calendar helpers
# ============================================================
def get_date_for_day(day_index: int, start_date: date = START_DATE) -> date:
    return start_date + timedelta(days=int(day_index))


def get_weekday_idx(day_index: int, start_date: date = START_DATE) -> int:
    return get_date_for_day(day_index, start_date).weekday()


def get_weekday_label(day_index: int, start_date: date = START_DATE) -> str:
    return ITALIAN_WEEKDAY_LABELS[get_weekday_idx(day_index, start_date)]


def is_holiday(day_index: int, start_date: date = START_DATE, holidays: set = None) -> bool:
    hols = holidays if holidays is not None else HOLIDAYS
    return get_date_for_day(day_index, start_date) in hols


def is_weekend(day_index: int, start_date: date = START_DATE) -> bool:
    return get_weekday_idx(day_index, start_date) >= 5


def is_prefestive(day_index: int, start_date: date = START_DATE, holidays: set = None,
                  days_in_episode: int = DAYS_IN_EPISODE) -> bool:
    nxt = day_index + 1
    if nxt >= days_in_episode:
        return False
    return is_holiday(nxt, start_date, holidays) or is_weekend(nxt, start_date)


def get_special_day_types(day_index: int, start_date: date = START_DATE,
                          holidays: set = None, days_in_episode: int = DAYS_IN_EPISODE):
    types = []
    wd = get_weekday_idx(day_index, start_date)
    if wd == 5:
        types.append('SAT')
    if wd == 6:
        types.append('SUN')
    if is_holiday(day_index, start_date, holidays):
        types.append('HOL')
    if is_prefestive(day_index, start_date, holidays, days_in_episode):
        types.append('PRE')
    return tuple(types)


def get_date_label(day_index: int, start_date: date = START_DATE) -> str:
    d = get_date_for_day(day_index, start_date)
    return f"{get_weekday_label(day_index, start_date)} {d.strftime('%d/%m/%Y')}"

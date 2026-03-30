from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from multiprocessing.util import info
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hospital_scheduler.config.settings import (
    AGENT_SHIFTS, ALL_SHIFTS, SHIFT_HOURS, SHIFT_COVERS, SHIFT_TIME_WINDOWS, COVERAGE_SLOTS,
    EMPLOYEES, N_EMPLOYEES, CONTRACT_TYPES, AP_PROBABILITY_BASE,
    EMPLOYEE_TYPE_MEDICO, EMPLOYEE_TYPE_INFERMIERE, ROLE_NAMES,
    DAYS_IN_EPISODE, WEEK_LEN, WEEKLY_REST_ENABLED,
    START_DATE, HOLIDAYS, ITALIAN_WEEKDAY_LABELS,
    COVERAGE_REQUIREMENTS, CONSECUTIVE_LIMITS,
    MAX_MONTHLY_OVERTIME_PROXY, SPECIAL_DAY_ALTERNATION,
    PREFERENCE_TYPES, MONTHLY_HOURS_TOLERANCE,
)
from hospital_scheduler.config.rewards import (
    REWARD_DAILY, PENALTY_DAILY, REWARD_WEEKLY, PENALTY_WEEKLY,
    REWARD_MONTHLY, PENALTY_MONTHLY, REWARD_WEIGHTS_BASE,
    ROTATION_IMBALANCE_ALPHA, COVERAGE_RESIDUAL_MAX_PERCENTAGES,
    SHIFT_PROPORTIONALITY_LAMBDAS,
)

ROLE_ORDER = [ROLE_NAMES[EMPLOYEE_TYPE_MEDICO], ROLE_NAMES[EMPLOYEE_TYPE_INFERMIERE]]
SHIFT_IDS = [0, 1, 2, 3]
COVER_TO_ACTION = {'M': 0, 'P': 1, 'N': 2}
SPECIAL_SHIFT_NAMES = set(SPECIAL_DAY_ALTERNATION['apply_to_shift_types'])
SPECIAL_ROLES = set(SPECIAL_DAY_ALTERNATION['apply_to_roles'])


class HospitalSchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ferie_table=None,
        reward_weights=None,
        ap_probability_dirigenza: float = AP_PROBABILITY_BASE,
        preferences_table=None,
        start_date=None,
        days_in_episode=None,
        holidays=None,
    ):
        super().__init__()
        self._start_date = start_date if start_date is not None else START_DATE
        self._days_in_episode = days_in_episode if days_in_episode is not None else DAYS_IN_EPISODE
        self._holidays = holidays if holidays is not None else HOLIDAYS
        self.ferie_table = ferie_table or {}
        self.preferences_table = preferences_table or {}
        self.reward_weights = dict(REWARD_WEIGHTS_BASE)
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.ap_probability_dirigenza = float(ap_probability_dirigenza)

        self.action_space = spaces.MultiDiscrete([len(AGENT_SHIFTS)] * N_EMPLOYEES)

        self.global_feature_size = 1 + 7 + 4 + 6
        self.employee_feature_size = 17
        obs_size = self.global_feature_size + N_EMPLOYEES * self.employee_feature_size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.schedule = None
        self.hours_week = None
        self.hours_month = None
        self.last_shift = None
        self.last_shift_end = None
        self.same_shift_streak = None
        self.consecutive_work_days = None
        self.consecutive_rest_days = None
        self.shift_counts = None
        self.special_day_work_counts = None
        self.weekend_work_counts = None  #conta i weekend lavorati
        self.last_weekend_shift_coverage = None  # Track which emp covered each weekend shift
        self.last_emp_reward = None
        self.current_day = 0
        self.total_jolly = 0
        self.total_mp = 0
        self.total_hard_overrides = 0
        self.total_soft_violations = 0
        self.total_coverage_violations = 0
        self.jolly_used_week = None
        self.ap_active_weeks = None  # Subset of {1,2,4} drawn each episode (2 or 3 weeks, 50/50)
        self.ap_week_plan = None
        self.ap_day_infermieri = None  # Single AP day per infermiere, drawn once at episode (1/3 probability)
        self.ap_assigned_sundays = {}  # Track {emp_id: sunday_start_of_interval} to limit 1 AP per sunday-sunday interval
        self.ap_count_monthly = {}  # Track {emp_id: count} for monthly AP limit (max 3 per dirigente)
        self.rest_week_plan = None
        self.ap_today = None
        self.rest_today = None
        self.final_shifts_today = None
        self.monthly_kpi_by_role = None
        self.monthly_kpi_by_employee = None
        self.jolly_days = None
        self.last_special_assignment_by_role_shift_type = None
        self.weekly_coverage_checks = None  # Track daily coverage check scores per week

    # ------------------------------------------------------------------
    # Calendar helpers
    # ------------------------------------------------------------------
    def _date_for_day(self, day_index: int):
        return self._start_date + timedelta(days=int(day_index))

    def _weekday_idx(self, day_index: int) -> int:
        return self._date_for_day(day_index).weekday()

    def _weekday_label(self, day_index: int) -> str:
        return ITALIAN_WEEKDAY_LABELS[self._weekday_idx(day_index)]

    def _is_holiday(self, day_index: int) -> bool:
        return self._date_for_day(day_index) in self._holidays

    def _is_weekend(self, day_index: int) -> bool:
        return self._weekday_idx(day_index) >= 5

    def _is_weekday(self, day_index: int) -> bool:
        """Returns True if day is a weekday (Mon-Fri) and not a holiday."""
        return not self._is_weekend(day_index) and not self._is_holiday(day_index)

    def _get_sunday_interval(self, day_index: int) -> tuple:
        """
        Returns (first_sunday, last_sunday) of the sunday-to-sunday interval containing this day.
        Interval goes from one Sunday through the following Saturday.
        Returns tuple (first_sunday_day_index, last_sunday_day_index)
        """
        current_weekday = self._weekday_idx(day_index)  # 0=Mon ... 6=Sun
        
        # Calculate days to go back to reach previous Sunday
        if current_weekday == 6:
            # Today is Sunday: this Sunday is the start of the interval
            days_to_prev_sunday = 0
        else:
            # Days from current day back to Sunday: (current_weekday + 1) gives days forward
            # So we need (7 - (current_weekday + 1)) = 6 - current_weekday
            days_to_prev_sunday = (current_weekday + 1) % 7
            if days_to_prev_sunday == 0:
                days_to_prev_sunday = 7
        
        # The interval starts at the previous Sunday (or today if today is Sunday)
        first_sunday = day_index - days_to_prev_sunday
        last_sunday = first_sunday + 6  # 6 days after, brings us to next Sunday
        
        return (max(first_sunday, -8), min(last_sunday, self._days_in_episode - 1))

    def _is_prefestive(self, day_index: int) -> bool:
        nxt = day_index + 1
        if nxt >= self._days_in_episode:
            return False
        return self._is_holiday(nxt) or self._is_weekend(nxt)

    def _worked_last_weekend(self, emp_idx: int, day: int) -> bool:
        """Check if employee worked last weekend (Saturday or Sunday)."""
        current_weekday = self._weekday_idx(day)  # 0=Mon ... 5=Sat, 6=Sun

        if current_weekday == 5:  # Current day is Saturday
            last_sat = day - 7
            last_sun = day - 6
        elif current_weekday == 6:  # Current day is Sunday
            last_sat = day - 8
            last_sun = day - 7
        else:
            return False

        emp_id = EMPLOYEES[emp_idx]['id']
        shift_sat = self.schedule.get(last_sat, {}).get(emp_id, -1)
        shift_sun = self.schedule.get(last_sun, {}).get(emp_id, -1)

        return (shift_sat not in (3, 6, -1)) or (shift_sun not in (3, 6, -1))

    def _check_consecutive_weekends(self, emp_idx: int, current_day: int, current_shift: int) -> tuple:
        """
        Check if employee worked last weekend and is working this Saturday (consecutive weekends).
        Called on Saturday (weekday_idx == 5).
        Returns: (worked_prev_weekend: bool, working_this_saturday: bool)
        
        Example:
        - If worked last Sat/Sun AND working this Saturday → (True, True) → penalty
        - If worked last Sat/Sun AND resting this Saturday → (True, False) → reward
        - If didn't work last weekend → (False, False or True) → neutral/no bonus
        """
        current_weekday = self._weekday_idx(current_day)
        
        # Only check on Saturday (weekday 5)
        if current_weekday != 5:
            return False, False
        
        emp_id = EMPLOYEES[emp_idx]['id']
        
        # Get last weekend (Saturday and Sunday of previous week)
        last_sat = current_day - 7
        last_sun = current_day - 6
        
        # Check if employee worked last weekend
        shift_last_sat = self.schedule.get(last_sat, {}).get(emp_id, -1)
        shift_last_sun = self.schedule.get(last_sun, {}).get(emp_id, -1)
        worked_last_weekend = (shift_last_sat not in (3, 6, -1)) or (shift_last_sun not in (3, 6, -1))
        
        # Check if working this Saturday (current shift parameter)
        working_this_saturday = current_shift not in (3, 6)  # Rest (3) and AP (6) are not work
        
        return worked_last_weekend, working_this_saturday

    def _get_saturday_shift_medico(self, emp_idx: int, current_day: int) -> int | None:
        """
        If current day is Sunday, return what shift the employee had on Saturday.
        Returns shift_id or None if not applicable.
        """
        current_weekday = self._weekday_idx(current_day)  # 0=Mon ... 5=Sat, 6=Sun
        
        if current_weekday != 6:  # Only for Sunday
            return None
        
        saturday = current_day - 1
        emp_id = EMPLOYEES[emp_idx]['id']
        shift = self.schedule.get(saturday, {}).get(emp_id, -1)
        
        return shift if shift != -1 else None

    def _special_day_types(self, day_index: int):
        types = []
        wd = self._weekday_idx(day_index)
        if wd == 5:
            types.append('SAT')
        if wd == 6:
            types.append('SUN')
        if self._is_holiday(day_index):
            types.append('HOL')
        if self._is_prefestive(day_index):
            types.append('PRE')
        return tuple(types)

    def _date_label(self, day_index: int) -> str:
        d = self._date_for_day(day_index)
        return f"{self._weekday_label(day_index)} {d.strftime('%d/%m/%Y')}"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_reward_weights(self, reward_weights: Dict[str, float]):
        self.reward_weights.update(reward_weights)

    def get_reward_weights(self) -> Dict[str, float]:
        return dict(self.reward_weights)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.schedule = {day: {emp['id']: -1 for emp in EMPLOYEES} for day in range(self._days_in_episode)}
        
        # Get employee indices by type
        dirigenza_indices = [emp['id'] for emp in EMPLOYEES if emp['type'] == EMPLOYEE_TYPE_MEDICO]
        comparto_indices = [emp['id'] for emp in EMPLOYEES if emp['type'] == EMPLOYEE_TYPE_INFERMIERE]
        
        # Shuffle for variety in weekend pair coverage assignments
        shuffled_dirigenza = dirigenza_indices.copy()
        shuffled_comparto = comparto_indices.copy()
        self.np_random.shuffle(shuffled_dirigenza)
        self.np_random.shuffle(shuffled_comparto)
        
        # Initialize virtual Saturday (day -8) with randomized coverage
        virtual_saturday = -8
        self.schedule[virtual_saturday] = {}
        
        # Initialize all employees to rest
        for emp in EMPLOYEES:
            self.schedule[virtual_saturday][emp['id']] = 3  # Rest by default
        
        # Assign directors (medici) to M, P, N shifts with randomization
        self.schedule[virtual_saturday][shuffled_dirigenza[0]] = 0  # M
        self.schedule[virtual_saturday][shuffled_dirigenza[1]] = 1  # P
        self.schedule[virtual_saturday][shuffled_dirigenza[2]] = 2  # N
        
        # Assign staff (infermieri) with MP optimization using shuffled indices
        self.schedule[virtual_saturday][shuffled_comparto[0]] = 4  # MP (covers both M and P)
        self.schedule[virtual_saturday][shuffled_comparto[1]] = 4  # MP (covers both M and P)
        self.schedule[virtual_saturday][shuffled_comparto[2]] = 2  # N
        
        # Initialize virtual Sunday (day -7) with optimal coverage for rotation constraint
        # Covers M, P, N with minimum staff: max 3 dirigenza (medici), max 3 comparto (infermieri)
        # Uses MP shift (shift 4) for infermieri to cover both M and P efficiently
        virtual_sunday = -7
        self.schedule[virtual_sunday] = {}
        
        # Initialize all employees to rest
        for emp in EMPLOYEES:
            self.schedule[virtual_sunday][emp['id']] = 3  # Rest by default
        
        # Assign directors (medici) to M, P, N shifts (1 per shift)
        self.schedule[virtual_sunday][dirigenza_indices[0]] = 0  # M
        self.schedule[virtual_sunday][dirigenza_indices[1]] = 1  # P
        self.schedule[virtual_sunday][dirigenza_indices[2]] = 2  # N
        
        # Assign staff (infermieri) with MP optimization
        self.schedule[virtual_sunday][comparto_indices[0]] = 4  # MP (covers both M and P)
        self.schedule[virtual_sunday][comparto_indices[1]] = 4  # MP (covers both M and P)
        self.schedule[virtual_sunday][comparto_indices[2]] = 2  # N
        
        self.hours_week = np.zeros(N_EMPLOYEES, dtype=np.float32)
        self.hours_month = np.zeros(N_EMPLOYEES, dtype=np.float32)
        self.last_shift = np.full(N_EMPLOYEES, 3, dtype=np.int32)
        self.last_shift_end = np.full(N_EMPLOYEES, -1000.0, dtype=np.float32)
        self.same_shift_streak = np.zeros(N_EMPLOYEES, dtype=np.int32)
        self.consecutive_work_days = np.zeros(N_EMPLOYEES, dtype=np.int32)
        self.consecutive_rest_days = np.zeros(N_EMPLOYEES, dtype=np.int32)
        self.shift_counts = np.zeros((N_EMPLOYEES, len(ALL_SHIFTS)), dtype=np.int32)
        self.special_day_work_counts = np.zeros(N_EMPLOYEES, dtype=np.int32)
        self.weekend_work_counts = np.zeros(N_EMPLOYEES, dtype=np.int32)
        self.last_weekend_shift_coverage = {}
        self.last_emp_reward = np.zeros(N_EMPLOYEES, dtype=np.float32)
        self.current_day = 0
        self.total_jolly = 0
        self.total_mp = 0
        self.total_hard_overrides = 0
        self.total_soft_violations = 0
        self.total_coverage_violations = 0
        self.jolly_used_week = defaultdict(int)
        # 50% chance: use all 3 AP weeks {1,2,4}; 50% chance: pick 2 out of {1,2,4}
        if self.np_random.random() < 0.5:
            self.ap_active_weeks = frozenset({1, 2, 4})
        else:
            dropped = int(self.np_random.choice([1, 2, 4]))
            self.ap_active_weeks = frozenset({1, 2, 4} - {dropped})
        self.ap_assigned_sundays = {}   # ← aggiungere
        self.ap_count_monthly = {}      # ← aggiungere    
        self.ap_week_plan = self._draw_ap_week(0)
        self.ap_day_infermieri = self._draw_ap_day_infermieri(0)
        self.rest_week_plan = self._draw_rest_week(0, ap_week_plan=self.ap_week_plan, ap_day_infermieri=self.ap_day_infermieri)
        self.ap_today = self._get_ap_today(0)
        self.rest_today = self._get_rest_today(0)
        self.final_shifts_today = {}
        self.monthly_kpi_by_role = None
        self.monthly_kpi_by_employee = None
        self.jolly_days = set()
        self.last_special_assignment_by_role_shift_type = {}
        self.weekly_coverage_checks = defaultdict(int)
        return self._get_obs(), {"date_label": self._date_label(0)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions):
        day = self.current_day
        date_label = self._date_label(day)
        weekday_idx = self._weekday_idx(day)
        special_day_types = self._special_day_types(day)
        is_special = bool(special_day_types)

        prev_last_shift = self.last_shift.copy()
        final_shifts: Dict[int, int] = {}
        info = {
            "day": day,
            "date": self._date_for_day(day).isoformat(),
            "date_label": date_label,
            "weekday_idx": weekday_idx,
            "weekday_label": self._weekday_label(day),
            "special_day_types": list(special_day_types),
            "is_weekend": bool(self._is_weekend(day)),
            "is_holiday": bool(self._is_holiday(day)),
            "is_prefestive": bool(self._is_prefestive(day)),
            "is_special_day": is_special,
            "violations": [],
            "soft_violations": [],
            "hard_overrides": [],
            "mp_activations": [],
            "jolly_activations": [],
            "alternation_derogations": [],
            "reward_components": {},
            "reward_components_weighted": {},
            "reward_weights": dict(self.reward_weights),
        }
        reward_components = {k: 0.0 for k in self.reward_weights}

        # 1) Individual hard handling and legality-aware assignment
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            proposed = int(actions[i])
            final = proposed

            

            if self.ap_today[i]:
                if proposed != 3:
                    reward_components['legality'] += PENALTY_DAILY['ap_violated']
                    info['hard_overrides'].append(f"{emp['name']}: AP override")
                    self.total_hard_overrides += 1
                else:
                    reward_components['legality'] += REWARD_DAILY['ap_rispettato']
                final = 6

            elif self.rest_today[i]:
                if proposed != 3:
                    reward_components['legality'] += PENALTY_DAILY['rest_day_violated']
                    info['hard_overrides'].append(f"{emp['name']}: riposo settimanale override")
                    self.total_hard_overrides += 1
                else:
                    reward_components['legality'] += REWARD_DAILY['riposo_settimanale_rispettato']
                final = 3

            elif (emp_id, day) in self.ferie_table:
                if proposed != 3:
                    reward_components['legality'] += PENALTY_DAILY['ferie_violated']
                    info['hard_overrides'].append(f"{emp['name']}: ferie override")
                    self.total_hard_overrides += 1
                final = 3
            elif not self._is_action_legal(i, proposed, day, prev_last_shift):
                reward_components['legality'] += PENALTY_DAILY['hard_override']
                if self._violates_special_day_alternation(i, proposed, day, prev_last_shift):
                    info['hard_overrides'].append(
                        f"{emp['name']}: alternanza obbligatoria giorno speciale {AGENT_SHIFTS[proposed]} -> R"
                    )
                else:
                    info['hard_overrides'].append(f"{emp['name']}: azione illegale {AGENT_SHIFTS[proposed]} -> R")
                self.total_hard_overrides += 1
                final = 3
            else:
                reward_components['preference'] += REWARD_DAILY['valid_assignment']

            if (
                final == 3
                and not self.rest_today[i]
                and not self.ap_today[i]
                and (emp_id, day) not in self.ferie_table
                and prev_last_shift[emp_id] != 2
            ):
                reward_components['preference'] += PENALTY_DAILY['unjustified_rest']
                info['soft_violations'].append(f"{emp['name']}: riposo non necessario")
                self.total_soft_violations += 1

            if (
                self._is_weekend(day)
                and final not in (3, 6)
                and self._worked_last_weekend(i, day)
            ):
                reward_components['fairness'] += PENALTY_DAILY['consecutive_weekend_work']
                info['soft_violations'].append(f"{emp['name']}: weekend consecutivo")
                self.total_soft_violations += 1

            reward_components['fairness'] += self._compute_soft_rotation_penalty(i, final, prev_last_shift, day)
            if day > 0 and final not in (3, 6):
                prev_shift = int(prev_last_shift[emp_id])
                if prev_shift not in (3, 6):
                    rest_hours = self._rest_hours_between(day - 1, prev_shift, day, final)
                    if rest_hours >= 11:
                        reward_components['legality'] += REWARD_DAILY['riposo_rispettato']
                        info['soft_violations'].append(f"{emp['name']}: riposo >= 11 ore")

            # 1.4) Weekend rest enforcement: if worked Saturday, force Sunday rest for medici
            # This reduces consecutive weekend work and spreads coverage via Jolly
            if emp['type'] == EMPLOYEE_TYPE_MEDICO and self._weekday_idx(day) == 6:  # Sunday
                saturday_shift = self._get_saturday_shift_medico(i, day)
                if saturday_shift is not None and saturday_shift not in (3, 6):  # Worked on Saturday (not Rest/AP)
                    if final not in (3, 6):  # Trying to work on Sunday
                        # Apply soft penalty for consecutive weekend work
                        reward_components['fairness'] += PENALTY_DAILY['weekend_consecutive_work']
                        info['soft_violations'].append(f"{emp['name']}: lavoro sia sabato che domenica")
                        self.total_soft_violations += 1
                        # Force Sunday rest (hard override for this optimization focus)
                        final = 3
            
            final_shifts[emp_id] = final

        # 1.5) MP weekend reward: incentivize MP shifts on weekends (reduces staffing burden)
        if self._is_weekend(day):
            for emp_id, shift in final_shifts.items():
                if shift == 4:  # MP shift
                    reward_components['efficiency'] += REWARD_DAILY['mp_weekend_bonus']

        # 2) Coverage and minimal repair
        coverage_detail = self._check_coverage_detail(final_shifts)
        mp_needed = any(
            coverage_detail[slot][role]['missing'] > 0
            for slot in ('M', 'P') for role in ROLE_ORDER
        )

        if not self._is_weekend(day):
            mp_needed = False

        if mp_needed:
            mp_candidate = self._find_mp_candidate(final_shifts, day, prev_last_shift)
            if mp_candidate is not None:
                final_shifts[mp_candidate] = 4
                reward_components['efficiency'] += PENALTY_DAILY['mp_activated']
                info['mp_activations'].append(EMPLOYEES[mp_candidate]['name'])
                self.total_mp += 1
                coverage_detail = self._check_coverage_detail(final_shifts)

        # Jolly Medico: external doctor for night when no internal medico covers N
        jolly_night_active = False
        if coverage_detail['N']['Medico']['missing'] > 0:
            avoidable = any(
                self._is_action_legal_base(i, 2, day, prev_last_shift)
                for i, emp in enumerate(EMPLOYEES) if emp['type'] == EMPLOYEE_TYPE_MEDICO
            )
            # Incentivize Jolly on weekends (reward), penalize on weekdays
            is_weekend = self._is_weekend(day)
            if not avoidable and is_weekend:
                # Inevitable weekend use: REWARD to encourage using Jolly (less employees on weekends)
                jolly_penalty = REWARD_DAILY['jolly_weekend_bonus']  # +1.20 bonus
                status = 'inevitable-weekend'
            else:
                penalty_key = 'jolly_avoidable' if avoidable else 'jolly_inevitable'
                jolly_penalty = PENALTY_DAILY[penalty_key]
                status = 'avoidable' if avoidable else 'inevitable-weekday'
            
            reward_components['efficiency'] += jolly_penalty
            info['jolly_activations'].append(f"Medico-N-{status}")
            self.total_jolly += 1
            self.jolly_used_week[day // WEEK_LEN] += 1
            self.jolly_days.add(day)
            jolly_night_active = True
            coverage_detail['N']['Medico']['missing'] = 0
            coverage_detail['N']['Medico']['assigned'] += 1

        # 3) Coverage reward after repair
        all_covered = True
        for slot in COVERAGE_SLOTS:
            for role in ROLE_ORDER:
                miss = coverage_detail[slot][role]['missing']
                assn = min(coverage_detail[slot][role]['assigned'], coverage_detail[slot][role]['required'])
                reward_components['coverage'] += assn * REWARD_DAILY['slot_covered']
                if miss > 0:
                    all_covered = False
                    penalty_key = 'slot_uncovered_medico' if role == 'Medico' else 'slot_uncovered_infermiere'
                    reward_components['coverage'] += miss * PENALTY_DAILY[penalty_key]
                    self.total_coverage_violations += miss
                    info['violations'].append(f"coverage:{slot}-{role}-missing:{miss}")
        if all_covered:
            reward_components['coverage'] += REWARD_DAILY['all_coverage_met']
            reward_components['coverage'] += REWARD_DAILY['complete_daily_coverage']

        # Dirigenza daily coverage
        medico_on_m = medico_on_p = medico_on_n = 0
        for emp in EMPLOYEES:
            if emp['type'] == EMPLOYEE_TYPE_MEDICO:
                shift = final_shifts.get(emp['id'], 3)
                if shift == 0:   medico_on_m += 1
                elif shift == 1: medico_on_p += 1
                elif shift == 2: medico_on_n += 1
                elif shift == 4: medico_on_m += 1; medico_on_p += 1
        if jolly_night_active:
            medico_on_n += 1

        if medico_on_m > 0 and medico_on_p > 0 and medico_on_n > 0:
            reward_components['coverage'] += REWARD_DAILY['dirigenza_daily_coverage']
            info['dirigenza_covered'] = True
        else:
            info['dirigenza_covered'] = False
            if medico_on_m == 0:
                info['dirigenza_missing'] = info.get('dirigenza_missing', []) + ['M']
                reward_components['coverage'] += PENALTY_DAILY['dirigenza_m_uncovered']
            if medico_on_p == 0:
                info['dirigenza_missing'] = info.get('dirigenza_missing', []) + ['P']
                reward_components['coverage'] += PENALTY_DAILY['dirigenza_p_uncovered']
            if medico_on_n == 0:
                info['dirigenza_missing'] = info.get('dirigenza_missing', []) + ['N']
                reward_components['coverage'] += PENALTY_DAILY['dirigenza_n_uncovered']
        if medico_on_n > 0 and not jolly_night_active:
            reward_components['coverage'] += REWARD_DAILY['dirigenza_night_covered']

        # Infermieri daily coverage
        inf_on_m = inf_on_p = inf_on_n = 0
        for emp in EMPLOYEES:
            if emp['type'] == EMPLOYEE_TYPE_INFERMIERE:
                shift = final_shifts.get(emp['id'], 3)
                if shift == 0:   inf_on_m += 1
                elif shift == 1: inf_on_p += 1
                elif shift == 2: inf_on_n += 1
                elif shift == 4: inf_on_m += 1; inf_on_p += 1

        if inf_on_m > 0 and inf_on_p > 0 and inf_on_n > 0:
            reward_components['coverage'] += REWARD_DAILY['infermieri_daily_coverage']
            info['infermieri_covered'] = True
        else:
            info['infermieri_covered'] = False
            if inf_on_m == 0:
                info['infermieri_missing'] = info.get('infermieri_missing', []) + ['M']
                reward_components['coverage'] += PENALTY_DAILY['infermieri_m_uncovered']
            if inf_on_p == 0:
                info['infermieri_missing'] = info.get('infermieri_missing', []) + ['P']
                reward_components['coverage'] += PENALTY_DAILY['infermieri_p_uncovered']
            if inf_on_n == 0:
                info['infermieri_missing'] = info.get('infermieri_missing', []) + ['N']
                reward_components['coverage'] += PENALTY_DAILY['infermieri_n_uncovered']
        if inf_on_n > 0:
            reward_components['coverage'] += REWARD_DAILY['infermieri_night_covered']

        # --- Adaptive weekly coverage check tracking ---
        # Score per giorno: +1 se dirigenza coperta (M+P+N), +1 se comparto coperto (M+P+N) → max 2
        daily_check_score = (
            (1 if info.get('dirigenza_covered', False) else 0)
            + (1 if info.get('infermieri_covered', False) else 0)
        )
        self.weekly_coverage_checks[day // WEEK_LEN] += daily_check_score
        info['daily_coverage_check_score'] = daily_check_score
        # Reward giornaliero: solo giorni feriali (lun-ven) e solo se entrambi coperti (score == 2)
        if not self._is_weekend(day) and daily_check_score == 2:
            reward_components['coverage'] += REWARD_DAILY['weekday_complete_coverage']

        # Medico-Infermiere Ratio check (1 medico : 2 infermieri)
        # Verifica che il rapporto tra infermieri e medici sia coerente per ogni turno
        ratio_shifts_ok = []
        
        # Morning shift (M)
        if medico_on_m > 0 and inf_on_m >= medico_on_m * 2:
            ratio_shifts_ok.append('M')
            reward_components['coverage'] += REWARD_DAILY['medico_infermiere_ratio']
            info['medico_inf_ratio_ok'] = info.get('medico_inf_ratio_ok', []) + ['M']
        
        # Afternoon shift (P)
        if medico_on_p > 0 and inf_on_p >= medico_on_p * 2:
            ratio_shifts_ok.append('P')
            reward_components['coverage'] += REWARD_DAILY['medico_infermiere_ratio']
            info['medico_inf_ratio_ok'] = info.get('medico_inf_ratio_ok', []) + ['P']
        
        # Night shift (N)
        if medico_on_n > 0 and inf_on_n >= medico_on_n * 2:
            ratio_shifts_ok.append('N')
            reward_components['coverage'] += REWARD_DAILY['medico_infermiere_ratio']
            info['medico_inf_ratio_ok'] = info.get('medico_inf_ratio_ok', []) + ['N']

        #Massimo 1 medico e 2 infermieri su turno N
        medico_night_count = 0
        infermiere_night_count = 0
        for emp in EMPLOYEES:
            shift = final_shifts.get(emp['id'], 3)
            if shift == 2:  # N shift
                if emp['type'] == EMPLOYEE_TYPE_MEDICO:
                    medico_night_count += 1
                elif emp['type'] == EMPLOYEE_TYPE_INFERMIERE:
                    infermiere_night_count += 1

        if medico_night_count == 1 and infermiere_night_count <= 2:
            reward_components['coverage'] += REWARD_DAILY['copertura_n_minima']
            info['copertura_n_minima_ok'] = True
        else:
            reward_components['coverage'] += PENALTY_DAILY['copertura_n_minima_violata']
            info['copertura_n_minima_ok'] = False

        # Shift distribution on weekdays (feriali): adaptive percentage-based check
        # Targets: M 55-65%, P 15-25%, N 15-25%
        if self._is_weekday(day):
            is_dist_ok, distribution_detail, dist_reward = self._check_shift_distribution_adaptive(final_shifts)
            reward_components['coverage'] += dist_reward
            info['shift_distribution_ok'] = is_dist_ok
            info['shift_distribution_detail'] = distribution_detail
        #-------------------------------------------------

        # Soft preferences
        info['preference_violations'] = []
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            pref_key = (emp_id, day)
            if pref_key in self.preferences_table:
                pref_type = self.preferences_table[pref_key]
                blocked_shifts = PREFERENCE_TYPES.get(pref_type, set())
                assigned_shift = final_shifts.get(emp_id, 3)
                if assigned_shift in blocked_shifts:
                    reward_components['preference'] += PENALTY_DAILY['preference_violated']
                    info['preference_violations'].append(
                        f"{emp['name']}: {pref_type} violata (assegnato {ALL_SHIFTS[assigned_shift]})"
                    )
                    info['soft_violations'].append(f"{emp['name']}: preferenza {pref_type} violata")
                    self.total_soft_violations += 1
                else:
                    reward_components['preference'] += REWARD_DAILY['preference_respected']

        # 4) Calendar/special-day fairness + alternation rule
        reward_components['calendar'] += self._compute_calendar_fairness(final_shifts, day, info)

        # 4b) Combined coverage checks (medici + infermieri minimal coverage, residual, proportionality)
        medici_dist = self._get_combined_slot_distribution(final_shifts, EMPLOYEE_TYPE_MEDICO)
        infermieri_dist = self._get_combined_slot_distribution(final_shifts, EMPLOYEE_TYPE_INFERMIERE)
        
        # Check 1: Minimal coverage (at least 1 medico and 1 infermiere per shift)
        if self._check_coverage_first(medici_dist, infermieri_dist):
            reward_components['coverage'] += REWARD_DAILY['coverage_first_check_pass']
            info['coverage_first_check_pass'] = True
        else:
            reward_components['coverage'] += PENALTY_DAILY['coverage_first_check_fail']
            info['coverage_first_check_pass'] = False
            info['violations'].append('coverage_first_check_fail')
        
        # Check 2: Residual distribution (percentage limits after subtracting 1)
        if self._check_coverage_residual(medici_dist, infermieri_dist):
            reward_components['coverage'] += REWARD_DAILY['coverage_residual_pass']
            info['coverage_residual_pass'] = True
        else:
            reward_components['coverage'] += PENALTY_DAILY['coverage_residual_fail']
            info['coverage_residual_pass'] = False
            info['violations'].append('coverage_residual_fail')
        
        # Check 3: Shift proportionality (infermieri/medici ratio per shift)
        shift_prop_results = self._check_shift_proportionality(medici_dist, infermieri_dist)
        if all(shift_prop_results.values()):
            reward_components['coverage'] += REWARD_DAILY['shift_proportion_all_pass']
            info['shift_proportion_all_pass'] = True
        else:
            reward_components['coverage'] += PENALTY_DAILY['shift_proportion_fail']
            info['shift_proportion_all_pass'] = False
            failed_shifts = [s for s, ok in shift_prop_results.items() if not ok]
            info['violations'].append(f'shift_proportion_fail: {failed_shifts}')

        # 5) Commit shifts
        self.final_shifts_today = dict(final_shifts)
        self._commit_day(final_shifts, prev_last_shift, day)

        # 5.5) Consecutive weekends check for dirigenza (medici)
        # Check on Saturday: if worked last weekend, reward for rest this Saturday, penalty for working again
        if self._weekday_idx(day) == 5:  # Saturday
            for i, emp in enumerate(EMPLOYEES):
                if emp['type'] == EMPLOYEE_TYPE_MEDICO:
                    current_shift = final_shifts.get(emp['id'], 3)
                    worked_prev_weekend, working_this_saturday = self._check_consecutive_weekends(i, day, current_shift)
                    
                    if worked_prev_weekend:
                        if working_this_saturday:
                            # Worked last weekend AND working this Saturday → penalty
                            reward_components['fairness'] += PENALTY_DAILY['consecutive_weekend_work_multi']
                            info['hard_overrides'].append(f"Consecutive weekend work penalty: {emp['name']}")
                        else:
                            # Worked last weekend BUT resting this Saturday → reward (recovery)
                            reward_components['fairness'] += REWARD_DAILY['consecutive_weekend_rest_recovery']
                            info['hard_overrides'].append(f"Consecutive weekend rest recovery reward: {emp['name']}")

        # 6) Weekly/monthly bonuses/penalties
        if (day + 1) % WEEK_LEN == 0:
            weekly_reward, weekly_info = self._compute_weekly_reward(day // WEEK_LEN)
            reward_components['weekly'] += weekly_reward
            info['weekly'] = weekly_info
            self.hours_week[:] = 0.0
            
            # Draw AP plan only at the end of weeks 1, 2, and 4 (3 times per month)
            current_date = self._date_for_day(day)
            month_start = current_date.replace(day=1)
            days_from_month_start = (current_date - month_start).days
            week_of_month = days_from_month_start // 7 + 1  # Week 1, 2, 3, or 4
            
            if week_of_month in self.ap_active_weeks:
                nxt = self.current_day + 1
                self.ap_week_plan = self._draw_ap_week(nxt)
                self.ap_day_infermieri = self._draw_ap_day_infermieri(nxt)
                self.rest_week_plan = self._draw_rest_week(nxt, ap_week_plan=self.ap_week_plan, ap_day_infermieri=self.ap_day_infermieri)

        self.current_day += 1
        self.ap_today = self._get_ap_today(self.current_day)
        self.rest_today = self._get_rest_today(self.current_day)
        terminated = self.current_day >= self._days_in_episode

        if terminated:
            monthly_reward, monthly_info = self._compute_monthly_reward()
            reward_components['monthly'] += monthly_reward
            info['monthly'] = monthly_info

        reward = 0.0
        weighted = {}
        for k, v in reward_components.items():
            weighted[k] = v * self.reward_weights.get(k, 1.0)
            reward += weighted[k]

        info['reward_components'] = reward_components
        info['reward_components_weighted'] = weighted
        info['coverage_detail'] = coverage_detail
        info['hard_overrides_today'] = len(info['hard_overrides'])
        info['soft_violations_today'] = len(info['soft_violations'])
        info['coverage_violations_today'] = sum(
            cov[role]['missing'] for cov in coverage_detail.values() for role in ROLE_ORDER
        )
        info['jolly_used_today'] = len(info['jolly_activations'])
        info['mp_used_today'] = len(info['mp_activations'])
        info['final_shifts'] = {emp['name']: ALL_SHIFTS[final_shifts[emp['id']]] for emp in EMPLOYEES}
        if jolly_night_active:
            info['final_shifts']['Jolly Medico'] = 'N'
        info['jolly_night_active'] = jolly_night_active
        info['total_hard_overrides'] = self.total_hard_overrides
        info['total_soft_violations'] = self.total_soft_violations
        info['total_mp'] = self.total_mp
        info['total_jolly'] = self.total_jolly
        self.last_emp_reward = np.full(N_EMPLOYEES, reward / max(1, N_EMPLOYEES), dtype=np.float32)
        obs = self._get_obs() if not terminated else np.zeros_like(self._get_obs())
        return obs, float(reward), bool(terminated), False, info

    # ------------------------------------------------------------------
    # Legality helpers
    # ------------------------------------------------------------------
    def _get_shift_window_for_day(self, day: int, shift_id: int):
        window = SHIFT_TIME_WINDOWS.get(shift_id)
        if window is None:
            return None
        start, end = window
        base = day * 24
        return base + start, base + end

    def _rest_hours_between(self, prev_day: int, prev_shift: int, next_day: int, next_shift: int) -> float:
        prev_window = self._get_shift_window_for_day(prev_day, prev_shift)
        next_window = self._get_shift_window_for_day(next_day, next_shift)
        if prev_window is None or next_window is None:
            return 24.0
        return float(next_window[0] - prev_window[1])

    def _is_action_legal_base(self, emp_idx: int, action: int, day: int, prev_last_shift=None) -> bool:
        if action not in SHIFT_IDS:
            return False
        emp = EMPLOYEES[emp_idx]
        emp_id = emp['id']
        contract = CONTRACT_TYPES[emp['contract']]
        prev_shifts = self.last_shift if prev_last_shift is None else prev_last_shift

        # Always allow rest
        if action == 3:
            return True
        
        # Only mask: daily rest hours between shifts
        prev_shift = int(prev_shifts[emp_id])
        rest_hours = self._rest_hours_between(max(day - 1, 0), prev_shift, day, action)
        if day > 0 and prev_shift not in (3, 6) and rest_hours < contract['min_daily_rest_hours']:
            return False
        
        return True

    def _get_last_special_assignee(self, role: str, shift_name: str, special_type: str):
        return self.last_special_assignment_by_role_shift_type.get((role, shift_name, special_type))

    def _exists_legal_alternative_for_special_slot(self, current_emp_idx: int, role: str, action: int, day: int, prev_last_shift=None) -> bool:
        for alt_idx, alt_emp in enumerate(EMPLOYEES):
            if alt_idx == current_emp_idx:
                continue
            if ROLE_NAMES[alt_emp['type']] != role:
                continue
            if self._is_action_legal_base(alt_idx, action, day, prev_last_shift):
                return True
        return False

    def _violates_special_day_alternation(self, emp_idx: int, action: int, day: int, prev_last_shift=None) -> bool:
        if not SPECIAL_DAY_ALTERNATION['enabled']:
            return False
        if action not in (0, 1, 2):
            return False
        emp = EMPLOYEES[emp_idx]
        role = ROLE_NAMES[emp['type']]
        shift_name = AGENT_SHIFTS[action]
        if role not in SPECIAL_ROLES or shift_name not in SPECIAL_SHIFT_NAMES:
            return False
        special_types = self._special_day_types(day)
        if not special_types:
            return False
        for special_type in special_types:
            last_emp_id = self._get_last_special_assignee(role, shift_name, special_type)
            if last_emp_id == emp['id'] and self._exists_legal_alternative_for_special_slot(emp_idx, role, action, day, prev_last_shift):
                return True
        return False

    def _is_action_legal(self, emp_idx: int, action: int, day: int, prev_last_shift=None) -> bool:
        return self._is_action_legal_base(emp_idx, action, day, prev_last_shift)

    # ------------------------------------------------------------------
    # AP / rest weekly plans
    # ------------------------------------------------------------------
    def _draw_ap_week(self, week_start: int): #ap smart
        plan = np.full(N_EMPLOYEES, -1, dtype=np.int32)
        week_days = list(range(week_start, min(week_start + WEEK_LEN, self._days_in_episode)))
        sunday_interval = self._get_sunday_interval(week_start)  # Get the sunday-sunday interval
        
        # Calculate max AP per day for medici (30% limit)
        total_medici = sum(1 for e in EMPLOYEES if e['type'] == EMPLOYEE_TYPE_MEDICO)
        max_medici_ap_same_day = max(1, int(np.floor(total_medici * 0.3)))
        
        # Track AP assignments per day
        ap_count_by_day = {day: 0 for day in week_days}
        
        # Shuffle medici for fair distribution
        medici_indices = [i for i, e in enumerate(EMPLOYEES) if e['type'] == EMPLOYEE_TYPE_MEDICO]
        self.np_random.shuffle(medici_indices)
        
        for i in medici_indices:
            emp = EMPLOYEES[i]
            
            # Check if this medico already has an AP in this sunday-sunday interval
            if emp['id'] in self.ap_assigned_sundays and self.ap_assigned_sundays[emp['id']] == sunday_interval[0]:
                continue  # Skip: already has AP in this interval
            
            # Check monthly AP limit (max 3 for dirigenza per month)
            monthly_ap_count = self.ap_count_monthly.get(emp['id'], 0)
            if monthly_ap_count >= 3:
                continue  # Skip: already reached max 3 AP for this month
            
            excluded = {d for d in week_days if (emp['id'], d) in self.ferie_table}
            if self.rest_week_plan is not None and self.rest_week_plan[i] >= 0:
                excluded.add(int(self.rest_week_plan[i]))
            # Exclude Sunday (weekday 6)
            excluded.update(d for d in week_days if self._weekday_idx(d) == 6)
            
            # Find candidates: days not excluded AND respecting 30% limit
            candidates = []
            for day in week_days:
                if day not in excluded and ap_count_by_day[day] < max_medici_ap_same_day:
                    candidates.append(day)
            
            # Se nessun giorno rispetta il 30% limit, salta: meglio nessun AP che violare il limite
            if not candidates:
                continue
            
            # Assign AP day
            if candidates:
                ap_day = int(self.np_random.choice(candidates))
                plan[i] = ap_day
                ap_count_by_day[ap_day] += 1
                # Track this AP assignment in the sunday-sunday interval
                self.ap_assigned_sundays[emp['id']] = sunday_interval[0]
                # Increment monthly counter
                self.ap_count_monthly[emp['id']] = monthly_ap_count + 1
        return plan

    def _draw_ap_day_infermieri(self, week_start: int):
        """Draw a single AP day per infermiere with 1/3 probability.
        Day is randomly selected from the week, excluding Sunday, ferie, and scheduled rest days.
        Respects the sunday-sunday interval constraint: max 1 AP per interval.
        Distributes AP across days to avoid concentration (max 30% infermieri per day).
        """
        plan = np.full(N_EMPLOYEES, -1, dtype=np.int32)
        week_days = list(range(week_start, min(week_start + WEEK_LEN, self._days_in_episode)))
        sunday_interval = self._get_sunday_interval(week_start)  # Get the sunday-sunday interval
        
        # Calculate max AP per day for infermieri (30% limit)
        total_infermieri = sum(1 for e in EMPLOYEES if e['type'] == EMPLOYEE_TYPE_INFERMIERE)
        max_infermieri_ap_same_day = max(1, int(np.floor(total_infermieri * 0.3)))
        
        # Track AP assignments per day
        ap_count_by_day = {day: 0 for day in week_days}
        
        # Shuffle infermieri for fair distribution
        infermieri_indices = [i for i, e in enumerate(EMPLOYEES) if e['type'] == EMPLOYEE_TYPE_INFERMIERE]
        self.np_random.shuffle(infermieri_indices)
        
        for i in infermieri_indices:
            emp = EMPLOYEES[i]
            
            # Check if this infermiere already has an AP in this sunday-sunday interval
            if emp['id'] in self.ap_assigned_sundays and self.ap_assigned_sundays[emp['id']] == sunday_interval[0]:
                continue  # Skip: already has AP in this interval
            
            # Check monthly AP limit (max 1 for infermieri per month)
            monthly_ap_count = self.ap_count_monthly.get(emp['id'], 0)
            if monthly_ap_count >= 1:
                continue  # Skip: already has 1 AP for this month
            
            # 1/3 probability to get an AP day this month
            if self.np_random.random() >= 1.0 / 3.0:
                continue  # No AP day for this infermiere
            
            # Exclude ferie
            excluded = {d for d in week_days if (emp['id'], d) in self.ferie_table}
            # Exclude scheduled rest days
            if self.rest_week_plan is not None and self.rest_week_plan[i] >= 0:
                excluded.add(int(self.rest_week_plan[i]))
            # Exclude Sunday (weekday 6)
            excluded.update(d for d in week_days if self._weekday_idx(d) == 6)
            
            # Find candidates: days not excluded AND respecting 30% limit
            candidates = []
            for day in week_days:
                if day not in excluded and ap_count_by_day[day] < max_infermieri_ap_same_day:
                    candidates.append(day)
            
            # Se nessun giorno rispetta il 30% limit, salta: meglio nessun AP che violare il limite
            if not candidates:
                continue
            
            # Assign AP day
            if candidates:
                ap_day = int(self.np_random.choice(candidates))
                plan[i] = ap_day
                ap_count_by_day[ap_day] += 1
                # Track this AP assignment in the sunday-sunday interval
                self.ap_assigned_sundays[emp['id']] = sunday_interval[0]
                # Increment monthly counter
                self.ap_count_monthly[emp['id']] = monthly_ap_count + 1
        
        return plan

    def _get_ap_today(self, day: int):
        if day >= self._days_in_episode:
            return np.zeros(N_EMPLOYEES, dtype=bool)
        # Check both dirigenza AP (weekly plan) and infermieri AP (single day per month)
        ap_today = (self.ap_week_plan == day).copy() if self.ap_week_plan is not None else np.zeros(N_EMPLOYEES, dtype=bool)
        if self.ap_day_infermieri is not None:
            ap_today |= (self.ap_day_infermieri == day)
        return ap_today

    def _draw_rest_week(self, week_start: int, ap_week_plan=None, ap_day_infermieri=None):
        """
        Draw weekly rest days with intelligent distribution.
        Considers existing AP assignments to avoid over-depleting staff on same day.
        Combined limit (rest + AP) ≤ 40% per role per day.
        """
        plan = np.full(N_EMPLOYEES, -1, dtype=np.int32)
        if not WEEKLY_REST_ENABLED:
            return plan
        
        week_days = list(range(week_start, min(week_start + WEEK_LEN, self._days_in_episode)))
        
        # Calculate max employees per role (combined rest + AP ≤ 40%)
        total_medici = sum(1 for e in EMPLOYEES if e['type'] == EMPLOYEE_TYPE_MEDICO)
        total_infermieri = sum(1 for e in EMPLOYEES if e['type'] == EMPLOYEE_TYPE_INFERMIERE)
        max_medici_combined = max(1, int(np.floor(total_medici * 0.4)))
        max_infermieri_combined = max(1, int(np.floor(total_infermieri * 0.4)))
        
        # Track combined (rest + AP) assignments per day per role
        combined_count_by_day = {day: {'medici': 0, 'infermieri': 0} for day in week_days}
        
        # First pass: count existing AP assignments
        if ap_week_plan is not None:
            for i, ap_day in enumerate(ap_week_plan):
                if ap_day >= 0 and ap_day in combined_count_by_day:
                    emp = EMPLOYEES[i]
                    role_key = 'medici' if emp['type'] == EMPLOYEE_TYPE_MEDICO else 'infermieri'
                    combined_count_by_day[ap_day][role_key] += 1
        
        if ap_day_infermieri is not None:
            for i, ap_day in enumerate(ap_day_infermieri):
                if ap_day >= 0 and ap_day in combined_count_by_day:
                    combined_count_by_day[ap_day]['infermieri'] += 1
        
        # Shuffle employees for fair distribution
        employee_indices = list(range(N_EMPLOYEES))
        self.np_random.shuffle(employee_indices)
        
        for idx in employee_indices:
            emp = EMPLOYEES[idx]
            excluded = {d for d in week_days if (emp['id'], d) in self.ferie_table}
            
            # Exclude AP day for medici
            if emp['type'] == EMPLOYEE_TYPE_MEDICO and ap_week_plan is not None and ap_week_plan[idx] >= 0:
                excluded.add(int(ap_week_plan[idx]))
            
            # Exclude AP day for infermieri
            if emp['type'] == EMPLOYEE_TYPE_INFERMIERE and ap_day_infermieri is not None and ap_day_infermieri[idx] >= 0:
                excluded.add(int(ap_day_infermieri[idx]))
            
            # Determine role-specific limits
            role_key = 'medici' if emp['type'] == EMPLOYEE_TYPE_MEDICO else 'infermieri'
            max_allowed = max_medici_combined if emp['type'] == EMPLOYEE_TYPE_MEDICO else max_infermieri_combined
            
            # Find candidates: days not excluded AND respecting combined limit
            candidates = []
            for day in week_days:
                if day not in excluded and combined_count_by_day[day][role_key] < max_allowed:
                    candidates.append(day)
            
            # Fallback: pick day with minimum combined count
            if not candidates:
                candidates = sorted(
                    [d for d in week_days if d not in excluded],
                    key=lambda d: combined_count_by_day[d][role_key]
                )
            
            # Assign rest day
            if candidates:
                rest_day = int(self.np_random.choice(candidates))
                plan[idx] = rest_day
                combined_count_by_day[rest_day][role_key] += 1
        
        return plan

    def _get_rest_today(self, day: int):
        if day >= self._days_in_episode or self.rest_week_plan is None:
            return np.zeros(N_EMPLOYEES, dtype=bool)
        return self.rest_week_plan == day

    # ------------------------------------------------------------------
    # Adaptive shift distribution (percentage-based)
    # ------------------------------------------------------------------
    def _check_shift_distribution_adaptive(self, final_shifts: Dict[int, int]) -> tuple:
        """
        Check shift distribution using percentage-based targets for M, P, N.
        Targets: M 55-65%, P 15-25%, N 15-25%
        Returns (is_ok, distribution_dict, penalty_reward)
        """
        # Count employees by role
        total_medici = sum(1 for emp in EMPLOYEES if emp['type'] == EMPLOYEE_TYPE_MEDICO)
        total_infermieri = sum(1 for emp in EMPLOYEES if emp['type'] == EMPLOYEE_TYPE_INFERMIERE)
        
        # Count by shift
        m_medici = m_infermieri = 0
        p_medici = p_infermieri = 0
        n_medici = n_infermieri = 0
        
        for emp in EMPLOYEES:
            shift = final_shifts.get(emp['id'], 3)
            is_medico = emp['type'] == EMPLOYEE_TYPE_MEDICO
            
            if shift == 0:  # M shift
                if is_medico:
                    m_medici += 1
                else:
                    m_infermieri += 1
            elif shift == 1:  # P shift
                if is_medico:
                    p_medici += 1
                else:
                    p_infermieri += 1
            elif shift == 2:  # N shift
                if is_medico:
                    n_medici += 1
                else:
                    n_infermieri += 1
            elif shift == 4:  # MP counts toward both M and P
                if is_medico:
                    m_medici += 1
                    p_medici += 1
                else:
                    m_infermieri += 1
                    p_infermieri += 1
        
        # Define percentage targets: M 55-65%, P 15-25%, N 15-25%
        targets = {
            'M': (0.55, 0.65),
            'P': (0.15, 0.25),
            'N': (0.15, 0.25),
        }
        
        # Calculate expected ranges (using ceiling/floor for integer values)
        ranges_medici = {}
        ranges_infermieri = {}
        
        for shift_name, (min_pct, max_pct) in targets.items():
            min_med = max(1, int(np.ceil(total_medici * min_pct)))
            max_med = int(np.floor(total_medici * max_pct))
            ranges_medici[shift_name] = (min_med, max_med)
            
            min_inf = max(1, int(np.ceil(total_infermieri * min_pct)))
            max_inf = int(np.floor(total_infermieri * max_pct))
            ranges_infermieri[shift_name] = (min_inf, max_inf)
        
        # Check if distributions are within acceptable ranges
        m_ok = (ranges_medici['M'][0] <= m_medici <= ranges_medici['M'][1] and 
                ranges_infermieri['M'][0] <= m_infermieri <= ranges_infermieri['M'][1])
        p_ok = (ranges_medici['P'][0] <= p_medici <= ranges_medici['P'][1] and 
                ranges_infermieri['P'][0] <= p_infermieri <= ranges_infermieri['P'][1])
        n_ok = (ranges_medici['N'][0] <= n_medici <= ranges_medici['N'][1] and 
                ranges_infermieri['N'][0] <= n_infermieri <= ranges_infermieri['N'][1])
        
        is_ok = m_ok and p_ok and n_ok
        
        # Compute reward/penalty based on how close to ideal midpoint (50% of range)
        penalty_reward = 0.0
        for shift_name, counts in [('M', (m_medici, m_infermieri)), 
                                    ('P', (p_medici, p_infermieri)), 
                                    ('N', (n_medici, n_infermieri))]:
            med_count, inf_count = counts
            med_range = ranges_medici[shift_name]
            inf_range = ranges_infermieri[shift_name]
            
            # Ideal point: middle of the range
            med_ideal = (med_range[0] + med_range[1]) / 2.0
            inf_ideal = (inf_range[0] + inf_range[1]) / 2.0
            
            # Bonus if close to ideal (< 0.5 away), penalty if far
            med_deviation = abs(med_count - med_ideal)
            inf_deviation = abs(inf_count - inf_ideal)
            
            if med_deviation < 0.5:
                penalty_reward += REWARD_DAILY.get('shift_distribution', 0.3)
            elif med_deviation > 1.5:
                penalty_reward -= PENALTY_DAILY.get('shift_distribution_violata', 0.5)
            
            if inf_deviation < 0.5:
                penalty_reward += REWARD_DAILY.get('shift_distribution', 0.3)
            elif inf_deviation > 1.5:
                penalty_reward -= PENALTY_DAILY.get('shift_distribution_violata', 0.5)
        
        distribution = {
            'M_medici': m_medici, 'M_medici_range': f"{ranges_medici['M'][0]}-{ranges_medici['M'][1]}", 'M_medici_ok': m_ok,
            'M_infermieri': m_infermieri, 'M_infermieri_range': f"{ranges_infermieri['M'][0]}-{ranges_infermieri['M'][1]}",
            'P_medici': p_medici, 'P_medici_range': f"{ranges_medici['P'][0]}-{ranges_medici['P'][1]}", 'P_medici_ok': p_ok,
            'P_infermieri': p_infermieri, 'P_infermieri_range': f"{ranges_infermieri['P'][0]}-{ranges_infermieri['P'][1]}",
            'N_medici': n_medici, 'N_medici_range': f"{ranges_medici['N'][0]}-{ranges_medici['N'][1]}", 'N_medici_ok': n_ok,
            'N_infermieri': n_infermieri, 'N_infermieri_range': f"{ranges_infermieri['N'][0]}-{ranges_infermieri['N'][1]}",
        }
        
        return is_ok, distribution, penalty_reward

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------
    def _check_coverage_detail(self, shifts_dict: Dict[int, int]):
        detail = {
            slot: {
                role: {'required': COVERAGE_REQUIREMENTS[slot][role], 'assigned': 0, 'missing': 0}
                for role in ROLE_ORDER
            }
            for slot in COVERAGE_SLOTS
        }
        for emp in EMPLOYEES:
            shift = shifts_dict.get(emp['id'], 3)
            role = ROLE_NAMES[emp['type']]
            for slot in SHIFT_COVERS.get(shift, []):
                detail[slot][role]['assigned'] += 1
        for slot in COVERAGE_SLOTS:
            for role in ROLE_ORDER:
                req = detail[slot][role]['required']
                ass = detail[slot][role]['assigned']
                detail[slot][role]['missing'] = max(0, req - ass)
        return detail

    def _find_mp_candidate(self, current_shifts: Dict[int, int], day: int, prev_last_shift):
        candidates = []
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            if current_shifts.get(emp_id, 3) not in (0, 1, 3):
                continue
            if self.ap_today[i] or self.rest_today[i] or ((emp_id, day) in self.ferie_table):
                continue
            contract = CONTRACT_TYPES[emp['contract']]
            prev_shift = int(prev_last_shift[emp_id])
            if day > 0 and prev_shift not in (3, 6):
                if self._rest_hours_between(day - 1, prev_shift, day, 4) < contract['min_daily_rest_hours']:
                    continue
            if self.hours_week[emp_id] + SHIFT_HOURS[4] > contract['weekly_hours'] + contract['max_weekly_overtime']:
                continue
            alternation_conflict = 0
            role = ROLE_NAMES[emp['type']]
            for slot_name in ('M', 'P'):
                if SPECIAL_DAY_ALTERNATION['enabled'] and slot_name in SPECIAL_SHIFT_NAMES and role in SPECIAL_ROLES:
                    for special_type in self._special_day_types(day):
                        last_emp_id = self._get_last_special_assignee(role, slot_name, special_type)
                        if last_emp_id == emp_id:
                            alternation_conflict += 1
            candidates.append((alternation_conflict, emp_id))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _get_medico_slot_distribution(self, shifts_dict: Dict[int, int]) -> dict:
        """
        Returns count of medici assigned to each slot (M, P, N).
        MP counts as both M and P.
        Returns: {'M': count, 'P': count, 'N': count}
        """
        m_count = p_count = n_count = 0
        for emp in EMPLOYEES:
            if emp['type'] == EMPLOYEE_TYPE_MEDICO:
                shift = shifts_dict.get(emp['id'], 3)
                if shift == 0:   m_count += 1
                elif shift == 1: p_count += 1
                elif shift == 2: n_count += 1
                elif shift == 4: m_count += 1; p_count += 1
        return {'M': m_count, 'P': p_count, 'N': n_count}

    def _get_combined_slot_distribution(self, shifts_dict: Dict[int, int], emp_type: int) -> Dict[str, int]:
        """
        Returns count of personnel (medici or infermieri) assigned to each slot (M, P, N).
        MP counts as both M and P.
        Args:
            shifts_dict: Dictionary mapping emp_id to shift
            emp_type: EMPLOYEE_TYPE_MEDICO or EMPLOYEE_TYPE_INFERMIERE
        Returns: {'M': count, 'P': count, 'N': count}
        """
        m_count = p_count = n_count = 0
        for emp in EMPLOYEES:
            if emp['type'] == emp_type:
                shift = shifts_dict.get(emp['id'], 3)
                if shift == 0:   m_count += 1
                elif shift == 1: p_count += 1
                elif shift == 2: n_count += 1
                elif shift == 4: m_count += 1; p_count += 1
        return {'M': m_count, 'P': p_count, 'N': n_count}

    def _check_coverage_first(self, medici_dist: Dict[str, int], infermieri_dist: Dict[str, int]) -> bool:
        """
        First coverage check: verify that all shifts (M, P, N) have at least 1 medico AND 1 infermiere.
        Returns: True if all shifts are covered, False otherwise
        """
        medici_ok = medici_dist['M'] > 0 and medici_dist['P'] > 0 and medici_dist['N'] > 0
        infermieri_ok = infermieri_dist['M'] > 0 and infermieri_dist['P'] > 0 and infermieri_dist['N'] > 0
        return medici_ok and infermieri_ok

    def _check_coverage_residual(self, medici_dist: Dict[str, int], infermieri_dist: Dict[str, int]) -> bool:
        """
        Second coverage check: verify that residual distribution (after subtracting 1 from each)
        respects percentage limits defined in COVERAGE_RESIDUAL_MAX_PERCENTAGES.
        Returns: True if residual respects limits, False otherwise
        """
        def check_residual_percentages(dist: Dict[str, int]) -> bool:
            residual = {
                'M': max(0, dist['M'] - 1),
                'P': max(0, dist['P'] - 1),
                'N': max(0, dist['N'] - 1)
            }
            total = residual['M'] + residual['P'] + residual['N']
            if total == 0:
                # When residual is exactly 0, it means exactly 1 per shift (all in ferie/AP)
                # This case is acceptable
                return True

            pct_m = residual['M'] / total if total > 0 else 0.0
            pct_p = residual['P'] / total if total > 0 else 0.0
            pct_n = residual['N'] / total if total > 0 else 0.0

            return (pct_m <= COVERAGE_RESIDUAL_MAX_PERCENTAGES['res_m_pct'] and
                    pct_p <= COVERAGE_RESIDUAL_MAX_PERCENTAGES['res_p_pct'] and
                    pct_n <= COVERAGE_RESIDUAL_MAX_PERCENTAGES['res_n_pct'])

        medici_ok = check_residual_percentages(medici_dist)
        infermieri_ok = check_residual_percentages(infermieri_dist)
        return medici_ok and infermieri_ok

    def _check_shift_proportionality(self, medici_dist: Dict[str, int], infermieri_dist: Dict[str, int]) -> Dict[str, bool]:
        """
        Check if the infermieri/medici ratio respects the proportionality constraints.
        For each shift, checks: n_infermieri_shift == floor(lambda_shift * n_medici_shift)
        Returns: {'M': bool, 'P': bool, 'N': bool} indicating pass/fail for each shift
        """
        shifts = ['M', 'P', 'N']
        result = {}

        for shift in shifts:
            n_medici = medici_dist[shift]
            n_infermieri = infermieri_dist[shift]

            if n_medici == 0:
                # If no medici on this shift, no constraint
                result[shift] = True
            else:
                lambda_key = f'lambda_{shift.lower()}'
                lambda_val = SHIFT_PROPORTIONALITY_LAMBDAS[lambda_key]
                expected_infermieri = int(lambda_val * n_medici)  # floor
                result[shift] = (n_infermieri == expected_infermieri)

        return result

    # ------------------------------------------------------------------
    # Rotation / fairness penalties
    # ------------------------------------------------------------------
    def _compute_weekly_rotation_balance(self, emp_id: int, day: int, window_days: int = 7):
        start_day = max(0, day - window_days + 1)
        count_m = count_p = count_n = 0
        for d in range(start_day, day + 1):
            if d < len(self.schedule) and emp_id in self.schedule[d]:
                shift = self.schedule[d][emp_id]
                if shift == 0:   count_m += 1
                elif shift == 1: count_p += 1
                elif shift == 2: count_n += 1
        counts = np.array([count_m, count_p, count_n], dtype=np.float32)
        return float(np.std(counts)), {'M': count_m, 'P': count_p, 'N': count_n}

    def _compute_soft_rotation_penalty(self, emp_idx: int, final_shift: int, prev_last_shift, day: int) -> float:
        emp = EMPLOYEES[emp_idx]
        emp_id = emp['id']
        role = ROLE_NAMES[emp['type']]
        shift_name = ALL_SHIFTS[final_shift]
        prev = int(prev_last_shift[emp_id])
        penalty = 0.0

        streak = int(self.same_shift_streak[emp_id] + 1) if prev == final_shift else 1
        limit = CONSECUTIVE_LIMITS[role].get(shift_name, 99)
        if streak > limit:
            if shift_name == 'N':
                penalty += PENALTY_DAILY['consecutive_night'] * (streak - limit)
            elif shift_name == 'R':
                penalty += PENALTY_DAILY['consecutive_rest'] * (streak - limit)
            else:
                penalty += PENALTY_DAILY['consecutive_same_shift'] * (streak - limit)

        peer_indices = [e['id'] for e in EMPLOYEES if e['type'] == emp['type']]

        # Quadratic adaptive rotation penalty: applied to all shift types simultaneously.
        # penalty = -alpha * (emp_count - role_mean)^2 per shift type (M, P, N, MP)
        # The squared deviation makes the penalty grow non-linearly: the longer the agent
        # ignores rotation, the heavier the cost becomes.
        if final_shift in (0, 1, 2, 4):  # Only for active shifts, not Rest or AP (M, P, N, MP)
            for shift_id in (0, 1, 2, 4):  # M, P, N, MP
                counts_peer = self.shift_counts[peer_indices, shift_id]
                role_mean = float(np.mean(counts_peer)) if len(counts_peer) else 0.0
                emp_count = float(self.shift_counts[emp_id, shift_id])
                deviation = emp_count - role_mean
                penalty -= ROTATION_IMBALANCE_ALPHA * (deviation ** 2)

        if self._special_day_types(day) and final_shift not in (3, 6):
            peer_special = self.special_day_work_counts[peer_indices]
            mean_special = float(np.mean(peer_special)) if len(peer_special) else 0.0
            if self.special_day_work_counts[emp_id] > mean_special + 1.0:
                penalty += PENALTY_DAILY['special_day_concentration']
        return penalty

    def _compute_calendar_fairness(self, final_shifts: Dict[int, int], day: int, info: Dict) -> float:
        special_types = self._special_day_types(day)
        if not special_types:
            return 0.0
        reward = 0.0
        special_workers = [emp['id'] for emp in EMPLOYEES if final_shifts.get(emp['id'], 3) not in (3, 6)]
        medics = [eid for eid in special_workers if EMPLOYEES[eid]['type'] == EMPLOYEE_TYPE_MEDICO]
        nurses = [eid for eid in special_workers if EMPLOYEES[eid]['type'] == EMPLOYEE_TYPE_INFERMIERE]
        for group in (medics, nurses):
            if not group:
                continue
            counts = self.special_day_work_counts[group]
            if np.max(counts) - np.min(counts) <= 1:
                reward += REWARD_DAILY['special_day_rotation_bonus']
            else:
                reward += PENALTY_DAILY['special_day_concentration']
                info['soft_violations'].append('special-day imbalance')

        for i, emp in enumerate(EMPLOYEES):
            role = ROLE_NAMES[emp['type']]
            final_shift = final_shifts.get(emp['id'], 3)
            shift_name = ALL_SHIFTS[final_shift]
            if role not in SPECIAL_ROLES or shift_name not in SPECIAL_SHIFT_NAMES:
                continue
            if final_shift not in (0, 1, 2):
                continue
            for special_type in special_types:
                last_emp_id = self._get_last_special_assignee(role, shift_name, special_type)
                if last_emp_id is None:
                    reward += SPECIAL_DAY_ALTERNATION['bonus_if_respected']
                    continue
                if last_emp_id != emp['id']:
                    reward += SPECIAL_DAY_ALTERNATION['bonus_if_respected']
                    continue
                alternative_exists = self._exists_legal_alternative_for_special_slot(i, role, final_shift, day)
                if alternative_exists:
                    reward += SPECIAL_DAY_ALTERNATION['soft_penalty_if_avoidable']
                    info['soft_violations'].append(
                        f"alternanza evitabile non rispettata: {emp['name']} {special_type}-{shift_name}"
                    )
                    info['alternation_derogations'].append(f"avoidable:{emp['name']}:{special_type}:{shift_name}")
                    self.total_soft_violations += 1
                else:
                    reward += SPECIAL_DAY_ALTERNATION['soft_penalty_if_inevitable']
                    info['alternation_derogations'].append(f"inevitable:{emp['name']}:{special_type}:{shift_name}")
        return reward

    def _commit_day(self, final_shifts: Dict[int, int], prev_last_shift, day: int):
        special_types = self._special_day_types(day)
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            shift = int(final_shifts[emp_id])
            self.schedule[day][emp_id] = shift
            hours = SHIFT_HOURS.get(shift, 0)
            self.hours_week[emp_id] += hours
            self.hours_month[emp_id] += hours
            self.shift_counts[emp_id, shift] += 1
            if special_types and shift not in (3, 6):
                self.special_day_work_counts[emp_id] += 1
            # Track weekend work for rotation
            if self._is_weekend(day) and shift not in (3, 6):
                self.weekend_work_counts[emp_id] += 1
            if shift == prev_last_shift[emp_id]:
                self.same_shift_streak[emp_id] += 1
            else:
                self.same_shift_streak[emp_id] = 1
            if shift in (3, 6):
                self.consecutive_rest_days[emp_id] += 1
                self.consecutive_work_days[emp_id] = 0
            else:
                self.consecutive_work_days[emp_id] += 1
                self.consecutive_rest_days[emp_id] = 0
            self.last_shift[emp_id] = shift
            window = self._get_shift_window_for_day(day, shift)
            self.last_shift_end[emp_id] = -1000.0 if window is None else float(window[1])
            if special_types and shift in (0, 1, 2, 4):
                role = ROLE_NAMES[emp['type']]
                for slot in SHIFT_COVERS.get(shift, []):
                    for special_type in special_types:
                        self.last_special_assignment_by_role_shift_type[(role, slot, special_type)] = emp_id

    # ------------------------------------------------------------------
    # Weekly / monthly rewards
    # ------------------------------------------------------------------
    def _compute_weekly_reward(self, week: int):
        reward = 0.0
        info = {}
        week_start, week_end = week * WEEK_LEN, min((week + 1) * WEEK_LEN, self._days_in_episode)
        week_schedule = range(week_start, week_end)

        weeks_without_rest = []
        hours_on_target = []
        weekly_hours_map = {}
        for emp in EMPLOYEES:
            emp_id = emp['id']
            contract = CONTRACT_TYPES[emp['contract']]
            weekly_hours_map[emp['name']] = float(self.hours_week[emp_id])
            has_rest = any(self.schedule[d][emp_id] == 3 for d in week_schedule)
            if not has_rest:
                reward += PENALTY_WEEKLY['week_without_rest']
                weeks_without_rest.append(emp['name'])
            if self.hours_week[emp_id] > contract['weekly_hours'] + contract['max_weekly_overtime']:
                reward += PENALTY_WEEKLY['weekly_limit_exceeded']
                info.setdefault('weekly_limit_exceeded', []).append(emp['name'])
            if self.hours_week[emp_id] >= contract['weekly_hours']:
                reward += REWARD_WEEKLY['hours_on_target']
                hours_on_target.append(emp['name'])
        if not weeks_without_rest:
            reward += REWARD_WEEKLY['weekly_rest_all_ok']
        else:
            info['weeks_without_rest'] = weeks_without_rest
        if hours_on_target:
            info['hours_on_target'] = hours_on_target

        week_role_penalties = []
        for role_type, role_name in ((EMPLOYEE_TYPE_MEDICO, 'Medico'), (EMPLOYEE_TYPE_INFERMIERE, 'Infermiere')):
            ids = [e['id'] for e in EMPLOYEES if e['type'] == role_type]
            if not ids:
                continue
            for shift_id in (0, 1, 2):
                week_counts = [sum(1 for d in week_schedule if self.schedule[d][emp_id] == shift_id) for emp_id in ids]
                if np.std(week_counts) > 1.5:
                    reward += PENALTY_WEEKLY['weekly_role_imbalance']
                    week_role_penalties.append(f"{role_name}-{ALL_SHIFTS[shift_id]}")
        if week_role_penalties:
            info['weekly_role_imbalance'] = week_role_penalties
        else:
            reward += REWARD_WEEKLY['balanced_roles']

        for role_type, role_name in ((EMPLOYEE_TYPE_MEDICO, 'Medico'), (EMPLOYEE_TYPE_INFERMIERE, 'Infermiere')):
            ids = [e['id'] for e in EMPLOYEES if e['type'] == role_type]
            if not ids:
                continue
            night_counts = [sum(1 for d in week_schedule if self.schedule[d][emp_id] == 2) for emp_id in ids]
            if len(night_counts) > 0 and np.std(night_counts) <= 1.0:
                reward += REWARD_WEEKLY['night_balance']

        if self.jolly_used_week.get(week, 0) > 2:
            reward += PENALTY_WEEKLY['too_many_jolly']
            info['too_many_jolly'] = self.jolly_used_week.get(week, 0)
        else:
            reward += REWARD_WEEKLY['week_clean'] * (1.0 if self.jolly_used_week.get(week, 0) == 0 else 0.0)

        # Adaptive coverage deficit penalty
        # Formula LINEARE: coeff * (14 - n) dove n = somma check giornalieri settimana (0..14)
        # n=0: penalità massima (-7.0); n=13: -0.5; n=14: 0.0
        # Segnale forte anche nell'ultimo giorno mancante (vs iperbolica che dava -0.025)
        weekly_checks = self.weekly_coverage_checks.get(week, 0)
        adaptive_deficit = PENALTY_WEEKLY['adaptive_coverage_deficit'] * (14 - weekly_checks)
        reward += adaptive_deficit
        info['adaptive_coverage_deficit'] = round(adaptive_deficit, 4)
        info['weekly_coverage_checks_total'] = weekly_checks

        info['weekly_hours'] = weekly_hours_map
        return reward, info

    def _compute_monthly_reward(self):
        reward = REWARD_MONTHLY['episode_complete']
        info = {}
        monthly_emp = []
        for emp in EMPLOYEES:
            emp_id = emp['id']
            contract = CONTRACT_TYPES[emp['contract']]
            total_hours = float(self.hours_month[emp_id])
            
            # Add 20 minutes bonus for each M and P shift worked
            m_shift_count = int(self.shift_counts[emp_id, 0])  # Mattina
            p_shift_count = int(self.shift_counts[emp_id, 1])  # Pomeriggio
            bonus_hours = (m_shift_count + p_shift_count) * (20.0 / 60.0)  # 20 min per shift = 1/3 hour
            total_hours += bonus_hours
            
            ordinary_target = contract['weekly_hours'] * (self._days_in_episode / WEEK_LEN)
            overtime_proxy = max(0.0, total_hours - ordinary_target)
            item = {
                'dipendente': emp['name'],
                'ruolo': ROLE_NAMES[emp['type']],
                'contratto': contract['name'],
                'ore_totali': total_hours,
                'straordinario_proxy': overtime_proxy,
                'notti': int(self.shift_counts[emp_id, 2]),
                'mattine': int(self.shift_counts[emp_id, 0]),
                'pomeriggi': int(self.shift_counts[emp_id, 1]),
                'riposi': int(self.shift_counts[emp_id, 3]),
                'mp': int(self.shift_counts[emp_id, 4]),
                'ap': int(self.shift_counts[emp_id, 6]),
                'giorni_speciali_lavorati': int(self.special_day_work_counts[emp_id]),
            }
            monthly_emp.append(item)
            if overtime_proxy > MAX_MONTHLY_OVERTIME_PROXY:
                reward += PENALTY_MONTHLY['monthly_overtime_proxy_exceeded']
        self.monthly_kpi_by_employee = monthly_emp

        full_coverage = True
        for d in range(self._days_in_episode):
            cov = self._check_coverage_detail(self.schedule[d])
            if any(cov[s][r]['missing'] > 0 for s in COVERAGE_SLOTS for r in ROLE_ORDER):
                full_coverage = False
                break
        if full_coverage:
            reward += REWARD_MONTHLY['full_coverage']
            info['full_coverage'] = True

        role_kpis = []
        for role_name in ROLE_ORDER:
            rows = [r for r in monthly_emp if r['ruolo'] == role_name]
            if not rows:
                continue
            nights = [r['notti'] for r in rows]
            ms = [r['mattine'] for r in rows]
            ps = [r['pomeriggi'] for r in rows]
            specials = [r['giorni_speciali_lavorati'] for r in rows]
            if np.std(nights) <= 1.25 and np.std(ms) <= 1.75 and np.std(ps) <= 1.75:
                reward += REWARD_MONTHLY['good_shift_balance']
            else:
                reward += PENALTY_MONTHLY['monthly_shift_imbalance']
            if np.std(specials) <= 1.25:
                reward += REWARD_MONTHLY['good_special_day_balance']
            else:
                reward += PENALTY_MONTHLY['monthly_special_day_imbalance']
            role_kpis.append({
                'ruolo': role_name,
                'ore_totali': float(sum(r['ore_totali'] for r in rows)),
                'notti_totali': int(sum(r['notti'] for r in rows)),
                'special_days_totali': int(sum(r['giorni_speciali_lavorati'] for r in rows)),
            })

        info['monthly_hours_imbalance'] = []
        all_balanced = True
        for role_name in ROLE_ORDER:
            rows = [r for r in monthly_emp if r['ruolo'] == role_name]
            if not rows:
                continue
            hours_list = [r['ore_totali'] for r in rows]
            role_mean = float(np.mean(hours_list))
            for r in rows:
                delta = abs(r['ore_totali'] - role_mean)
                if delta > MONTHLY_HOURS_TOLERANCE:
                    all_balanced = False
                    reward += PENALTY_MONTHLY['monthly_hours_imbalance']
                    info['monthly_hours_imbalance'].append(
                        f"{r['dipendente']}: {r['ore_totali']:.0f}h (media {role_mean:.0f}h, delta {delta:.0f}h)"
                    )
        if all_balanced:
            reward += REWARD_MONTHLY['monthly_hours_balanced']
            info['monthly_hours_balanced'] = True

        self.monthly_kpi_by_role = role_kpis
        info['employee_kpis'] = monthly_emp
        info['role_kpis'] = role_kpis
        info['total_jolly'] = self.total_jolly
        info['total_mp'] = self.total_mp
        info['total_hard_overrides'] = self.total_hard_overrides
        info['total_soft_violations'] = self.total_soft_violations
        info['total_coverage_violations'] = self.total_coverage_violations
        return reward, info

    # ------------------------------------------------------------------
    # Observation / masking / render
    # ------------------------------------------------------------------
    def _availability_counts(self, day: int):
        counts = []
        for role_type in (EMPLOYEE_TYPE_MEDICO, EMPLOYEE_TYPE_INFERMIERE):
            idxs = [i for i, e in enumerate(EMPLOYEES) if e['type'] == role_type]
            denom = max(1, len(idxs))
            for action in (0, 1, 2):
                legal = sum(1 for i in idxs if self._is_action_legal(i, action, day))
                counts.append(legal / denom)
        return counts

    def _worked_last_same_special(self, emp_idx: int, day: int) -> float:
        emp = EMPLOYEES[emp_idx]
        role = ROLE_NAMES[emp['type']]
        shift_id = int(self.last_shift[emp['id']])
        shift_name = ALL_SHIFTS[shift_id]
        if shift_name not in SPECIAL_SHIFT_NAMES or role not in SPECIAL_ROLES:
            return 0.0
        for special_type in self._special_day_types(day):
            if self._get_last_special_assignee(role, shift_name, special_type) == emp['id']:
                return 1.0
        return 0.0

    def _get_obs(self):
        day = min(self.current_day, self._days_in_episode - 1)
        weekday_idx = self._weekday_idx(day)
        weekday_oh = [1.0 if weekday_idx == k else 0.0 for k in range(7)]
        global_feats = [
            day / max(1, self._days_in_episode - 1),
            *weekday_oh,
            float(self._is_weekend(day)),
            float(self._is_holiday(day)),
            float(self._is_prefestive(day)),
            float(bool(self._special_day_types(day))),
            *self._availability_counts(day),
        ]
        obs = list(global_feats)
        pref_type_ids = {name: idx + 1 for idx, name in enumerate(sorted(PREFERENCE_TYPES.keys()))}
        n_pref_types = len(pref_type_ids)
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            contract = CONTRACT_TYPES[emp['contract']]
            max_week = contract['weekly_hours'] + contract['max_weekly_overtime']
            max_month = contract['weekly_hours'] * 4 + MAX_MONTHLY_OVERTIME_PROXY
            legal_flags = [1.0 if self._is_action_legal(i, a, day) else 0.0 for a in SHIFT_IDS]
            pref_key = (emp_id, day)
            pref_val = 0.0
            if pref_key in self.preferences_table:
                pref_val = pref_type_ids.get(self.preferences_table[pref_key], 0) / n_pref_types
            obs.extend([
                float(emp['type']),
                min(1.0, self.hours_week[emp_id] / max_week),
                min(1.0, self.hours_month[emp_id] / max_month),
                self.last_shift[emp_id] / 6.0,
                float((emp_id, day) in self.ferie_table),
                float(self.ap_today[i]),
                float(self.rest_today[i]),
                pref_val,
                min(1.0, self.same_shift_streak[emp_id] / 4.0),
                min(1.0, self.consecutive_work_days[emp_id] / 6.0),
                min(1.0, self.consecutive_rest_days[emp_id] / 3.0),
                min(1.0, self.special_day_work_counts[emp_id] / 8.0),
                self._worked_last_same_special(i, day),
                *legal_flags,
            ])
        return np.array(obs, dtype=np.float32)

    def action_masks(self):
        mask = []
        day = self.current_day
        for i, emp in enumerate(EMPLOYEES):
            emp_id = emp['id']
            # Se il dipendente ha AP, riposo pianificato o ferie oggi,
            # l'unica azione consentita è R (3) — evita hard_override e segnali contraddittori.
            forced_rest = (
                (self.ap_today is not None and self.ap_today[i])
                or (self.rest_today is not None and self.rest_today[i])
                or (emp_id, day) in self.ferie_table
            )
            # Regola weekend medici: se un medico ha lavorato sabato, domenica è riposo obbligatorio.
            # Questo lascia libero l'agente di decidere SABATO chi mettere e in quale turno,
            # sapendo che quella persona non potrà lavorare domenica.
            if (
                not forced_rest
                and emp['type'] == EMPLOYEE_TYPE_MEDICO
                and self._weekday_idx(day) == 6  # domenica
            ):
                sat_shift = self._get_saturday_shift_medico(i, day)
                if sat_shift is not None and sat_shift not in (3, 6):  # ha lavorato sabato
                    forced_rest = True
            for action in SHIFT_IDS:
                if forced_rest:
                    mask.append(action == 3)
                else:
                    mask.append(self._is_action_legal(i, action, day))
        return np.asarray(mask, dtype=bool)

    def render(self):
        header = ['Dipendente'] + [self._date_label(d) for d in range(min(self.current_day, self._days_in_episode))]
        print(' | '.join(header))
        for emp in EMPLOYEES:
            row = [emp['name']]
            for d in range(min(self.current_day, self._days_in_episode)):
                row.append(ALL_SHIFTS[self.schedule[d][emp['id']]])
            print(' | '.join(row))

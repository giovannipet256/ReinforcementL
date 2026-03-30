from __future__ import annotations

# ==============================================================================
# REWARD WEIGHTS
# Moltiplicatori globali per categoria. Usati in step() per scalare ogni
# reward_components[key] prima di sommare il reward totale.
# Chiavi obbligatorie (usate nell'environment): coverage, legality, fairness,
# efficiency, calendar, weekly, monthly, preference.
# ==============================================================================
REWARD_WEIGHTS_BASE = {
    'coverage':   1.0,
    'legality':   1.0,
    'fairness':   1.0,
    'efficiency': 1.0,
    'calendar':   1.0,
    'weekly':     1.0,
    'monthly':    1.0,
    'preference': 1.0,
}

# ==============================================================================
# ROTATION IMBALANCE ALPHA
# Coefficiente per la penalità quadratica sulla rotazione dei turni.
# Formula: penalty = -ALPHA * (conteggio_dipendente - media_ruolo)^2
# Applicata per ogni tipo di turno (M, P, N, MP) in _compute_soft_rotation_penalty.
# ==============================================================================
ROTATION_IMBALANCE_ALPHA = 0.15   # ← imposta il valore desiderato

# ==============================================================================
# REWARD / PENALTY GIORNALIERI
# Applicati ad ogni step (ogni giorno simulato).
# Riferimento rapido alle chiavi usate nell'environment:
#
#   REWARD_DAILY
#   ├── slot_covered               → +reward per ogni slot coperto (coverage)
#   ├── all_coverage_met           → bonus se tutti gli slot sono coperti
#   ├── valid_assignment           → assegnazione legale confermata (preference)
#   ├── underused_role_shift_bonus → (non usata in env, disponibile)
#   ├── special_day_rotation_bonus → bilanciamento giorni speciali (calendar)
#   ├── night_rotation_bonus       → (non usata in env, disponibile)
#   ├── weekly_rotation_bonus      → (non usata in env, disponibile)
#   ├── mp_weekend_bonus           → turno MP il weekend (efficiency)
#   ├── jolly_weekend_bonus        → jolly nel weekend (efficiency)
#   ├── complete_daily_coverage    → copertura completa (coverage)
#   ├── dirigenza_daily_coverage   → tutti i turni medici coperti (coverage)
#   ├── dirigenza_night_covered    → notte medica coperta (coverage)
#   ├── infermieri_daily_coverage  → tutti i turni infermieri coperti (coverage)
#   ├── infermieri_night_covered   → notte infermieri coperta (coverage)
#   ├── preference_respected       → preferenza rispettata (preference)
#   └── weekday_complete_coverage  → dirigenza+comparto entrambi coperti (solo lun-ven) (coverage)
#
#   PENALTY_DAILY
#   ├── hard_override              → azione illegale forzata a riposo (legality)
#   ├── ferie_violated             → ferie non rispettate (legality)
#   ├── ap_violated                → AP non rispettato (legality)
#   ├── rest_day_violated          → riposo settimanale violato (legality)
#   ├── rest_11h_violated          → (non usata in env, disponibile)
#   ├── hours_exceeded             → (non usata in env, disponibile)
#   ├── slot_uncovered_medico      → slot non coperto - medico (coverage)
#   ├── slot_uncovered_infermiere  → slot non coperto - infermiere (coverage)
#   ├── mp_activated               → turno MP attivato come riparo (efficiency)
#   ├── jolly_inevitable           → jolly inevitabile (efficiency)
#   ├── jolly_avoidable            → jolly evitabile (efficiency)
#   ├── unjustified_rest           → riposo non giustificato (preference)
#   ├── consecutive_same_shift     → stesso turno consecutivo (fairness)
#   ├── consecutive_rest           → riposi consecutivi (fairness)
#   ├── consecutive_night          → notti consecutive (fairness)
#   ├── role_shift_imbalance       → (non usata in env, disponibile)
#   ├── special_day_concentration  → concentrazione su giorni speciali (fairness/calendar)
#   ├── weekly_rotation_imbalance  → (non usata in env, disponibile)
#   ├── weekend_consecutive_work   → lavoro su sabato e domenica (fairness)
#   ├── weekday_medico_distribution→ distribuzione turni medici feriali (coverage)
#   ├── dirigenza_m_uncovered      → mattina medici scoperta (coverage)
#   ├── dirigenza_p_uncovered      → pomeriggio medici scoperto (coverage)
#   ├── dirigenza_n_uncovered      → notte medici scoperta (coverage)
#   ├── infermieri_m_uncovered     → mattina infermieri scoperta (coverage)
#   ├── infermieri_p_uncovered     → pomeriggio infermieri scoperto (coverage)
#   ├── infermieri_n_uncovered     → notte infermieri scoperta (coverage)
#   ├── preference_violated        → preferenza violata (preference)
#   └── consecutive_weekend_work   → weekend consecutivi lavorati (fairness)
# ==============================================================================

REWARD_DAILY = {
    'riposo_rispettato':          1.1,
    'copertura_n_minima':         1.1,
    'shift_distribution':         1.1,
    'ap_rispettato':              1.1,
    'riposo_settimanale_rispettato':      1.1,
    'slot_covered':               0.7,
    'all_coverage_met':           1.4,
    'valid_assignment':           0.5,
    'underused_role_shift_bonus': 0.5,
    'special_day_rotation_bonus': 0.6,
    'night_rotation_bonus':       0.5,
    'weekly_rotation_bonus':      0.4,
    'mp_weekend_bonus':           0.7,
    'jolly_weekend_bonus':        0.0,
    'complete_daily_coverage':    1.2,
    'dirigenza_daily_coverage':   0.8,
    'dirigenza_night_covered':    1.0,
    'infermieri_daily_coverage':  0.8,
    'infermieri_night_covered':   1.0,
    'preference_respected':       0.3,
    'weekday_complete_coverage':  0.8,  # Reward quando sia dirigenza che comparto coperti (lun-ven)
    'medico_infermiere_ratio':    0.5,
    # New combined coverage check rewards (medici + infermieri minimal coverage)
    'coverage_first_check_pass': 3.0,  # Strong reward when all shifts have at least 1 medico and 1 infermiere
    'coverage_residual_pass': 1.5,     # Moderate reward when residual respects percentage limits
    # Shift proportionality check rewards (infermieri vs medici ratio)
    'shift_proportion_all_pass': 0.5,  # Reward when all shifts (M, P, N) respect proportionality constraints
    'consecutive_weekend_rest_recovery': 1.3,  # Reward quando dirigente riposa dopo aver lavorato il weekend precedente
}

PENALTY_DAILY = {
    'riposo_violato':                -1.0,  # (non usata in env, disponibile)
    'copertura_n_minima_violata':    -3.0,  # (non usata in env, disponibile)
    'shift_distribution_violata':    -1.5,  # (non usata in env, disponibile)
    'hard_override':                 -1.5,
    'ferie_violated':                -2.0,
    'ap_violated':                   -1.0,
    'rest_day_violated':             -1.0,
    'rest_11h_violated':             -1.2,
    'hours_exceeded':                -1.0,
    'slot_uncovered_medico':         -1.0,
    'slot_uncovered_infermiere':     -0.8,
    'mp_activated':                  -0.3,
    'jolly_inevitable':              -0.5,
    'jolly_avoidable':               -0.9,
    'unjustified_rest':              -0.7,###
    'consecutive_same_shift':        -0.4,
    'consecutive_rest':              -0.5,
    'consecutive_night':             -1.0,###
    'role_shift_imbalance':          -0.4,
    'special_day_concentration':     -0.5,
    'weekly_rotation_imbalance':     -0.3,
    'weekend_consecutive_work':      -0.7,
    'weekday_medico_distribution':   -0.4,
    'dirigenza_m_uncovered':         -0.9,
    'dirigenza_p_uncovered':         -0.9,
    'dirigenza_n_uncovered':         -0.9,
    'infermieri_m_uncovered':        -0.8,
    'infermieri_p_uncovered':        -0.8,
    'infermieri_n_uncovered':        -0.8,
    'preference_violated':           -0.3,
    'consecutive_weekend_work':      -0.8,
    'consecutive_weekend_work_multi': -1.2,  # Penalità quando dirigente lavora 2+ weekend di fila
    # New combined coverage check penalties (medici + infermieri minimal coverage)
    'coverage_first_check_fail': -3.0,  # Strong penalty when minimal coverage (1 medico + 1 infermiere per shift) not met
    'coverage_residual_fail': -0.8,     # Soft penalty when residual exceeds percentage limits
    # Shift proportionality check penalties (infermieri vs medici ratio)
    'shift_proportion_fail': -0.6,      # Soft penalty when at least one shift proportionality is not met
}

# ==============================================================================
# REWARD / PENALTY SETTIMANALI
# Applicati alla fine di ogni settimana (ogni 7 giorni) in _compute_weekly_reward.
#
#   REWARD_WEEKLY
#   ├── week_clean          → settimana senza jolly (efficiency)
#   ├── weekly_rest_all_ok  → tutti i dipendenti hanno avuto almeno un riposo
#   ├── hours_on_target     → ore settimanali raggiunte (per dipendente)
#   ├── balanced_roles      → distribuzione turni equilibrata per ruolo
#   └── night_balance       → bilanciamento notti per ruolo
#
#   PENALTY_WEEKLY
#   ├── weekly_limit_exceeded     → superamento ore settimanali massime
#   ├── week_without_rest         → settimana senza riposo (per dipendente)
#   ├── weekly_role_imbalance     → turni squilibrati per ruolo nella settimana
#   ├── too_many_jolly            → troppi jolly usati nella settimana
#   └── adaptive_coverage_deficit → penalità adattiva copertura lineare: coeff * (14 - n)
#                                   n = somma check giornalieri settimana (max 14)
#                                   n=0: penalità max = -7.0; n=13: -0.5; n=14: 0.0
#                                   (segnale forte anche per n vicino a 14, a diff. dell'iperbolica)
# ==============================================================================

REWARD_WEEKLY = {
    'week_clean':          1.0,
    'weekly_rest_all_ok':  1.2,
    'hours_on_target':     0.8,
    'balanced_roles':      1.0,
    'night_balance':       0.9,
}

PENALTY_WEEKLY = {
    'weekly_limit_exceeded':       -1.2,
    'week_without_rest':           -1.5,
    'weekly_role_imbalance':       -0.7,
    'too_many_jolly':              -1.0,
    'adaptive_coverage_deficit':   -0.5,###   # Moltiplicatore: penalità lineare = coeff * (14 - n)
}

# ==============================================================================
# REWARD / PENALTY MENSILI
# Applicati alla fine dell'episodio (ultimo giorno) in _compute_monthly_reward.
#
#   REWARD_MONTHLY
#   ├── episode_complete           → bonus fisso per completamento episodio
#   ├── full_coverage              → copertura completa su tutti i giorni
#   ├── good_special_day_balance   → giorni speciali distribuiti bene per ruolo
#   ├── good_shift_balance         → turni M/P/N distribuiti bene per ruolo
#   └── monthly_hours_balanced     → ore mensili bilanciate per ruolo
#
#   PENALTY_MONTHLY
#   ├── monthly_overtime_proxy_exceeded → straordinario mensile superato
#   ├── monthly_shift_imbalance         → turni mensili squilibrati per ruolo
#   ├── monthly_special_day_imbalance   → giorni speciali mensili squilibrati
#   └── monthly_hours_imbalance         → ore mensili sbilanciate per dipendente
# ==============================================================================

REWARD_MONTHLY = {
    'episode_complete':         1.5,
    'full_coverage':            2.0,
    'good_special_day_balance': 1.2,
    'good_shift_balance':       1.3,
    'monthly_hours_balanced':   1.4,
}

PENALTY_MONTHLY = {
    'monthly_overtime_proxy_exceeded':  -1.8,
    'monthly_shift_imbalance':          -1.2,
    'monthly_special_day_imbalance':    -1.0,
    'monthly_hours_imbalance':          -1.5,
}

# ==============================================================================
# COVERAGE CHECK CONFIGURATION
# Parametri per i check di copertura combinata (medici + infermieri)
# ==============================================================================

# Combined coverage check parameters
# After subtracting 1 from each shift type (M, P, N) in residual count,
# the percentage of remaining staff per shift must not exceed these thresholds
COVERAGE_RESIDUAL_MAX_PERCENTAGES = {
    'res_m_pct': 0.80,  # Max 80% on morning shift after subtracting 1
    'res_p_pct': 0.40,  # Max 40% on afternoon shift after subtracting 1
    'res_n_pct': 0.30,  # Max 30% on night shift after subtracting 1
}

# Shift proportionality check parameters
# For each shift, the number of infermieri should respect: n_infermieri == floor(lambda * n_medici)
# where lambda is the proportionality factor (infermieri per medico)
SHIFT_PROPORTIONALITY_LAMBDAS = {
    'lambda_m': 2.0,  # Morning: 1 medico → 2 infermieri (coerente con COVERAGE_REQUIREMENTS M→Medico:1,Infermiere:2)
    'lambda_p': 2.0,  # Afternoon: 1 medico → 2 infermieri (coerente con COVERAGE_REQUIREMENTS P→Medico:1,Infermiere:2)
    'lambda_n': 1.0,  # Night: 1 medico → 1 infermiere (coerente con COVERAGE_REQUIREMENTS N→Medico:1,Infermiere:1)
}

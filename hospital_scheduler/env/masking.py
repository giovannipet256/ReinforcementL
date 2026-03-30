from __future__ import annotations

import numpy as np
import gymnasium as gym


class HospitalActionMasker(gym.Wrapper):
    """
    Wrapper che espone le action mask all'algoritmo PPO (tramite MaskablePPO
    di sb3-contrib). L'environment interno calcola le mask in action_masks();
    questo wrapper le rende accessibili all'esterno e fornisce utility di debug.

    COME FUNZIONA IL MASKING:
    ─────────────────────────
    Per ogni dipendente e ogni turno possibile, la mask vale:
        True  → azione CONSENTITA (il PPO può sceglierla)
        False → azione BLOCCATA   (il PPO non può sceglierla)

    La logica reale che produce True/False vive in:
        HospitalSchedulingEnv._is_action_legal(emp_idx, action, day)

    Quel metodo chiama in cascata:
        1. _is_action_legal_base  → vincoli hard (ore, riposo, ferie, AP...)
        2. _violates_special_day_alternation → alternanza giorni speciali

    Per aggiungere una nuova regola di masking aggiungi un controllo in uno
    dei due metodi sopra (vedi GUIDA nel file README_REWARD_GUIDE.md).

    STRUTTURA DEL FLAT ARRAY:
    ─────────────────────────
    action_masks() restituisce un array 1-D di bool di lunghezza:
        N_EMPLOYEES × len(AGENT_SHIFTS)

    Ordinamento: [emp_0_shift_0, emp_0_shift_1, ..., emp_1_shift_0, ...]
    Turni (AGENT_SHIFTS): 0=M, 1=P, 2=N, 3=R
    """

    def __init__(self, env):
        super().__init__(env)

    # ------------------------------------------------------------------
    # Interfaccia richiesta da MaskablePPO (sb3-contrib)
    # ------------------------------------------------------------------

    def action_masks(self):
        """Delega all'environment sottostante. Non modificare."""
        return self.env.action_masks()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def action_mask_matrix(self):
        """
        Restituisce le mask come matrice 2-D: shape (N_EMPLOYEES, N_ACTIONS).
        Utile per debug e ispezione.
        """
        flat = np.asarray(self.action_masks(), dtype=bool)
        nvec = self.action_space.nvec          # [N_ACTIONS] * N_EMPLOYEES
        n_actions = int(nvec[0])
        return flat.reshape(len(nvec), n_actions)

    def explain_current_masks(self):
        """
        Restituisce una lista di dict leggibili con lo stato delle mask
        per il giorno corrente. Aggiorna i label se aggiungi nuovi turni.

        Esempio output:
            [{'employee_index': 0, 'M': True, 'P': True, 'N': False, 'R': True}, ...]
        """
        matrix = self.action_mask_matrix()
        labels = []
        for emp_idx in range(matrix.shape[0]):
            entry = {'employee_index': emp_idx}
            # ── Aggiungi qui nuovi turni se estendi AGENT_SHIFTS ──────────
            shift_names = ['M', 'P', 'N', 'R']
            for col, name in enumerate(shift_names):
                if col < matrix.shape[1]:
                    entry[name] = bool(matrix[emp_idx, col])
            # ─────────────────────────────────────────────────────────────
            labels.append(entry)
        return labels

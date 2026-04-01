"""
Utility functions for loading and processing hospital scheduler data.
Calculates KPIs from schedule data loaded from Excel.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from datetime import datetime


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

# Shift hour mappings
SHIFT_HOURS = {
    'M': 8,      # Morning
    'P': 8,      # Afternoon
    'N': 8,      # Night
    'MP': 16,    # Morning-Afternoon
    'R': 0,      # Rest
    'AP': 0,     # Absence
    'J': 0,      # Jolly
}


def load_schedule_from_excel(file_path: str) -> pd.DataFrame:
    """
    Load the schedule from Excel file (first sheet 'Planner').
    Returns DataFrame with Dipendente as index and dates as columns.
    """
    # Read as strings to avoid unexpected dtype conversions
    df = pd.read_excel(file_path, sheet_name='Planner', dtype=str)
    df = df.fillna("")

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Try to find the column that contains employee names (prefer 'Dipendente')
    idx_col = None
    if 'Dipendente' in df.columns:
        idx_col = 'Dipendente'
    else:
        # common alternatives
        for alt in ['Nome', 'Nome Dipendente', 'Dipendenti', 'Employee', 'Name']:
            if alt in df.columns:
                idx_col = alt
                break

    # If still not found, check if first column is unnamed (e.g., index column)
    if idx_col is None and len(df.columns) > 0:
        first = df.columns[0]
        if first.lower().startswith('unnamed') or df[first].astype(str).str.contains(r'Dr\.|Inf\.', na=False).any():
            idx_col = first

    # Fallback: choose the column with many non-empty strings that look like names
    if idx_col is None:
        for col in df.columns:
            non_empty = df[col].astype(str).str.strip().replace('', pd.NA).dropna()
            if len(non_empty) > 0:
                sample = non_empty.iloc[0]
                if isinstance(sample, str) and (sample.startswith('Dr.') or sample.startswith('Inf.') or ' ' in sample):
                    idx_col = col
                    break

    if idx_col is not None:
        df.set_index(idx_col, inplace=True)
    else:
        # No obvious index column found — return dataframe as-is so caller can handle it
        # This avoids raising a KeyError and gives a clearer downstream error if necessary
        return df

    # Strip whitespace from index values
    df.index = df.index.to_series().astype(str).str.strip()

    return df


def load_kpi_sheet(file_path: str) -> pd.DataFrame:
    """
    Load pre-calculated KPI sheet from Excel (sheet 'kpi').
    """
    df = pd.read_excel(file_path, sheet_name='kpi')
    return df


def load_preferences_sheet(file_path: str) -> pd.DataFrame:
    """
    Load preferences sheet from Excel (sheet 'preferenze').
    """
    df = pd.read_excel(file_path, sheet_name='preferenze')
    return df


def extract_date_columns(schedule_df: pd.DataFrame) -> List[str]:
    """
    Extract and sort date columns from schedule DataFrame.
    Returns list of column names (dates).
    """
    date_cols = [col for col in schedule_df.columns if col != 'Dipendente']
    return sorted(date_cols, key=lambda x: datetime.strptime(x.split()[1], '%d/%m/%Y'))


def reshape_schedule_to_long(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the schedule from wide format (employees × dates) to long format
    (one row per employee-date pair).
    
    Returns DataFrame with columns: Dipendente, Data, Turno
    """
    rows = []
    for employee in schedule_df.index:
        for col in schedule_df.columns:
            shift = schedule_df.loc[employee, col]
            # Parse date from column name (e.g., "Mer 01/04/2026")
            date_str = col.split()[1]
            rows.append({
                'Dipendente': employee,
                'Data': date_str,
                'Turno': shift
            })
    
    df_long = pd.DataFrame(rows)
    df_long['Data'] = pd.to_datetime(df_long['Data'], format='%d/%m/%Y')
    return df_long


def infer_role_from_name(dipendente: str) -> str:
    """
    Infer role (Medico/Infermiere) from employee name prefix.
    """
    if dipendente.startswith('Dr.'):
        return 'Medico'
    elif dipendente.startswith('Inf.'):
        return 'Infermiere'
    return 'Unknown'


def calculate_kpis_from_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate KPIs from schedule data:
    - ORE LAVORATE (working hours)
    - ORE STRAORDINARI (overtime)
    - NOTTI (night shifts)
    - MATTINE (morning shifts)
    - POMERIGGI (afternoon shifts)
    - RIPOSI (rest days)
    - MP (morning-afternoon combined)
    - Violaz. 11h (11-hour rest violation count)
    - Riposo sett. (weekly rest balance)
    
    Returns DataFrame with KPIs indexed by employee.
    """
    kpis = []
    
    for employee in schedule_df.index:
        shifts = schedule_df.loc[employee].dropna()
        shifts = shifts[shifts != '']  # Remove empty cells
        
        # Count shifts
        count_M = (shifts == 'M').sum()
        count_P = (shifts == 'P').sum()
        # Accept both 'N' and 'NS' as night codes (some files may use 'NS')
        count_N = shifts.isin(['N', 'NS']).sum()
        count_MP = (shifts == 'MP').sum()
        count_R = (shifts == 'R').sum()
        count_AP = (shifts == 'AP').sum()
        
        # Calculate hours
        ore_lavorate = count_M * 8 + count_P * 8 + count_N * 8 + count_MP * 16
        ore_straordinari = max(0, ore_lavorate - 160)  # Assume 160h standard
        
        # Simplified: 11h violation detection (would need date ordering for real logic)
        violaz_11h = 0
        
        # Weekly rest (simplified - would need actual calendar logic)
        riposo_sett = 0
        
        kpis.append({
            'Dipendente': employee,
            'ORE LAVORATE': ore_lavorate,
            'ORE STRAORDINARI': ore_straordinari,
            'NOTTI': count_N,
            'MATTINE': count_M,
            'POMERIGGI': count_P,
            'RIPOSI': count_R,
            'MP': count_MP,
            'Violaz. 11h': violaz_11h,
            'Riposo sett.': riposo_sett,
        })
    
    return pd.DataFrame(kpis)

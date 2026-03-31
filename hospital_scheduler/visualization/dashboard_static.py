
"""
Hospital Scheduler Dashboard - Simple view mode.
Loads schedule from Excel file and displays calendar with KPIs.
No schedule generation or prediction - just visualization.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import streamlit as st
from datetime import datetime

from reports_static import (
    load_schedule_from_excel,
    load_kpi_sheet,
    load_preferences_sheet,
    extract_date_columns,
    reshape_schedule_to_long,
    infer_role_from_name,
)

# ============================================================
# Configuration
# ============================================================

st.set_page_config(page_title="Hospital Scheduler Dashboard", layout="wide")

# Color mapping for shifts (original colors from previous dashboard)
SHIFT_COLORS_MAP = {
    'M': {'bg': '#dbeafe', 'text': '#1e3a8a'},       # Light blue bg, dark blue text
    'P': {'bg': '#fef3c7', 'text': '#92400e'},       # Light yellow bg, brown text
    'N': {'bg': '#1a237e', 'text': '#ffffff'},       # Dark blue bg, white text
    'R': {'bg': '#dcfce7', 'text': '#166534'},       # Light green bg, dark green text
    'MP': {'bg': '#fde68a', 'text': '#7c2d12'},      # Yellow bg, brown text
    'AP': {'bg': '#f3e8ff', 'text': '#6b21a8'},      # Light purple bg, dark purple text
    'J': {'bg': '#fee2e2', 'text': '#991b1b'},       # Light red bg, dark red text
}

SHIFT_LABELS = {
    'M': 'Mattina',
    'P': 'Pomeriggio',
    'N': 'Notte',
    'MP': 'Mattina-Pom',
    'R': 'Riposo',
    'AP': 'Assenza',
    'J': 'Jolly',
}

# ============================================================
# Utility Functions
# ============================================================

def create_calendar_table(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a calendar table for display with shift information.
    Returns a formatted DataFrame suitable for HTML/table display.
    """
    # Already in wide format: employees × dates
    return schedule_df.copy()


def style_calendar_cell(val: str) -> str:
    """
    Apply styling to calendar cells based on shift type (original colors).
    """
    if pd.isna(val) or val == '':
        return 'background-color: #f0f0f0; color: #999;'
    
    shift_key = str(val)
    if shift_key in SHIFT_COLORS_MAP:
        colors = SHIFT_COLORS_MAP[shift_key]
        bold = 'font-weight: bold;' if shift_key in ['MP', 'J'] else ''
        return f"background-color: {colors['bg']}; color: {colors['text']}; text-align: center; {bold}"
    
    return 'background-color: #f0f0f0; color: #999;'


def aggregate_shift_counts(schedule_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Count occurrences of each shift type per employee.
    Returns dict: {employee: {shift_type: count}}
    """
    counts = {}
    for employee in schedule_df.index:
        shifts = schedule_df.loc[employee].dropna()
        shifts = shifts[shifts != '']
        shift_counts = shifts.value_counts().to_dict()
        counts[employee] = shift_counts
    return counts


def build_kpi_table(schedule_df: pd.DataFrame, kpi_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build KPI table from schedule or from pre-loaded KPI sheet.
    If kpi_df is provided, use it directly; otherwise calculate from schedule.
    """
    if kpi_df is not None and not kpi_df.empty:
        return kpi_df.copy()
    
    # Calculate from schedule
    kpis = []
    shift_counts = aggregate_shift_counts(schedule_df)
    
    for employee in schedule_df.index:
        shifts = schedule_df.loc[employee].dropna()
        shifts = shifts[shifts != '']
        
        # Count shifts
        counts = shifts.value_counts().to_dict()
        m_count = counts.get('M', 0)
        p_count = counts.get('P', 0)
        n_count = counts.get('N', 0)
        mp_count = counts.get('MP', 0)
        r_count = counts.get('R', 0)
        ap_count = counts.get('AP', 0)
        
        # Calculate hours (standard shift = 8 hours, MP = 16)
        ore_lavorate = m_count * 8 + p_count * 8 + n_count * 8 + mp_count * 16
        
        # Overtime (assume 160 hours is standard monthly)
        ore_straordinari = max(0, ore_lavorate - 160)
        
        kpis.append({
            'Dipendente': employee,
            'ORE LAVORATE': ore_lavorate,
            'ORE STRAORDINARI': ore_straordinari,
            'NOTTI': n_count,
            'MATTINE': m_count,
            'POMERIGGI': p_count,
            'RIPOSI': r_count,
            'ASSENZE': ap_count,
            'MP': mp_count,
        })
    
    return pd.DataFrame(kpis)


def build_preference_table(pref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format preferences table for display.
    """
    if pref_df.empty:
        return pref_df
    
    return pref_df.copy()


# ============================================================
# Main Dashboard
# ============================================================

def main():
    st.title("🏥 Hospital Scheduler Dashboard")
    st.markdown("Visualizza il calendario delle assegnazioni e analizza i KPI")
    
    # Sidebar: File upload
    with st.sidebar:
        st.header("📂 Carica Schedule")
        uploaded_file = st.file_uploader(
            "Seleziona file Excel (.xlsx)",
            type=['xlsx'],
            help="Carica il file Excel con i fogli: Planner, kpi, preferenze"
        )
    
    if uploaded_file is None:
        st.info("👈 Carica un file Excel per iniziare")
        return
    
    # # Save uploaded file temporarily
    # temp_path = Path(f"/tmp/uploaded_{uploaded_file.name}")
    # with open(temp_path, 'wb') as f:
    #     f.write(uploaded_file.getbuffer())
    # Save uploaded file temporarily
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"uploaded_{uploaded_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Load data from Excel
        schedule_df = load_schedule_from_excel(str(temp_path))
        
        # Try to load pre-calculated KPIs
        try:
            kpi_df = load_kpi_sheet(str(temp_path))
        except:
            kpi_df = None
        
        # Try to load preferences
        try:
            pref_df = load_preferences_sheet(str(temp_path))
        except:
            pref_df = None
        
        # Display file info
        st.sidebar.success(f"✓ Caricato: {uploaded_file.name}")
        st.sidebar.info(f"📊 Dipendenti: {len(schedule_df)}")
        date_cols = extract_date_columns(schedule_df)
        if date_cols:
            st.sidebar.info(f"📅 Giorni: {len(date_cols)}")
        
        # ============================================================
        # Main Content
        # ============================================================
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Dipendenti", len(schedule_df))
        col2.metric("📅 Giorni", len(date_cols) if date_cols else 0)
        col3.metric("🔄 Turni unici", len(SHIFT_LABELS))
        
        # Count total working days
        schedule_long = reshape_schedule_to_long(schedule_df)
        total_shifts = len(schedule_long[~schedule_long['Turno'].isin(['R', 'AP', ''])])
        col4.metric("📌 Assegnazioni", total_shifts)
        
        # ============================================================
        # CALENDAR SECTION
        # ============================================================
        st.subheader("📋 Calendario Assegnazioni")
        
        # Create styled calendar DataFrame
        calendar_display = schedule_df.copy()
        
        # Style the calendar
        styled_calendar = calendar_display.style.map(style_calendar_cell)
        st.dataframe(styled_calendar, use_container_width=True, height=400)
        
        # Shift legend
        with st.expander("🎨 Legenda colori", expanded=False):
            cols = st.columns(4)
            for i, (shift_code, shift_label) in enumerate(SHIFT_LABELS.items()):
                col = cols[i % 4]
                colors = SHIFT_COLORS_MAP.get(shift_code, {'bg': '#ffffff', 'text': '#000000'})
                bold = 'font-weight: bold;' if shift_code in ['MP', 'J'] else ''
                col.markdown(
                    f"<div style='background-color: {colors['bg']}; padding: 10px; "
                    f"color: {colors['text']}; border-radius: 5px; text-align: center; {bold}'>"
                    f"<strong>{shift_label}</strong><br/>({shift_code})</div>",
                    unsafe_allow_html=True
                )
        
        # ============================================================
        # KPI SECTION
        # ============================================================
        st.subheader("📊 KPI per Dipendente")
        
        kpi_table = build_kpi_table(schedule_df, kpi_df)
        
        if not kpi_table.empty:
            # Display KPI table
            st.dataframe(kpi_table, use_container_width=True, hide_index=True)
            
            # KPI export
            col1, col2 = st.columns([10, 2])
            with col2:
                csv_kpi = kpi_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 CSV",
                    data=csv_kpi,
                    file_name=f"kpi_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Nessun KPI disponibile")
        
        # ============================================================
        # PREFERENCES SECTION
        # ============================================================
        if pref_df is not None and not pref_df.empty:
            st.subheader("❤️ Preferenze Dipendenti")
            st.dataframe(pref_df, use_container_width=True, hide_index=True)
        
        # ============================================================
        # SHIFT STATISTICS
        # ============================================================
        st.subheader("📈 Statistiche Turni")
        
        shift_counts = aggregate_shift_counts(schedule_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribuzione per dipendente**")
            shift_stat_rows = []
            for emp, counts in shift_counts.items():
                row = {'Dipendente': emp}
                row.update(counts)
                shift_stat_rows.append(row)
            
            shift_stat_df = pd.DataFrame(shift_stat_rows)
            st.dataframe(shift_stat_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Distribuzione globale**")
            total_by_shift = {}
            for shift_type in SHIFT_LABELS.keys():
                count = sum(
                    schedule_df.values.flatten().tolist().count(shift_type)
                )
                if count > 0:
                    total_by_shift[shift_type] = count
            
            if total_by_shift:
                shift_total_df = pd.DataFrame(
                    list(total_by_shift.items()),
                    columns=['Turno', 'Totale']
                )
                shift_total_df['Label'] = shift_total_df['Turno'].map(SHIFT_LABELS)
                st.dataframe(
                    shift_total_df[['Label', 'Totale']],
                    use_container_width=True,
                    hide_index=True
                )
        
        # ============================================================
        # EXPORT SECTION
        # ============================================================
        st.subheader("💾 Esporta Dati")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_schedule = schedule_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Schedule (CSV)",
                data=csv_schedule,
                file_name=f"schedule_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
        
        with col2:
            if not kpi_table.empty:
                csv_kpi = kpi_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 KPI (CSV)",
                    data=csv_kpi,
                    file_name=f"kpi_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if pref_df is not None and not pref_df.empty:
                csv_pref = pref_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Preferenze (CSV)",
                    data=csv_pref,
                    file_name=f"preferences_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Errore nel caricamento: {str(e)}")
        st.info("Assicurati che il file Excel contenga i fogli: 'Planner', 'kpi', 'preferenze'")


if __name__ == "__main__":
    main()
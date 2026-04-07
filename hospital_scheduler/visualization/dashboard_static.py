"""
Hospital Scheduler Dashboard - Simple view mode.
Loads schedule from Excel file and displays calendar with KPIs.
No schedule generation or prediction - just visualization.
"""

from __future__ import annotations

import io
import re
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

st.set_page_config(page_title="Cruscotto Pianificazione Ospedaliera", layout="wide")

# Color mapping for shifts (original colors from previous dashboard)
SHIFT_COLORS_MAP = {
    'M': {'bg': '#dbeafe', 'text': '#1e3a8a'},       # Light blue bg, dark blue text
    'P': {'bg': '#fef3c7', 'text': '#92400e'},       # Light yellow bg, brown text
    'N': {'bg': '#1a237e', 'text': '#ffffff'},       # Dark blue bg, white text
    'NS': {'bg': '#1a237e', 'text': '#ffffff'},      # Night + Smonto (same as N)
    'R': {'bg': '#dcfce7', 'text': '#166534'},       # Light green bg, dark green text
    'A': {'bg': '#fee2e2', 'text': '#991b1b'},       # Light red bg, dark red text
    'MP': {'bg': '#fde68a', 'text': '#7c2d12'},      # Yellow bg, brown text
    'AP': {'bg': '#f3e8ff', 'text': '#6b21a8'},      # Light purple bg, dark purple text
}

SHIFT_LABELS = {
    'M': 'Mattina',
    'P': 'Pomeriggio',
    'NS': 'Notte+Smonto',
    'MP': 'Mattina-Pomeriggio',
    'R': 'Riposo',
    'A': 'Assenze',
    'AP': 'Aggiornamento Professionale',

}

# Default font-size settings (in pixels) - can be adjusted via sidebar sliders
METRIC_LABEL_DEFAULT_PX = 20  # default metric label font in px
METRIC_VALUE_DEFAULT_PX = 42  # ~2.6rem
LEGEND_FONT_DEFAULT_PX = 16   # ~1rem
PLANNER_FONT_DEFAULT_PX = 20  # default planner cell font in px
# runtime-adjustable planner font (will be set from sidebar slider in main)
PLANNER_FONT_SIZE = f"{PLANNER_FONT_DEFAULT_PX}px"

# ============================================================
# Utility Functions
# ============================================================

def create_calendar_table(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a calendar table for display with shift information.
    Returns a formatted DataFrame suitable for HTML/table display.
    """
    # Already in wide format: employees Ã— dates
    return schedule_df.copy()


def style_calendar_cell(val: str) -> str:
    """
    Apply styling to calendar cells based on shift type (original colors).
    """
    # Normalize various representations of missing values to be considered empty
    try:
        sval = '' if pd.isna(val) else str(val).strip()
    except Exception:
        sval = str(val).strip()

    if sval == '' or sval.lower() in ('none', 'nan'):
        # render truly empty cells as white/blank
        return f'background-color: #ffffff; color: #000; text-align: center; font-size: {PLANNER_FONT_SIZE};'

    shift_key = sval
    if shift_key in SHIFT_COLORS_MAP:
        colors = SHIFT_COLORS_MAP[shift_key]
        bold = 'font-weight: bold;' if shift_key in ['MP', 'J'] else ''
        return f"background-color: {colors['bg']}; color: {colors['text']}; text-align: center; font-size: {PLANNER_FONT_SIZE}; {bold}"
    
    return f'background-color: #f0f0f0; color: #999; text-align: center; font-size: {PLANNER_FONT_SIZE};'


def aggregate_shift_counts(schedule_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Count occurrences of each shift type per employee.
    Returns dict: {employee: {shift_type: count}}
    """
    counts_by_employee: Dict[str, Dict[str, int]] = {}
    for employee in schedule_df.index:
        shifts = schedule_df.loc[employee].dropna()
        shifts = shifts[shifts != '']
        # Treat 'NS' as equivalent to 'N' for counting purposes
        shifts_normalized = shifts.replace({'NS': 'N'})
        counts = shifts_normalized.value_counts().to_dict()
        counts_by_employee[employee] = counts

    return counts_by_employee


def build_kpi_table(schedule_df: pd.DataFrame, kpi_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a KPI table for display. If `kpi_df` is provided and non-empty, return a copy of it.
    Otherwise compute basic KPIs from the `schedule_df` (ORE LAVORATE, NOTTI, etc.).
    """
    if kpi_df is not None and not kpi_df.empty:
        return kpi_df.copy()

    kpis = []
    for employee in schedule_df.index:
        shifts = schedule_df.loc[employee].dropna()
        shifts = shifts[shifts != '']
        # Normalize NS -> N for counting
        shifts_normalized = shifts.replace({'NS': 'N'})
        counts = shifts_normalized.value_counts().to_dict()

        m_count = counts.get('M', 0)
        p_count = counts.get('P', 0)
        n_count = counts.get('N', 0)
        mp_count = counts.get('MP', 0)
        r_count = counts.get('R', 0)
        a_count = counts.get('A', 0)
        ap_count = counts.get('AP', 0)
        assenze_count = a_count + ap_count

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
            'ASSENZE': assenze_count,
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


def _norm_col_name(name: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    alias_map = {_norm_col_name(a): a for a in aliases}
    for col in df.columns:
        if _norm_col_name(col) in alias_map:
            return col
    return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.', regex=False).str.strip(),
        errors='coerce'
    ).fillna(0)


# ============================================================
# Main Dashboard
# ============================================================

def main():
    st.title("Cruscotto Pianificazione Ospedaliera")
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
        st.info("👉 Carica un file Excel per iniziare")
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

        # Clean up possible empty rows produced by the Excel file
        # - remove rows with empty/blank index labels
        # - replace empty strings with NA and drop rows that are entirely empty
        try:
            schedule_df = schedule_df.copy()
            # normalize index to stripped strings
            idx = schedule_df.index.to_series().astype(str).str.strip()
            schedule_df.index = idx
            # drop rows with empty index name
            schedule_df = schedule_df.loc[schedule_df.index != '']
            # replace empty strings with NA and drop rows that are all NA
            schedule_df = schedule_df.replace('', pd.NA).dropna(how='all')
            # Keep original codes (e.g., 'NS') for display; we'll treat 'NS' as equivalent to 'N' in counts
        except Exception:
            # if cleanup fails, continue with original dataframe
            pass
        
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
        st.sidebar.success(f"✔️ Caricato: {uploaded_file.name}")
        st.sidebar.info(f"👥 Dipendenti: {len(schedule_df)-1}")
        date_cols = extract_date_columns(schedule_df)
        if date_cols:
            st.sidebar.info(f"📅 Giorni: {len(date_cols)}")
        
        # ============================================================
        # Main Content
        # ============================================================
        # Sidebar sliders to allow live adjustment of metric/legend font sizes
        label_px = st.sidebar.slider("Metric label font (px)", 12, 96, METRIC_LABEL_DEFAULT_PX)
        value_px = st.sidebar.slider("Metric value font (px)", 12, 96, METRIC_VALUE_DEFAULT_PX)
        legend_px = st.sidebar.slider("Legenda font (px)", 10, 36, LEGEND_FONT_DEFAULT_PX)

        # Planner cell font size (affects shift labels like 'M', 'P', 'NS' in the main planner)
        planner_px = st.sidebar.slider("Planner cell font (px)", 8, 48, PLANNER_FONT_DEFAULT_PX)
        # update global variable used by style_calendar_cell
        global PLANNER_FONT_SIZE
        PLANNER_FONT_SIZE = f"{planner_px}px"

        label_size = f"{label_px}px"
        value_size = f"{value_px}px"
        legend_size = f"{legend_px}px"

        st.markdown(f"""
            <style>
            /* Metric labels (small text above value) - target multiple possible DOM structures */
            div[data-testid="stMetricLabel"] p,
            div[data-testid="stMetricLabel"] span,
            div[data-testid="stMetricLabel"] > div,
            .stMetricLabel p,
            .stMetricLabel span,
            .stMetricLabel span {{
                font-size: {label_size} !important;
                font-weight: 600 !important;
                text-align: center !important;
                display: block !important;
                margin: 0 0 4px 0 !important;
            }}
            /* Metric values (large numbers) - target multiple selectors */
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] span,
            div[data-testid="stMetricValue"] > div,
            .stMetricValue,
            .stMetricValue span {{
                font-size: {value_size} !important;
                font-weight: 700 !important;
                text-align: center !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Summary metrics
        col1, col2 = st.columns(2)
        # Summary metrics custom richiesti
        kpi_table = build_kpi_table(schedule_df, kpi_df)
        num_dipendenti = len(schedule_df)
        ore_col = find_column(kpi_table, ['ORE LAVORATE', 'ORELAVORATE'])
        notti_col = find_column(kpi_table, ['NOTTI', 'NOTTE', 'NIGHTS'])
        viol_col = find_column(kpi_table, ['Violazione 11h'])
        riposo_col = find_column(kpi_table, ['Riposo sett.', 'Riposo sett', 'Riposo settimanale'])

        # Build a computed fallback KPI table from the planner so we can safely
        # read missing columns without raising KeyError. Do not overwrite the
        # original `kpi_table` (keep displayed table intact).
        fallback_kpi_table = build_kpi_table(schedule_df, None)

        def _read_numeric_column(col_name, primary_df, fallback_df):
            if col_name and col_name in primary_df.columns:
                return to_numeric_series(primary_df[col_name])
            # try fallback
            col_fb = find_column(fallback_df, [col_name]) if isinstance(col_name, str) else None
            if col_fb and col_fb in fallback_df.columns:
                return to_numeric_series(fallback_df[col_fb])
            return pd.Series([], dtype=float)

        # ore_totali: sum from primary kpi_table if present, else fallback
        ore_series = None
        if ore_col and ore_col in kpi_table.columns:
            ore_series = to_numeric_series(kpi_table[ore_col])
        else:
            ore_fb_col = find_column(fallback_kpi_table, ['ORE LAVORATE', 'ORELAVORATE'])
            ore_series = to_numeric_series(fallback_kpi_table[ore_fb_col]) if ore_fb_col else pd.Series([], dtype=float)
        ore_totali = int(ore_series.sum()) if not ore_series.empty else 0

        # notti_totali
        if notti_col and notti_col in kpi_table.columns:
            notti_series = to_numeric_series(kpi_table[notti_col])
        else:
            notti_fb_col = find_column(fallback_kpi_table, ['NOTTI', 'NOTTE', 'NIGHTS'])
            notti_series = to_numeric_series(fallback_kpi_table[notti_fb_col]) if notti_fb_col else pd.Series([], dtype=float)
        notti_totali = int(notti_series.sum()) if notti_series is not None and not notti_series.empty else 0
        # Jolly / Notte: conta quante volte il valore 'N' o 'NS' appare nel planner (support legacy NS)
        try:
            jolly_count = int(schedule_df.iloc[-1,:].isin(['N', 'NS']).sum().sum())
        except Exception:
            jolly_count = int((schedule_df == 'N').sum().sum())
        # Violaz. 11h: conta quante volte nei kpi c'è un valore diverso da 0 nella colonna 'Violaz. 11h'
        violaz_11h_count = 0
        # st.write(f"DEBUG: kpi_table columns: {kpi_table.columns.tolist()}")
        # st.write(kpi_table.head())  # Debug: show schedule dataframe
        if viol_col and viol_col in kpi_table.columns:
            series = kpi_table[viol_col]
            # st.write(f"DEBUG: series for {viol_col}: {series}")
            cnt = 0
            for v in series:
                # ignore empty-like values
                if pd.isna(v):
                    continue
                sv = str(v).strip()
                if sv == '' or sv.lower() == 'nan':
                    continue
                # try to parse numeric (handle commas)
                try:
                    num = float(str(sv).replace(',', '.'))
                    if abs(num) > 1e-9:
                        cnt += 1
                except Exception:
                    # non-numeric but non-empty -> count as a violation
                    cnt += 1
            violaz_11h_count = int(cnt)
        # Riposo sett.: warning = quante volte valore diverso da 0 nella colonna 'Riposo sett.'
        riposo_sett_warning = int((to_numeric_series(kpi_table[riposo_col]) != 0).sum()) if riposo_col else 0

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        def render_metric(column, label, value, label_size, value_size):
            # Simple HTML-based metric so we can control label/value fonts reliably
            html = (
                f"<div style='text-align:center;padding:6px 4px'>"
                f"<div style='font-size:{label_size};font-weight:600;color:var(--text-color,#111);margin-bottom:6px'>{label}</div>"
                f"<div style='font-size:{value_size};font-weight:700;color:var(--text-color,#111)'>" +
                f"{value}</div></div>"
            )
            column.markdown(html, unsafe_allow_html=True)

        render_metric(col1, "Dipendenti", num_dipendenti-1, label_size, value_size)
        render_metric(col2, "Ore totali pianificate", ore_totali, label_size, value_size)
        render_metric(col3, "N/S assegnati", notti_totali, label_size, value_size)
        render_metric(col4, "Notte Dipartimentale", jolly_count, label_size, value_size)
        render_metric(col5, "Violazioni 11h", violaz_11h_count, label_size, value_size)
        #col6.metric("Warning Riposo sett.", riposo_sett_warning)
        
        # Count total working days
        schedule_long = reshape_schedule_to_long(schedule_df)
        total_shifts = len(schedule_long[~schedule_long['Turno'].isin(['R', 'A', 'AP', ''])])
        


        # ============================================================
        # CALENDAR SECTION
        # ============================================================
        st.subheader("Calendario Assegnazioni")
        
        # Create styled calendar DataFrame
        calendar_display = schedule_df.copy()
        # Normalize missing-like values so the planner shows blank cells instead of 'None'/'nan'
        calendar_display = calendar_display.replace({None: '', 'None': '', 'nan': ''}).fillna('')
        calendar_display.index = [
            'Notte Dip.' if 'jolly medico' in str(idx).strip().lower() else idx
            for idx in calendar_display.index
        ]
        
        # Style the calendar (cells based on shift type)
        styled_calendar = (
            calendar_display.style
            .map(style_calendar_cell)
            # Make index (resource names) bold
            .applymap(lambda x: 'font-weight: bold;', subset=pd.IndexSlice[:, calendar_display.columns[0]:calendar_display.columns[0]])
        )
        
        # Apply bold to all headers via CSS
        styled_calendar = styled_calendar.set_table_styles([
            {'selector': 'th', 'props': [('font-weight', 'bold !important')]},
        ], overwrite=False)
        
        # Compute number of rows that actually contain data (exclude fully-empty rows)
        try:
            displayed_rows = len(calendar_display.dropna(how='all'))
        except Exception:
            displayed_rows = len(calendar_display)
        displayed_rows = max(1, displayed_rows)
        # Use a slightly smaller per-row height and lower minimum to reduce extra space
        planner_height = min(900, max(180, 80 + 32 * displayed_rows))
        st.dataframe(styled_calendar, use_container_width=True, height=planner_height)
        
        # Shift legend shown as simple text (larger font, match subheader)
        st.subheader("Legenda calendario:")
        labels = []
        for shift_code, shift_label in SHIFT_LABELS.items():
            colors = SHIFT_COLORS_MAP.get(shift_code, {'bg': '#ffffff', 'text': '#000000'})
            bold = 'font-weight: bold;' if shift_code in ['MP', 'J'] else ''
            labels.append(
                f"<span style='background-color: {colors['bg']}; color: {colors['text']}; padding: 8px 10px; "
                f"border-radius: 6px; display: inline-block; margin-right: 8px; margin-bottom: 4px; "
                f"font-size: {legend_size}; vertical-align: middle; {bold}'>{shift_label} ({shift_code})</span>"
            )
        # Render all labels in one line/block so spacing is controlled by CSS above
        st.markdown(f"<div style='line-height: 1.6;'>{''.join(labels)}</div>", unsafe_allow_html=True)
        # Styled disclaimer: orange box with alert emoji and extra spacing
        disclaimer_px = legend_px + 2 if 'legend_px' in locals() else (LEGEND_FONT_DEFAULT_PX + 2)
        box_style = (
            "background-color: #fff4e6; "
            "border: 1px solid #ffb547; "
            "padding: 12px 14px; "
            "border-radius: 6px; "
            "margin-top: 18px; margin-bottom: 14px;"
        )
        emoji_style = f"font-size: {disclaimer_px}px; line-height:1;"
        text_style = f"color: #0b1226; font-size: {disclaimer_px}px; font-weight: 600; line-height:1.2;"
        st.markdown(
            f"<div style='{box_style}'><div style='display:flex;align-items:flex-start;gap:10px'>"
            f"<div style='{emoji_style}'>⚠️</div>"
            f"<div style='{text_style}'>Disclaimer assenze (voce A): Nella soluzione proposta con RL, per ragioni di complessità, si è scelto di rappresentare con A l’assenza totale del medico o dell'infermiere in uno specifico slot temporale. Per assenza si intendono tutte le situazioni in cui il personale non è disponibile: ferie, congedi parentali e altri permessi.</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Single CSV export button for planner only
        csv_schedule = calendar_display.to_csv().encode('utf-8')
        st.download_button(
            label="Scarica planner (CSV)",
            data=csv_schedule,
            file_name=f"planner_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
        
        # ============================================================
        # KPI SECTION
        # ============================================================
        st.subheader("KPI per Dipendente")
        
        kpi_table = build_kpi_table(schedule_df, kpi_df)
        
        if not kpi_table.empty:
            # Render KPI table as styled HTML
            kpi_html = kpi_table.to_html(escape=False, index=False)
            # Wrap with style tags
            kpi_html_display = f"""
<style>
.kpi-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    font-family: sans-serif;
}}
.kpi-table th {{
    background-color: #1e3a8a;
    color: white;
    font-weight: bold;
    padding: 10px;
    text-align: left;
    border: 1px solid #ddd;
}}
.kpi-table td {{
    padding: 8px 10px;
    border: 1px solid #ddd;
}}
.kpi-table tbody tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
</style>
{kpi_html.replace('<table', '<table class="kpi-table"')}
            """
            st.markdown(kpi_html_display, unsafe_allow_html=True)
        else:
            st.warning("Nessun KPI disponibile")
        
        # ============================================================
        # PREFERENCES SECTION
        # ============================================================
        if pref_df is not None and not pref_df.empty:
            st.subheader("Preferenze Dipendenti")
            # Render Preferences table as styled HTML
            pref_html = pref_df.to_html(escape=False, index=False)
            # Wrap with style tags
            pref_html_display = f"""
<style>
.pref-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    font-family: sans-serif;
}}
.pref-table th {{
    background-color: #1e3a8a;
    color: white;
    font-weight: bold;
    padding: 10px;
    text-align: left;
    border: 1px solid #ddd;
}}
.pref-table td {{
    padding: 8px 10px;
    border: 1px solid #ddd;
}}
.pref-table tbody tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
</style>
{pref_html.replace('<table', '<table class="pref-table"')}
            """
            st.markdown(pref_html_display, unsafe_allow_html=True)
        
        # ============================================================
        # SHIFT STATISTICS
        # ============================================================
        # st.subheader("Statistiche Turni")
        
        # shift_counts = aggregate_shift_counts(schedule_df)
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.write("**Distribuzione per dipendente**")
        #     shift_stat_rows = []
        #     for emp, counts in shift_counts.items():
        #         row = {'Dipendente': emp}
        #         row.update(counts)
        #         shift_stat_rows.append(row)
            
        #     shift_stat_df = pd.DataFrame(shift_stat_rows)
        #     st.dataframe(shift_stat_df, use_container_width=True, hide_index=True)
        
        # with col2:
        #     st.write("**Distribuzione globale**")
        #     total_by_shift = {}
        #     for shift_type in SHIFT_LABELS.keys():
        #         count = sum(
        #             schedule_df.values.flatten().tolist().count(shift_type)
        #         )
        #         if count > 0:
        #             total_by_shift[shift_type] = count
            
        #     if total_by_shift:
        #         shift_total_df = pd.DataFrame(
        #             list(total_by_shift.items()),
        #             columns=['Turno', 'Totale']
        #         )
        #         shift_total_df['Label'] = shift_total_df['Turno'].map(SHIFT_LABELS)
        #         st.dataframe(
        #             shift_total_df[['Label', 'Totale']],
        #             use_container_width=True,
        #             hide_index=True
        #         )

        # R in A è classe assenza che racchiude....
    except Exception as e:
        st.error(f"❌ Errore nel caricamento: {str(e)}")
        st.info("Assicurati che il file Excel contenga i fogli: 'Planner', 'kpi', 'preferenze'")


if __name__ == "__main__":
    main()

from __future__ import annotations
 
import calendar as _cal
from io import BytesIO
from pathlib import Path
 
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
 
from hospital_scheduler.config.settings import (
    EMPLOYEES, CONTRACT_TYPES, SHIFT_HOURS, SHIFT_NAME_TO_ID,
    SHIFT_TIME_WINDOWS, MAX_MONTHLY_OVERTIME_PROXY, PREFERENCE_TYPES, MESI_IT,
)
from hospital_scheduler.config.holidays import get_holidays_for_month
from hospital_scheduler.training.ferie import (
    FERIE_LEVEL_1, FERIE_LEVEL_2, FERIE_LEVEL_3, FERIE_LEVEL_4, FERIE_LEVEL_5,
    generate_annual_ferie, get_ferie_for_month,
)
from hospital_scheduler.evaluation.metrics import DEFAULT_WEIGHTS
from hospital_scheduler.utils.io import get_latest_checkpoint
from hospital_scheduler.visualization.reports import run_debug, run_debug_month
 
st.set_page_config(page_title="Hospital RL Dashboard", layout="wide")
 
EMPLOYEE_META = {e["name"]: e for e in EMPLOYEES}
CONTRACT_BY_NAME = {e["name"]: CONTRACT_TYPES[e["contract"]] for e in EMPLOYEES}
 
 
# ============================================================
# Data loading helpers
# ============================================================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
 
 
def shift_hours_by_name(shift_name: str) -> int:
    sid = SHIFT_NAME_TO_ID.get(str(shift_name))
    return int(SHIFT_HOURS.get(sid, 0)) if sid is not None else 0
 
 
def rest_hours_between(prev_shift_name: str, next_shift_name: str) -> float:
    prev_id = SHIFT_NAME_TO_ID.get(str(prev_shift_name), 3)
    next_id = SHIFT_NAME_TO_ID.get(str(next_shift_name), 3)
    prev_window = SHIFT_TIME_WINDOWS.get(prev_id)
    next_window = SHIFT_TIME_WINDOWS.get(next_id)
    if prev_window is None or next_window is None:
        return 24.0
    return float((next_window[0] + 24) - prev_window[1])
 
 
# ============================================================
# DataFrame enrichment
# ============================================================
def ensure_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "giorno" in out.columns:
        out["giorno"] = pd.to_numeric(out["giorno"], errors="coerce")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    if "date_label" not in out.columns and "date" in out.columns:
        out["date_label"] = out["date"].dt.strftime("%a %d/%m/%Y")
    if "weekday_label" not in out.columns and "date" in out.columns:
        out["weekday_label"] = out["date"].dt.strftime("%a")
    return out
 
 
def enrich_debug(debug_df: pd.DataFrame) -> pd.DataFrame:
    if debug_df.empty:
        return debug_df
    df = ensure_calendar_columns(debug_df)
    sort_cols = [c for c in ["dipendente", "giorno"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df["ore_turno"] = df["turno_finale"].map(shift_hours_by_name)
    df["settimana"] = ((df["giorno"] - 1) // 7 + 1) if "giorno" in df.columns else 1
    df["turno_precedente"] = df.groupby("dipendente")["turno_finale"].shift(1)
    df["ore_riposo"] = df.apply(
        lambda r: rest_hours_between(r["turno_precedente"], r["turno_finale"])
        if pd.notna(r["turno_precedente"]) else 24.0,
        axis=1,
    )
    df["violazione_riposo_11h"] = df["ore_riposo"] < 11.0
    df["is_rest_like"] = df["turno_finale"].isin(["R", "AP"])
    if "is_special_day" in df.columns:
        df["giorno_speciale_lavorato"] = df["is_special_day"].astype(bool) & (~df["is_rest_like"])
    else:
        df["giorno_speciale_lavorato"] = False
    return df
 
 
# ============================================================
# KPI aggregation
# ============================================================
def aggregate_employee_kpis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for name, g in df.groupby("dipendente"):
        if name not in CONTRACT_BY_NAME:
            continue
        contract = CONTRACT_BY_NAME[name]
        weekly_target = contract["weekly_hours"]
        weekly_limit = contract["weekly_hours"] + contract["max_weekly_overtime"]
        weekly_hours = g.groupby("settimana")["ore_turno"].sum()
        total_hours = float(g["ore_turno"].sum())
        # Standard monthly hours: Dirigenza 152h (38*4), Infermieri 144h (36*4)
        standard_monthly_hours = weekly_target * 4
        overtime_proxy = max(0.0, total_hours - standard_monthly_hours)
        rows.append({
            "dipendente": name,
            "ruolo": g["ruolo"].iloc[0] if "ruolo" in g.columns else "",
            "contratto": contract["name"],
            "ore_totali": total_hours,
            "ore_straordinario": overtime_proxy,
            "notti": int((g["turno_finale"] == "N").sum()),
            "mattine": int((g["turno_finale"] == "M").sum()),
            "pomeriggi": int((g["turno_finale"] == "P").sum()),
            "riposi": int((g["turno_finale"] == "R").sum()),
            "mp": int((g["turno_finale"] == "MP").sum()),
            "giorni_speciali_lavorati": int(g["giorno_speciale_lavorato"].sum()),
            "violazioni_riposo_11h": int(g["violazione_riposo_11h"].sum()),
            "settimane_oltre_limite": int((weekly_hours > weekly_limit).sum()),
            "settimane_senza_riposo": int(
                (~g.groupby("settimana").apply(lambda x: (x["turno_finale"] == "R").any())).sum()
            ),
        })
    return pd.DataFrame(rows).sort_values(["ruolo", "dipendente"]).reset_index(drop=True)
 
 
def build_role_kpis(emp_kpis: pd.DataFrame) -> pd.DataFrame:
    if emp_kpis.empty:
        return pd.DataFrame()
    return (
        emp_kpis.groupby("ruolo")
        .agg(
            dipendenti=("dipendente", "count"),
            ore_totali=("ore_totali", "sum"),
            ore_straordinario=("ore_straordinario", "sum"),
            notti=("notti", "sum"),
            giorni_speciali_lavorati=("giorni_speciali_lavorati", "sum"),
            violazioni_riposo_11h=("violazioni_riposo_11h", "sum"),
            settimane_oltre_limite=("settimane_oltre_limite", "sum"),
            settimane_senza_riposo=("settimane_senza_riposo", "sum"),
        )
        .reset_index()
    )
 
 
def build_planner(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = ensure_calendar_columns(df)
    if "giorno" in tmp.columns and "date_label" in tmp.columns:
        ordered_labels = (
            tmp[["giorno", "date_label"]].drop_duplicates().sort_values("giorno")["date_label"].tolist()
        )
    else:
        ordered_labels = tmp["date_label"].drop_duplicates().tolist() if "date_label" in tmp.columns else []
 
    pivot = tmp.pivot(index="dipendente", columns="date_label", values="turno_finale")
    real_employees = [e["name"] for e in EMPLOYEES]
    pivot = pivot.reindex(index=[n for n in real_employees if n in pivot.index])
    if ordered_labels:
        pivot = pivot.reindex(columns=ordered_labels)
 
    jolly_rows = tmp[tmp["dipendente"] == "Jolly Medico"]
    jolly_row_data = {
        label: (jolly_rows[jolly_rows["date_label"] == label]["turno_finale"].iloc[0]
                if not jolly_rows[jolly_rows["date_label"] == label].empty else None)
        for label in ordered_labels
    }
    jolly_series = pd.Series(jolly_row_data, name="Jolly Medico")
    pivot = pd.concat([pivot, jolly_series.to_frame().T])
    return pivot
 
 
def compute_warning_tables(emp_kpis: pd.DataFrame, df: pd.DataFrame):
    warnings = []
    for _, r in emp_kpis.iterrows():
        if r["dipendente"] not in CONTRACT_BY_NAME:
            continue
        contract = CONTRACT_BY_NAME[r["dipendente"]]
        weekly_limit = contract["weekly_hours"] + contract["max_weekly_overtime"]
        if r["violazioni_riposo_11h"] > 0:
            warnings.append({"Severità": "Alta", "Dipendente": r["dipendente"],
                             "Regola": "Riposo minimo 11h", "Dettaglio": f"{int(r['violazioni_riposo_11h'])} violazioni"})
        if r["settimane_oltre_limite"] > 0:
            warnings.append({"Severità": "Alta", "Dipendente": r["dipendente"],
                             "Regola": "Limite settimanale",
                             "Dettaglio": f"{int(r['settimane_oltre_limite'])} settimane oltre {weekly_limit}h"})
        if r["settimane_senza_riposo"] > 0:
            warnings.append({"Severità": "Alta", "Dipendente": r["dipendente"],
                             "Regola": "Riposo settimanale",
                             "Dettaglio": f"{int(r['settimane_senza_riposo'])} settimane senza R"})
        if r["ore_straordinario"] > MAX_MONTHLY_OVERTIME_PROXY:
            warnings.append({"Severità": "Alta", "Dipendente": r["dipendente"],
                             "Regola": "Straordinario proxy",
                             "Dettaglio": f"{r['ore_straordinario']:.1f}h > {MAX_MONTHLY_OVERTIME_PROXY}h"})
 
    if df.empty:
        return pd.DataFrame(warnings), pd.DataFrame()
 
    group_cols = [c for c in ["giorno", "date_label", "weekday_label"] if c in df.columns]
    day_warn = df.groupby(group_cols, as_index=False).agg(
        violazioni_riposo_11h=("violazione_riposo_11h", "sum"),
        lavoratori_giorno_speciale=("giorno_speciale_lavorato", "sum"),
    )
    if "giorno" in day_warn.columns:
        day_warn = day_warn.sort_values("giorno").reset_index(drop=True)
 
    return pd.DataFrame(warnings), day_warn
 
 
def aggregate_preferences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'preference_type' not in df.columns:
        return pd.DataFrame()
    pref_df = df[df['preference_type'].notna()].copy()
    if pref_df.empty:
        return pd.DataFrame()
    rows = []
    for name, g in pref_df.groupby('dipendente'):
        ruolo = g['ruolo'].iloc[0] if 'ruolo' in g.columns else 'N/A'
        num_pref = len(g)
        rispettate = (g['preference_rispettata'] == True).sum()
        non_rispettate = (g['preference_rispettata'] == False).sum()
        percentuale = (rispettate / num_pref * 100) if num_pref > 0 else 0.0
        rows.append({
            'dipendente': name, 'ruolo': ruolo,
            'num_preferenze': num_pref,
            'rispettate': int(rispettate), 'non_rispettate': int(non_rispettate),
            '% rispetto': f"{percentuale:.1f}%",
            'tipi': ', '.join(sorted(g['preference_type'].unique())),
        })
    return pd.DataFrame(rows).sort_values(['ruolo', 'dipendente']).reset_index(drop=True)
 
 
def style_planner(df: pd.DataFrame, debug_df: pd.DataFrame = None, ferie_table: dict = None):
    color_map = {
        "M": "background-color: #dbeafe; color: #1e3a8a;",
        "P": "background-color: #fef3c7; color: #92400e;",
        "N": "background-color: #1a237e; color: #ffffff;",
        "R": "background-color: #dcfce7; color: #166534;",
        "MP": "background-color: #fde68a; color: #7c2d12; font-weight: bold;",
        "AP": "background-color: #f3e8ff; color: #6b21a8;",
        "J": "background-color: #fee2e2; color: #991b1b; font-weight: bold;",
    }
    date_to_day: dict = {}
    if debug_df is not None and not debug_df.empty and "date_label" in debug_df.columns and "giorno" in debug_df.columns:
        for _, row in debug_df[["giorno", "date_label"]].drop_duplicates().iterrows():
            date_to_day[row["date_label"]] = int(row["giorno"]) - 1
    emp_name_to_id = {e["name"]: e["id"] for e in EMPLOYEES}
 
    def cell_style(val, row_label, col_label):
        if ferie_table and row_label in emp_name_to_id and col_label in date_to_day:
            emp_id = emp_name_to_id[row_label]
            day = date_to_day[col_label]
            if (emp_id, day) in ferie_table:
                return "background-color: #fca5a5; color: #991b1b; font-weight: bold; text-decoration: underline;"
        return color_map.get(str(val), "")
 
    def apply_styling(s):
        col_label = s.name
        return s.index.map(lambda row_label: cell_style(s[row_label], row_label, col_label))
 
    return df.style.apply(apply_styling, axis=0)
 
 
def export_planner_to_excel(planner_df: pd.DataFrame, debug_df: pd.DataFrame = None, ferie_table: dict = None) -> BytesIO:
    """
    Export planner to Excel with conditional formatting (colors based on shift type).
    """
    # Color definitions for shifts
    shift_colors = {
        "M": "DBEAFE",      # Light blue
        "P": "FEF3C7",      # Light yellow
        "N": "1a237e",      # Dark blue
        "R": "DCFCE7",      # Light green
        "MP": "FDE68A",     # Orange-yellow
        "AP": "F3E8FF",     # Light purple
        "J": "FEE2E2",      # Light red
    }
   
    shift_text_colors = {
        "M": "1e3a8a",      # Dark blue
        "P": "92400e",      # Dark brown
        "N": "ffffff",      # White
        "R": "166534",      # Dark green
        "MP": "7c2d12",     # Dark brown
        "AP": "6b21a8",     # Dark purple
        "J": "991b1b",      # Dark red
    }
   
    wb = Workbook()
    ws = wb.active
    ws.title = "Planner"
   
    # Write header row (employee names)
    ws.cell(row=1, column=1, value="Dipendente")
    for col_idx, col in enumerate(planner_df.columns, 2):
        ws.cell(row=1, column=col_idx, value=col)
   
    # Build ferie lookup
    date_to_day: dict = {}
    emp_name_to_id = {e["name"]: e["id"] for e in EMPLOYEES}
    if debug_df is not None and not debug_df.empty and "date_label" in debug_df.columns and "giorno" in debug_df.columns:
        for _, row in debug_df[["giorno", "date_label"]].drop_duplicates().iterrows():
            date_to_day[row["date_label"]] = int(row["giorno"]) - 1
   
    # Write data rows with formatting
    for row_idx, (emp_name, row_data) in enumerate(planner_df.iterrows(), 2):
        ws.cell(row=row_idx, column=1, value=emp_name)
       
        for col_idx, (col_label, shift) in enumerate(row_data.items(), 2):
            cell = ws.cell(row=row_idx, column=col_idx, value=shift)
           
            # Check if it's a ferie day
            is_ferie = False
            if ferie_table and emp_name in emp_name_to_id and col_label in date_to_day:
                emp_id = emp_name_to_id[emp_name]
                day = date_to_day[col_label]
                if (emp_id, day) in ferie_table:
                    is_ferie = True
           
            if is_ferie:
                # Ferie: light red background
                cell.fill = PatternFill(start_color="FCA5A5", end_color="FCA5A5", fill_type="solid")
                cell.font = Font(color="991b1b", bold=True)
            elif shift and str(shift) != "nan":
                shift_str = str(shift).strip()
                if shift_str in shift_colors:
                    cell.fill = PatternFill(start_color=shift_colors[shift_str], end_color=shift_colors[shift_str], fill_type="solid")
                    cell.font = Font(color=shift_text_colors[shift_str], bold=(shift_str in ["MP", "J"]))
           
            cell.alignment = Alignment(horizontal="center", vertical="center")
   
    # Adjust column widths
    ws.column_dimensions['A'].width = 15
    for col_idx in range(2, len(planner_df.columns) + 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 10
   
    # Save to BytesIO
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output
 
 
# ============================================================
# Main dashboard
# ============================================================
def main():
    title_placeholder = st.empty()
 
    debug_path = st.sidebar.text_input("CSV debug", "logs/debug_run.csv")
    daily_path = st.sidebar.text_input("CSV KPI giornalieri", "outputs/run_base_daily_log.csv")
 
    st.sidebar.subheader("Pianificazione Annuale")
    sel_year = st.sidebar.number_input("Anno", min_value=2025, max_value=2035, value=2026, step=1)
    sel_month = st.sidebar.selectbox("Mese", list(range(1, 13)), index=3, format_func=lambda m: MESI_IT[m])
    title_placeholder.title(f"Pianificazione {MESI_IT[sel_month]} {sel_year}")
    ferie_seed = st.sidebar.number_input("Seed ferie annuali", min_value=0, max_value=9999, value=42, step=1)
 
    @st.cache_data
    def _cached_annual_ferie(year: int, seed: int):
        return generate_annual_ferie(year, seed=seed)
 
    annual_ferie = _cached_annual_ferie(sel_year, ferie_seed)
    month_ferie = get_ferie_for_month(annual_ferie, sel_year, sel_month)
    month_holidays = get_holidays_for_month(sel_year, sel_month)
    days_in_sel_month = _cal.monthrange(sel_year, sel_month)[1]
 
    st.sidebar.caption(
        f"{MESI_IT[sel_month]} {sel_year}: **{days_in_sel_month}** giorni, "
        f"**{len(month_holidays)}** festivi, **{len(month_ferie)}** ferie"
    )
 
    curriculum_level = st.sidebar.selectbox("Livello curriculum (per evidenziare ferie)", [1, 2, 3, 4, 5])
    ferie_by_level = {1: FERIE_LEVEL_1, 2: FERIE_LEVEL_2, 3: FERIE_LEVEL_3,
                      4: FERIE_LEVEL_4, 5: FERIE_LEVEL_5}
 
    st.sidebar.markdown("---")
    n_episodes = st.sidebar.number_input("Numero episodi", min_value=1, max_value=500, value=100, step=10)
    model_path = st.sidebar.text_input("Model path", "models/ppo_hospital_final")
 
    latest_checkpoint, latest_timesteps = get_latest_checkpoint()
    if latest_checkpoint:
        st.sidebar.info(
            f"✅ **Checkpoint trovato**\n\n"
            f"**File**: `{latest_checkpoint.name}`\n\n"
            f"**Timesteps**: {latest_timesteps:,}\n\n"
            f"Questo sarà caricato al posto di `{model_path}`"
        )
    else:
        st.sidebar.warning(f"⚠️ **Nessun checkpoint trovato**\n\nVerrà caricato: `{model_path}`")
 
    st.sidebar.markdown("---")
    month_output_path = f"logs/debug_run_{sel_year}_{sel_month:02d}.csv"
   
    # Filter toggle
    filter_valid_only = st.sidebar.checkbox(
        "Filtra copertura completa (dirigenza + infermieri)",
        value=True,
        help="Se attivo: seleziona solo calendari con copertura completa su tutte le fasce orarie (M/MP, P/MP, N) per ENTRAMBI i comparti (dirigenza e infermieri) tutti i giorni\n\nSe disattivo: sceglie casualmente tra tutti gli episodi"
    )
 
    if st.sidebar.button(f"Predict {MESI_IT[sel_month]} {sel_year}", type="primary", use_container_width=True):
        with st.spinner(f"Predict {MESI_IT[sel_month]} {sel_year} — {n_episodes} episodi..."):
            run_debug_month(
                year=sel_year, month=sel_month, annual_ferie=annual_ferie,
                model_path=model_path, output_path=month_output_path,
                n_episodes=n_episodes, weights=DEFAULT_WEIGHTS,
                filter_valid_only=filter_valid_only,
            )
        st.cache_data.clear()
        debug_path = month_output_path
        st.rerun()
 
    if st.sidebar.button("Debug Run Legacy (Aprile)", use_container_width=True):
        with st.spinner(f"Eseguendo {n_episodes} episodi..."):
            run_debug(model_path=model_path, output_path=debug_path, n_episodes=n_episodes, weights=DEFAULT_WEIGHTS)
        st.cache_data.clear()
        st.rerun()
 
    month_csv = Path(month_output_path)
    if month_csv.exists():
        debug_path = month_output_path
        current_ferie_table = month_ferie
    else:
        current_ferie_table = ferie_by_level[curriculum_level]
 
    debug_df_raw = load_csv(debug_path)
    daily_df_raw = load_csv(daily_path)
    scores_log_path = str(Path(debug_path).parent / f"{Path(debug_path).stem}_scores_log.csv")
    scores_df = load_csv(scores_log_path)

    if debug_df_raw.empty:
        st.error(f"File debug non trovato o vuoto: {debug_path}")
        st.stop()
 
    debug_df = enrich_debug(debug_df_raw)
    daily_df = ensure_calendar_columns(daily_df_raw) if not daily_df_raw.empty else pd.DataFrame()
 
    # Handle multiple episodes: show selector with quality filters
    selected_episode = None
    if 'episode_id' in debug_df.columns:
        episodes_in_df = sorted(debug_df['episode_id'].unique())
        if len(episodes_in_df) > 1:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Selezione Episodio")

            has_quality = (
                not scores_df.empty
                and 'quality_score' in scores_df.columns
                and 'night_std_medico' in scores_df.columns
            )
            filtered_episodes = list(episodes_in_df)
            ep_quality: dict = {}

            if has_quality:
                st.sidebar.markdown("**Filtri qualità**")
                max_night_std = st.sidebar.slider(
                    "Max std notti (rotazione)",
                    min_value=0.0, max_value=3.0, value=3.0, step=0.1,
                    help="Deviazione standard delle notti per dipendente — più basso = rotazione più equa (3.0 = nessun filtro)",
                )
                max_unj_rest = st.sidebar.slider(
                    "Max riposi ingiustificati",
                    min_value=0, max_value=50, value=50, step=1,
                    help="Turni R oltre il minimo obbligatorio (1/settimana/dipendente)",
                )
                ep_q = (
                    scores_df[scores_df['episode'].isin(episodes_in_df)]
                    .set_index('episode')
                )
                ep_quality = ep_q.to_dict('index')
                filtered_episodes = [
                    e for e in episodes_in_df
                    if (
                        ep_quality.get(e, {}).get('night_std_medico', 0.) <= max_night_std
                        and ep_quality.get(e, {}).get('unjustified_R', 0) <= max_unj_rest
                    )
                ]
                if not filtered_episodes:
                    st.sidebar.warning("Nessun episodio passa i filtri \u2014 mostro tutti")
                    filtered_episodes = list(episodes_in_df)

            def _ep_label(x):
                if x == -1:
                    return "Fallback (nessun valido)"
                q = ep_quality.get(x, {})
                if q:
                    return (
                        f"Ep {int(x)} \u2502 Q:{q.get('quality_score', 0.):.2f}"
                        f" \u2502 N_std:{q.get('night_std_medico', 0.):.1f}/{q.get('night_std_inf', 0.):.1f}"
                        f" \u2502 R+:{int(q.get('unjustified_R', 0))}"
                    )
                return f"Episodio {int(x)}"

            selected_episode = st.sidebar.selectbox(
                f"Episodio ({len(filtered_episodes)} disponibili su {len(episodes_in_df)})",
                options=filtered_episodes,
                format_func=_ep_label,
            )
            debug_df = debug_df[debug_df['episode_id'] == selected_episode].copy()
            if 'episode_id' in debug_df.columns:
                debug_df = debug_df.drop('episode_id', axis=1)
   
    emp_kpis = aggregate_employee_kpis(debug_df)
    role_kpis = build_role_kpis(emp_kpis)
    warnings_df, day_warn_df = compute_warning_tables(emp_kpis, debug_df)
    planner = build_planner(debug_df)
 
 
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Dipendenti", len(emp_kpis))
    c2.metric("Ore totali", f"{emp_kpis['ore_totali'].sum():.0f}" if not emp_kpis.empty else "0")
    c3.metric("Notti", int(emp_kpis["notti"].sum()) if not emp_kpis.empty else 0)
    c4.metric("Violazioni 11h", int(emp_kpis["violazioni_riposo_11h"].sum()) if not emp_kpis.empty else 0)
    c5.metric("Warning", len(warnings_df))
    jolly_days_count = int((debug_df["dipendente"] == "Jolly Medico").sum()) if not debug_df.empty else 0
    c6.metric("Jolly Medico", f"{jolly_days_count}/{days_in_sel_month} gg")
 
    if not daily_df.empty and {"giorno", "date_label", "reward_total"}.issubset(daily_df.columns):
        st.subheader("Reward giornaliero")
        chart = daily_df[["giorno", "date_label", "reward_total"]].drop_duplicates().sort_values("giorno")
        st.line_chart(chart.set_index("date_label")["reward_total"])
 
    st.subheader("Planner mensile")
    col1, col2 = st.columns([10, 2])
    with col1:
        st.dataframe(style_planner(planner, debug_df, current_ferie_table), use_container_width=True)
    with col2:
        excel_file = export_planner_to_excel(planner, debug_df, current_ferie_table)
        st.download_button(
            label="📥 Excel",
            data=excel_file,
            file_name=f"planner_{sel_year}_{sel_month:02d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
 
    st.subheader("KPI per ruolo")
    st.dataframe(role_kpis, use_container_width=True, hide_index=True)
 
    st.subheader("KPI per dipendente")
    st.dataframe(emp_kpis, use_container_width=True, hide_index=True)
 
    st.subheader("Warning contrattuali / di equilibrio")
    if warnings_df.empty:
        st.info("Nessun warning rilevato.")
    else:
        st.dataframe(warnings_df, use_container_width=True, hide_index=True)
 
    st.subheader("Warning per giorno")
    if day_warn_df.empty:
        st.info("Nessun warning giornaliero disponibile.")
    else:
        st.dataframe(day_warn_df, use_container_width=True, hide_index=True)
 
    jolly_df = debug_df[debug_df["dipendente"] == "Jolly Medico"] if not debug_df.empty else pd.DataFrame()
    st.subheader("Jolly Medico (dottore esterno notte)")
    if jolly_df.empty:
        st.success("Nessun Jolly Medico attivato: tutte le notti coperte dai dirigenti interni.")
    else:
        st.warning(f"Jolly Medico attivato in **{len(jolly_df)}** giorni su 28.")
        jolly_detail_cols = [c for c in ["giorno", "date_label", "weekday_label", "turno_finale",
                                          "is_weekend", "is_holiday", "is_special_day"] if c in jolly_df.columns]
        st.dataframe(
            jolly_df[jolly_detail_cols].sort_values("giorno") if "giorno" in jolly_df.columns else jolly_df[jolly_detail_cols],
            use_container_width=True, hide_index=True,
        )
 
    st.subheader("Preferenze Dipendenti")
    prefs_df = aggregate_preferences(debug_df)
    if prefs_df.empty:
        st.info("Nessuna preferenza assegnata in questo episodio.")
    else:
        st.dataframe(prefs_df, use_container_width=True, hide_index=True)
 
    if not emp_kpis.empty:
        selected = st.selectbox("Dettaglio dipendente", options=emp_kpis["dipendente"].tolist())
        person = debug_df[debug_df["dipendente"] == selected].copy()
        detail_cols = [c for c in [
            "giorno", "date_label", "weekday_label", "turno_finale", "ore_turno",
            "giorno_speciale_lavorato", "ore_riposo", "violazione_riposo_11h",
            "preference_type", "preference_rispettata",
            "violazioni", "soft_violations", "hard_overrides",
        ] if c in person.columns]
        if "giorno" in person.columns:
            person = person.sort_values("giorno")
        st.dataframe(person[detail_cols], use_container_width=True, hide_index=True)
 
 
if __name__ == "__main__":
    main()
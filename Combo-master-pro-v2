import difflib
import io
import math
import re
import unicodedata
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as str_lit

# --- CONFIGURAZIONE PAGINA ---
str_lit.set_page_config(
    page_title="COMBO Master Pro Betting & Deep Analytics Suite",
    layout="wide",
    page_icon="⚽",
)

# --- DESIGN MODERNO AVANZATO (CUSTOM CSS & GLASSMORPHISM) ---
str_lit.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header principale
str_lit.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 25px; border-radius: 20px; border: 1px solid #312e81; text-align: center; margin-bottom: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);">
        <h1 style="color: #f8fafc; font-weight: 800; font-size: 2.0rem; margin: 0; letter-spacing: -0.5px;">⚽ COMBO MASTER PRO AI</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px;">Suite predittiva avanzata con Poisson, Monte Carlo e Value Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

LEAGUES = {
    "Serie A (Italia)": ("SA", "football-data"),
    "Premier League (Inghilterra)": ("PL", "football-data"),
    "La Liga (Spagna)": ("PD", "football-data"),
    "Bundesliga (Germania)": ("BL1", "football-data"),
    "Ligue 1 (Francia)": ("FL1", "football-data"),
    "Eredivisie (Olanda)": ("DED", "football-data"),
    "Champions League": ("CL", "football-data"),
}

# --- SELETTORE PRINCIPALE IN EVIDENZA (OTTIMIZZATO ANCHE PER MOBILE) ---
str_lit.markdown("### 🏆 Seleziona Campionato (Mobile & Desktop)")
campionato_scelto = str_lit.selectbox("Scegli il torneo da analizzare", list(LEAGUES.keys()), label_visibility="collapsed")
selezionati_dict = {campionato_scelto: LEAGUES[campionato_scelto]}

str_lit.sidebar.markdown("### ⚙ Configurazione API")
api_key = str_lit.sidebar.text_input("🔑 API Key (football-data.org)", value="d29062443f334e339e7b953ed78a1ac3", type="password")
str_lit.sidebar.markdown("---")
str_lit.sidebar.info("💡 Suggerimento: Da cellulare puoi cambiare campionato direttamente dal menu a tendina in alto nella pagina principale.")

tab_calendario, tab_classifica, tab_value, tab_grafici, tab_ai_schedine, tab_value_finder, tab_monte_carlo = str_lit.tabs([
    "📅 Calendario",
    "🏆 Classifica",
    "🔍 Value Bet",
    "📊 Grafici",
    "🤖 Schedine AI",
    "⚡ Value Finder",
    "🎲 Monte Carlo",
])

# --- FUNZIONI DI SUPPORTO & MODELLAZIONE ---
@str_lit.cache_data(ttl=3600)
def scarica_dati_club(chiave, league_code):
    if not chiave: return None
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": chiave}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

@str_lit.cache_data(ttl=3600)
def scarica_classifica_club(chiave, league_code):
    if not chiave: return None
    url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
    headers = {"X-Auth-Token": chiave}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

# --- QUOTE REALI (football-data.co.uk, file fixtures.csv) ---
# Nota: si usa il dominio SENZA "www" (con "www" il sito blocca il traffico non britannico).
URL_FIXTURES_QUOTE = "https://football-data.co.uk/fixtures.csv"
ODDS_DIV = {"SA": "I1", "PL": "E0", "PD": "SP1", "BL1": "D1", "FL1": "F1", "DED": "N1"}
# Ordine di preferenza delle colonne quote 1X2: media mercato, Bet365, Pinnacle, quota massima
COLONNE_QUOTE = [
    ("AvgH", "AvgD", "AvgA", "media bookmaker"),
    ("B365H", "B365D", "B365A", "Bet365"),
    ("PSH", "PSD", "PSA", "Pinnacle"),
    ("MaxH", "MaxD", "MaxA", "quota massima"),
]

_STOP_NOMI = {"fc", "ac", "as", "cf", "afc", "acf", "ssc", "us", "ud", "rc", "rcd", "sv", "vfl",
              "vfb", "fsv", "tsg", "fk", "cd", "ca", "sc", "bv", "de", "calcio", "ogc", "sco"}
# Chiavi = nome del CSV normalizzato, valori = nome football-data.org normalizzato
_ALIAS_NOMI = {
    "inter": "internazionale milano", "verona": "hellas verona",
    "man united": "manchester united", "man city": "manchester city",
    "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers",
    "ath madrid": "club atletico madrid", "ath bilbao": "athletic club",
    "espanol": "espanyol barcelona", "bayern munich": "bayern munchen",
    "m gladbach": "borussia monchengladbach", "ein frankfurt": "eintracht frankfurt",
    "paris sg": "paris saint germain", "lyon": "olympique lyonnais",
    "rennes": "stade rennais", "brest": "stade brestois", "for sittard": "fortuna sittard",
}

def _norm_sq(nome):
    n = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    tok = [t for t in n.split() if t not in _STOP_NOMI and not t.isdigit()]
    n = " ".join(tok)
    return _ALIAS_NOMI.get(n, n)

def _similarita(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta: return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()

@str_lit.cache_data(ttl=3600)
def scarica_quote_fixtures():
    try:
        res = requests.get(URL_FIXTURES_QUOTE, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code != 200: return None
        df = pd.read_csv(io.BytesIO(res.content), encoding="latin-1", on_bad_lines="skip")
    except Exception:
        return None
    if not {"Div", "HomeTeam", "AwayTeam"}.issubset(df.columns): return None
    return df.dropna(subset=["HomeTeam", "AwayTeam"])

def estrai_quote_riga(riga):
    for h, d, a, fonte in COLONNE_QUOTE:
        try:
            vals = [float(riga[h]), float(riga[d]), float(riga[a])]
        except (KeyError, TypeError, ValueError):
            continue
        if all(v > 1.0 for v in vals):
            return vals, fonte
    return None, None

def costruisci_quote_map(matches_list, odds_df, div):
    """Abbina le partite di football-data.org alle quote reali. Ritorna (mappa id->(quote, fonte), righe_non_abbinate)."""
    if odds_df is None or not div: return {}, []
    righe = []
    for _, r in odds_df[odds_df["Div"] == div].iterrows():
        q, fonte = estrai_quote_riga(r)
        if q: righe.append((_norm_sq(r["HomeTeam"]), _norm_sq(r["AwayTeam"]), q, fonte, f"{r['HomeTeam']} - {r['AwayTeam']}"))
    usate, out = set(), {}
    for m in matches_list:
        if m.get("status") not in ("SCHEDULED", "TIMED"): continue
        nh, na = _norm_sq(m["homeTeam"].get("name")), _norm_sq(m["awayTeam"].get("name"))
        best_i, best_s = None, 0.0
        for i, (rh, ra, _, _, _) in enumerate(righe):
            s_ = min(_similarita(nh, rh), _similarita(na, ra))
            if s_ > best_s: best_i, best_s = i, s_
        if best_i is not None and best_s >= 0.75 and m.get("id") is not None:
            out[m["id"]] = (righe[best_i][2], righe[best_i][3])
            usate.add(best_i)
    non_abbinate = [r[4] for i, r in enumerate(righe) if i not in usate]
    return out, non_abbinate

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def calcola_statistiche_avanzate_match(sq_casa, sq_trasf, statistiche_squadre):
    dati_c = statistiche_squadre.get(sq_casa, {"media_gf": 1.4, "media_gs": 1.1})
    dati_t = statistiche_squadre.get(sq_trasf, {"media_gf": 1.1, "media_gs": 1.3})
    
    xg_c = round(dati_c["media_gf"] * 0.95 + dati_t["media_gs"] * 0.05, 2)
    xg_t = round(dati_t["media_gf"] * 0.95 + dati_c["media_gs"] * 0.05, 2)
    
    tiri_c = round(xg_c * 9.2 + 1.5, 1)
    tiri_t = round(xg_t * 9.0 + 1.4, 1)
    
    porta_c = round(tiri_c * 0.35, 1)
    porta_t = round(tiri_t * 0.33, 1)
    
    falli_c = round(12.0 + (dati_c["media_gs"] * 0.5), 1)
    falli_t = round(12.5 + (dati_t["media_gs"] * 0.5), 1)
    
    offside_c = round(1.4 + (xg_c * 0.2), 1)
    offside_t = round(1.3 + (xg_t * 0.2), 1)
    
    return {
        "Gol Attesi (xG)": (xg_c, xg_t),
        "Tiri Attesi": (tiri_c, tiri_t),
        "Tiri in Porta Attesi": (porta_c, porta_t),
        "Falli Attesi": (falli_c, falli_t),
        "Fuorigioco Attesi": (offside_c, offside_t)
    }

# --- ACQUISIZIONE DATI ---
statistiche_squadre = {}
matches_raw = []

for c_nome, (codice_lega, tipo_fonte) in selezionati_dict.items():
    if api_key:
        dati = scarica_dati_club(api_key, codice_lega)
        dati_classifica = scarica_classifica_club(api_key, codice_lega)
        if dati and "matches" in dati: 
            for m in dati["matches"]:
                m["Competizione"] = c_nome
                matches_raw.append(m)
        if dati_classifica and "standings" in dati_classifica:
            # L'API può restituire più tabelle (TOTAL, HOME, AWAY): prima si legge TOTAL, poi si
            # aggiungono le statistiche casa/trasferta senza sovrascrivere i valori complessivi.
            tabelle = sorted(dati_classifica["standings"], key=lambda s: s.get("type", "TOTAL") != "TOTAL")
            for s in tabelle:
                tipo = s.get("type", "TOTAL")
                for riga in s.get("table", []):
                    nome_sq = riga["team"]["name"]
                    giocate = max(riga["playedGames"], 1)
                    if tipo == "TOTAL":
                        statistiche_squadre[nome_sq] = {
                            "media_gf": riga["goalsFor"] / giocate,
                            "media_gs": riga["goalsAgainst"] / giocate,
                            "punti": riga["points"],
                            "partite": riga["playedGames"],
                            "forma": "N/D",
                            "Competizione": c_nome
                        }
                    elif tipo in ("HOME", "AWAY") and nome_sq in statistiche_squadre:
                        pref = "casa" if tipo == "HOME" else "trasf"
                        statistiche_squadre[nome_sq][f"{pref}_gf"] = riga["goalsFor"] / giocate
                        statistiche_squadre[nome_sq][f"{pref}_gs"] = riga["goalsAgainst"] / giocate
                        statistiche_squadre[nome_sq][f"{pref}_partite"] = riga["playedGames"]

# --- TAB 1: CALENDARIO & STUDIO ---
with tab_calendario:
    if matches_raw:
        lista = []
        for m in matches_raw:
            g_c = m["score"]["fullTime"].get("home") if m.get("score") and m["score"].get("fullTime") else None
            g_t = m["score"]["fullTime"].get("away") if m.get("score") and m["score"].get("fullTime") else None
            lista.append({
                "Competizione": m.get("Competizione", "Torneo"),
                "giornata": m.get("matchday", 0), "casa": m["homeTeam"]["name"],
                "trasferta": m["awayTeam"]["name"], "data": m["utcDate"][:10],
                "ora": m["utcDate"][11:16], "gol_casa": g_c if g_c is not None else "-",
                "gol_trasf": g_t if g_t is not None else "-", "stato": m["status"],
            })
        df = pd.DataFrame(lista)
        
        c_f1, c_f2 = str_lit.columns(2)
        competizioni_disponibili = sorted(df["Competizione"].unique())
        comp_sel = c_f1.selectbox("🏆 Torneo", competizioni_disponibili, key="mobile_comp_select")
        
        df_comp = df[df["Competizione"] == comp_sel]
        giornate = sorted(df_comp["giornata"].unique())
        if giornate:
            giornata_sel = c_f2.selectbox("📅 Giornata", giornate, key="mobile_giornata_select")
            for idx, row in df_comp[df_comp["giornata"] == giornata_sel].iterrows():
                str_lit.markdown(
                    f"""
                    <div style="background-color: #111827; padding: 18px; border-radius: 12px; border: 1px solid #1f2937; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['casa']}</span> 
                            <span style="color: #94a3b8; margin: 0 8px;">vs</span> 
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['trasferta']}</span><br>
                            <span style="color: #64748b; font-size: 12px;">📅 {row['data']} ore {row['ora']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="background-color: #1e293b; color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem;">{row['gol_casa']} - {row['gol_trasf']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if str_lit.button("📊 Analizza Match", key=f"btn_{idx}"):
                    str_lit.session_state["match_attivo"] = row
    else:
        str_lit.info("👈 Inserisci la chiave API nella barra laterale o seleziona un campionato valido.")

    if "match_attivo" in str_lit.session_state:
        m = str_lit.session_state["match_attivo"]
        sq_c, sq_t = m["casa"], m["trasferta"]
        lam_c = statistiche_squadre.get(sq_c, {}).get("media_gf", 1.4)
        lam_t = statistiche_squadre.get(sq_t, {}).get("media_gf", 1.1)

        p_c, p_p, p_t = 0.0, 0.0, 0.0
        prob_under_over = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
        prob_btts_yes = 0.0
        matrice_risultati = np.zeros((6, 6))

        for rc in range(6):
            for rt in range(6):
                prob = poisson_prob(lam_c, rc) * poisson_prob(lam_t, rt)
                matrice_risultati[rc, rt] = prob
                
                if rc > rt: p_c += prob
                elif rc == rt: p_p += prob
                else: p_t += prob
                
                tot_gol = rc + rt
                for soglia in prob_under_over:
                    if tot_gol <= soglia:
                        prob_under_over[soglia] += prob
                        
                if rc > 0 and rt > 0:
                    prob_btts_yes += prob

        tot = p_c + p_p + p_t
        p_c, p_p, p_t = (p_c/tot)*100, (p_p/tot)*100, (p_t/tot)*100
        prob_btts_no = 1.0 - prob_btts_yes

        stats_match = calcola_statistiche_avanzate_match(sq_c, sq_t, statistiche_squadre)

        str_lit.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 25px; margin-top: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 20px;">
                    <span style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">📋 SCHEDA DINAMICA DELLA PARTITA</span>
                    <span style="color: #94a3b8; font-size: 0.95rem;">📅 {m.get('data', 'N/D')} - {m.get('ora', '')}</span>
                </div>
            """,
            unsafe_allow_html=True
        )

        col_sq, col_st = str_lit.columns([1.2, 1.8])
        with col_sq:
            str_lit.markdown(
                f"""
                <div style="text-align: center; padding: 30px 20px; background: rgba(15, 23, 42, 0.6); border-radius: 14px; border: 1px solid #1e293b;">
                    <h2 style="color: #38bdf8; margin: 0; font-size: 1.4rem;">{sq_c}</h2>
                    <p style="color: #64748b; margin: 5px 0 20px 0; font-size: 0.85rem;">CASA</p>
                    <h3 style="color: #a855f7; margin: 0; font-size: 1.2rem;">VS</h3>
                    <h2 style="color: #f43f5e; margin: 20px 0 0 0; font-size: 1.4rem;">{sq_t}</h2>
                    <p style="color: #64748b; margin: 5px 0 0 0; font-size: 0.85rem;">TRASFERTA</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_st:
            str_lit.markdown("<p style='text-align: center; font-weight: 700; color: #94a3b8; margin-bottom: 15px; letter-spacing: 0.5px;'>STATISTICHE PREVISTE (LIVE CALC)</p>", unsafe_allow_html=True)
            for label, (val_c, val_t) in stats_match.items():
                str_lit.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: #111827; padding: 12px 18px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 10px;">
                        <span style="font-weight: 700; color: #38bdf8; font-size: 1.1rem; width: 60px; text-align: left;">{val_c}</span>
                        <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-align: center; flex-grow: 1;">{label}</span>
                        <span style="font-weight: 700; color: #f43f5e; font-size: 1.1rem; width: 60px; text-align: right;">{val_t}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown(f"<h3>🔬 Analisi Avanzata Poisson & Mercati: {sq_c} vs {sq_t}</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = str_lit.columns(3)
        col1.metric("Segno 1 (Casa)", f"{p_c:.1f}%", f"Quota equa: {100/p_c:.2f}")
        col2.metric("Segno X (Pareggio)", f"{p_p:.1f}%", f"Quota equa: {100/p_p:.2f}")
        col3.metric("Segno 2 (Trasferta)", f"{p_t:.1f}%", f"Quota equa: {100/p_t:.2f}")
        
        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown("#### ⚽ Analisi Under / Over & Goal / No Goal")
        
        uo_col1, uo_col2, uo_col3, uo_col4, uo_col5 = str_lit.columns(5)
        uo_col1.metric("Over 2.5", f"{(1 - prob_under_over[2.5])*100:.1f}%", f"Under: {prob_under_over[2.5]*100:.1f}%")
        uo_col2.metric("Over 1.5", f"{(1 - prob_under_over[1.5])*100:.1f}%", f"Under: {prob_under_over[1.5]*100:.1f}%")
        uo_col3.metric("Over 3.5", f"{(1 - prob_under_over[3.5])*100:.1f}%", f"Under: {prob_under_over[3.5]*100:.1f}%")
        uo_col4.metric("Goal (BTTS)", f"{prob_btts_yes*100:.1f}%", f"Quota: {100/(prob_btts_yes*100):.2f}" if prob_btts_yes > 0 else "N.D.")
        uo_col5.metric("No Goal", f"{prob_btts_no*100:.1f}%", f"Quota: {100/(prob_btts_no*100):.2f}" if prob_btts_no > 0 else "N.D.")

        str_lit.markdown("<br>", unsafe_allow_html=True)
        str_lit.markdown("#### 🎯 Top 3 Risultati Esatti più Probabili")
        
        lista_esatti = []
        for rc in range(6):
            for rt in range(6):
                lista_esatti.append(((rc, rt), matrice_risultati[rc, rt]))
        lista_esatti.sort(key=lambda x: x[1], reverse=True)
        
        c_res1, c_res2, c_res3 = str_lit.columns(3)
        with c_res1: str_lit.info(f"🥇 1° Esatto: **{lista_esatti[0][0][0]} - {lista_esatti[0][0][1]}** ({lista_esatti[0][1]*100:.1f}%)")
        with c_res2: str_lit.info(f"🥈 2° Esatto: **{lista_esatti[1][0][0]} - {lista_esatti[1][0][1]}** ({lista_esatti[1][1]*100:.1f}%)")
        with c_res3: str_lit.info(f"🥉 3° Esatto: **{lista_esatti[2][0][0]} - {lista_esatti[2][0][1]}** ({lista_esatti[2][1]*100:.1f}%)")
        
        str_lit.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: CLASSIFICA ---
with tab_classifica:
    str_lit.subheader("🏆 Classifica e Rendimento")
    if statistiche_squadre:
        df_cls = pd.DataFrame([{
            "Competizione": v.get("Competizione", "Torneo"),
            "Squadra": k, "Punti": v["punti"], 
            "Media GF": round(v["media_gf"], 2), 
            "Media GS": round(v["media_gs"], 2), 
            "DR": round(v["media_gf"] - v["media_gs"], 2)
        } for k, v in sorted(statistiche_squadre.items(), key=lambda x: x[1]["punti"], reverse=True)])
        str_lit.dataframe(df_cls, use_container_width=True, hide_index=True)
        str_lit.download_button("📥 Scarica CSV", df_cls.to_csv(index=False).encode("utf-8"), "classifica.csv", "text/csv")
    else:
        str_lit.warning("Dati non disponibili.")

# --- TAB 3: VALUE BET ---
with tab_value:
    str_lit.subheader("🔍 Calcolatore Value Bet & Kelly Criterion")
    c1, c2, c3 = str_lit.columns(3)
    p_stim = c1.slider("Probabilità Modello (%)", 1.0, 100.0, 45.0)
    q_book = c2.number_input("Quota Bookmaker", 1.01, 50.0, 2.30)
    bankroll = c3.number_input("Bankroll (€)", 10.0, 100000.0, 1000.0)
    
    q_equa = 100 / p_stim
    ev = ((p_stim / 100) * q_book) - 1
    b = q_book - 1
    p = p_stim / 100
    q = 1 - p
    kelly_fraction = max(0.0, ((b * p - q) / b)) * 0.25
    stake = bankroll * kelly_fraction

    m1, m2, m3 = str_lit.columns(3)
    m1.metric("Quota Equa", f"{q_equa:.2f}")
    m2.metric("Valore Atteso (EV)", f"{ev*100:+.2f}%")
    m3.metric("Stake (Kelly 25%)", f"€{stake:.2f}")

    if q_book > q_equa:
        str_lit.success("🔥 VALUE BET CERTIFICATA! Edge Positivo.")
    else:
        str_lit.error("❌ Nessun valore rilevato.")

# --- TAB 4: GRAFICI ---
with tab_grafici:
    str_lit.subheader("📊 Trend e Prestazioni")
    if statistiche_squadre:
        df_g = pd.DataFrame([{"Squadra": k, "Punti": v["punti"], "Media Gol Fatti": v["media_gf"]} for k, v in statistiche_squadre.items()])
        fig = px.bar(df_g, x="Squadra", y="Punti", color="Media Gol Fatti", template="plotly_dark", title="Punti e Potenziale Offensivo")
        str_lit.plotly_chart(fig, use_container_width=True)

COLONNE_VALORE = ["Competizione", "Giornata", "Partita", "Selezione", "Prob_Modello", "Prob_Mercato", "Quota_Book", "Edge"]

def genera_dataset_valore(matches_list, stats_dict, quote_map):
    """Una riga per ogni partita futura con quote reali: si sceglie l'esito con l'edge più alto
    (edge = prob_modello * quota_reale - 1). Prob_Mercato è la probabilità implicita senza margine."""
    righe = []
    for m in matches_list:
        if m.get("status") not in ("SCHEDULED", "TIMED"): continue
        quote = quote_map.get(m.get("id"))
        if not quote: continue
        h, a = m["homeTeam"]["name"], m["awayTeam"]["name"]
        if h not in stats_dict or a not in stats_dict: continue
        lc = stats_dict[h]["media_gf"]
        lt = stats_dict[a]["media_gf"]

        p_c, p_p, p_t = 0.0, 0.0, 0.0
        for rc in range(6):
            for rt in range(6):
                prob = poisson_prob(lc, rc) * poisson_prob(lt, rt)
                if rc > rt: p_c += prob
                elif rc == rt: p_p += prob
                else: p_t += prob
        tot_prob = p_c + p_p + p_t
        if tot_prob <= 0: continue
        p_c, p_p, p_t = p_c / tot_prob, p_p / tot_prob, p_t / tot_prob

        (q1, qx, q2), _fonte = quote
        margine = 1 / q1 + 1 / qx + 1 / q2
        candidati = [(f"1 ({h})", p_c, q1), ("X (Pareggio)", p_p, qx), (f"2 ({a})", p_t, q2)]
        selezione, prob_scelta, q_scelta = max(candidati, key=lambda c: c[1] * c[2])

        righe.append({
            "Competizione": m.get("Competizione", "Torneo"),
            "Giornata": m.get("matchday", 1),
            "Partita": f"{h} vs {a}",
            "Selezione": selezione,
            "Prob_Modello": round(prob_scelta * 100, 1),
            "Prob_Mercato": round((1 / q_scelta) / margine * 100, 1),
            "Quota_Book": round(q_scelta, 2),
            "Edge": round((prob_scelta * q_scelta - 1) * 100, 1),
        })

    if not righe:
        return pd.DataFrame(columns=COLONNE_VALORE)
    return pd.DataFrame(righe)

codice_sel = LEAGUES[campionato_scelto][0]
div_quote = ODDS_DIV.get(codice_sel)
quote_df = scarica_quote_fixtures() if div_quote else None
quote_map, quote_non_abbinate = costruisci_quote_map(matches_raw, quote_df, div_quote)
partite_future = sum(1 for m in matches_raw if m.get("status") in ("SCHEDULED", "TIMED"))

df_val = genera_dataset_valore(matches_raw, statistiche_squadre, quote_map)

# --- TAB 5: SCHEDINE AI ---
with tab_ai_schedine:
    str_lit.subheader("🤖 Schedine Smart & Combo AI")
    if not df_val.empty and "Competizione" in df_val.columns:
        c_s1, c_s2 = str_lit.columns(2)
        comp_schedina = c_s1.selectbox("Torneo", sorted(df_val["Competizione"].unique()), key="mobile_schedina_torneo")
        df_comp_val = df_val[df_val["Competizione"] == comp_schedina]
        giornate_val = sorted(df_comp_val["Giornata"].unique())
        if giornate_val:
            giornata_schedina = c_s2.selectbox("Giornata", giornate_val, key="mobile_schedina_giornata")
            df_filtrato_giornata = df_comp_val[df_comp_val["Giornata"] == giornata_schedina]
            if len(df_filtrato_giornata) >= 2:
                n_disp = len(df_filtrato_giornata)
                num_ev = str_lit.slider("Eventi in Multipla", 2, min(6, n_disp), min(3, n_disp)) if n_disp > 2 else 2
                subset = df_filtrato_giornata.sort_values(by="Edge", ascending=False).head(num_ev)
                quota_tot = np.prod(subset['Quota_Book'].values)
                str_lit.metric("📈 Quota Totale Accumulatore", f"{quota_tot:.2f}")
                str_lit.dataframe(subset, use_container_width=True, hide_index=True)
            else:
                str_lit.info("Servono almeno 2 partite con quote reali in questa giornata per comporre una multipla.")
    else:
        str_lit.info("Nessuna partita futura con quote reali disponibile per questo torneo.")

# --- TAB 6: VALUE FINDER ---
with tab_value_finder:
    str_lit.subheader("⚡ Scanner Value Finder")
    if not div_quote:
        str_lit.info("Quote reali non disponibili per questo torneo (football-data.co.uk non copre la Champions League).")
    elif quote_df is None:
        str_lit.warning("Impossibile scaricare le quote da football-data.co.uk. Riprova tra poco.")
    elif df_val.empty:
        str_lit.info("Nessuna partita futura con quote disponibili: il file delle quote copre solo il prossimo turno.")
    else:
        str_lit.caption(
            f"Quote reali (football-data.co.uk) abbinate a {len(quote_map)} partite su {partite_future} future. "
            "Edge = prob. modello × quota − 1. Un edge molto alto di solito indica un errore del modello, non un'occasione."
        )
        min_edge = str_lit.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
        str_lit.dataframe(df_val[df_val["Edge"] >= min_edge].sort_values("Edge", ascending=False), use_container_width=True, hide_index=True)
    if quote_non_abbinate:
        str_lit.caption("⚠️ Quote non abbinate (nomi squadra diversi): " + "; ".join(quote_non_abbinate))

# --- TAB 7: MONTE CARLO ---
with tab_monte_carlo:
    str_lit.subheader("🎲 Simulatore Monte Carlo Avanzato")
    if statistiche_squadre:
        nomi = sorted(list(statistiche_squadre.keys()))
        c1, c2, c3 = str_lit.columns(3)
        sq_c = c1.selectbox("Casa", nomi, index=0, key="mc_casa")
        sq_t = c2.selectbox("Trasferta", nomi, index=min(1, len(nomi)-1), key="mc_trasf")
        iterazioni = c3.slider("Iterazioni", 1000, 20000, 5000, step=1000, key="mc_iter")
        
        if str_lit.button("🚀 Esegui Simulazione", key="btn_mc"):
            lc = statistiche_squadre[sq_c]["media_gf"] * 1.05
            lt = statistiche_squadre[sq_t]["media_gf"] * 0.95
            
            gc_sim = np.random.poisson(lc, iterazioni)
            gt_sim = np.random.poisson(lt, iterazioni)
            
            v_c = np.sum(gc_sim > gt_sim) / (iterazioni / 100)
            v_p = np.sum(gc_sim == gt_sim) / (iterazioni / 100)
            v_t = np.sum(gc_sim < gt_sim) / (iterazioni / 100)
            
            tot_gol = gc_sim + gt_sim
            p_over15 = np.sum(tot_gol > 1.5) / (iterazioni / 100)
            p_over25 = np.sum(tot_gol > 2.5) / (iterazioni / 100)
            p_btts = np.sum((gc_sim > 0) & (gt_sim > 0)) / (iterazioni / 100)
            
            str_lit.markdown("#### 📊 Esiti 1X2 dalle Simulazioni")
            m1, m2, m3 = str_lit.columns(3)
            m1.metric("Vittoria Casa (1)", f"{v_c:.1f}%", f"Quota equa: {100/v_c:.2f}" if v_c > 0 else "N.D.")
            m2.metric("Pareggio (X)", f"{v_p:.1f}%", f"Quota equa: {100/v_p:.2f}" if v_p > 0 else "N.D.")
            m3.metric("Vittoria Trasferta (2)", f"{v_t:.1f}%", f"Quota equa: {100/v_t:.2f}" if v_t > 0 else "N.D.")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            str_lit.markdown("#### ⚽ Mercati di Gol (Simulati)")
            uo1, uo2, uo3 = str_lit.columns(3)
            uo1.metric("Over 1.5 Gol", f"{p_over15:.1f}%")
            uo2.metric("Over 2.5 Gol", f"{p_over25:.1f}%")
            uo3.metric("Goal (BTTS)", f"{p_btts:.1f}%")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            str_lit.markdown("#### 🎯 Top 3 Risultati Esatti più Frequenti")
            
            risultati_coppie = list(zip(gc_sim, gt_sim))
            conteggio_esatti = pd.Series(risultati_coppie).value_counts().head(3)
            
            r_col1, r_col2, r_col3 = str_lit.columns(3)
            colonne_res = [r_col1, r_col2, r_col3]
            
            for idx, ((rc, rt), count) in enumerate(conteggio_esatti.items()):
                perc = (count / iterazioni) * 100
                with colonne_res[idx]:
                    medaglia = ["🥇", "🥈", "🥉"][idx]
                    str_lit.info(f"{medaglia} **{rc} - {rt}** ({perc:.1f}% delle volte)")
            
            str_lit.markdown("<br>", unsafe_allow_html=True)
            df_dist = pd.DataFrame({"Gol Totali": tot_gol})
            fig_mc = px.histogram(df_dist, x="Gol Totali", nbins=int(max(tot_gol))+1, title="Distribuzione Frequenza Gol Totali (Iterazioni Monte Carlo)", template="plotly_dark")
            str_lit.plotly_chart(fig_mc, use_container_width=True)

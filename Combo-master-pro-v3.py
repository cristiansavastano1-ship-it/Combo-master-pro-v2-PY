
import difflib
import io
import math
import re
import unicodedata
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="COMBO Master Pro Betting & Deep Analytics Suite",
    layout="wide",
    page_icon="⚽",
)

# --- DESIGN MODERNO AVANZATO (CUSTOM CSS & GLASSMORPHISM) ---
st.markdown(
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
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 25px; border-radius: 20px; border: 1px solid #312e81; text-align: center; margin-bottom: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);">
        <h1 style="color: #f8fafc; font-weight: 800; font-size: 2.0rem; margin: 0; letter-spacing: -0.5px;">⚽ COMBO MASTER PRO AI</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px;">Dixon-Coles, backtest walk-forward e confronto con le quote reali</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================= DATI =======================
LEAGUES = {
    "Serie A (Italia)": "I1",
    "Premier League (Inghilterra)": "E0",
    "La Liga (Spagna)": "SP1",
    "Bundesliga (Germania)": "D1",
    "Ligue 1 (Francia)": "F1",
    "Eredivisie (Olanda)": "N1",
}
API_CODICI = {"I1": "SA", "E0": "PL", "SP1": "PD", "D1": "BL1", "F1": "FL1", "N1": "DED"}
# Dominio SENZA "www": con "www" football-data.co.uk blocca il traffico non britannico.
BASE_URL = "https://football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0"}
N_STAGIONI = 3
COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
STAT_COLONNE = {"Tiri": ("HS", "AS"), "Tiri in Porta": ("HST", "AST"), "Corner": ("HC", "AC"), "Falli": ("HF", "AF")}

_STOP_NOMI = {"fc", "ac", "as", "cf", "afc", "acf", "ssc", "us", "ud", "rc", "rcd", "sv", "vfl", "vfb", "fsv", "tsg",
              "fk", "cd", "ca", "sc", "bv", "de", "calcio", "ogc", "sco"}
_ALIAS_NOMI = {
    "inter": "internazionale milano", "verona": "hellas verona", "man united": "manchester united",
    "man city": "manchester city", "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers",
    "ath madrid": "club atletico madrid", "ath bilbao": "athletic club", "espanol": "espanyol barcelona",
    "bayern munich": "bayern munchen", "m gladbach": "borussia monchengladbach", "ein frankfurt": "eintracht frankfurt",
    "paris sg": "paris saint germain", "lyon": "olympique lyonnais", "rennes": "stade rennais",
    "brest": "stade brestois", "for sittard": "fortuna sittard",
}


def _norm_sq(nome):
    n = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = " ".join(t for t in n.split() if t not in _STOP_NOMI and not t.isdigit())
    return _ALIAS_NOMI.get(n, n)


def _similarita(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:
        return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


def mappa_nomi(nomi_api, nomi_target):
    """Associa i nomi di football-data.org a quelli di football-data.co.uk (usati dal modello)."""
    norm_t = {t: _norm_sq(t) for t in nomi_target}
    out = {}
    for n in nomi_api:
        na = _norm_sq(n)
        best, best_s = None, 0.0
        for t, nt in norm_t.items():
            s_ = _similarita(na, nt)
            if s_ > best_s:
                best, best_s = t, s_
        out[n] = best if best is not None and best_s >= 0.75 else n
    return out


def prima_tripla(riga, colonne):
    for cols in colonne:
        try:
            v = [float(riga[c]) for c in cols]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v):
            return v
    return None


def senza_margine(quote):
    p = 1.0 / np.asarray(quote, dtype=float)
    return p / p.sum()


def _leggi_csv(url):
    try:
        res = requests.get(url, timeout=20, headers=UA)
        if res.status_code != 200:
            return None
        return pd.read_csv(io.BytesIO(res.content), encoding="latin-1", on_bad_lines="skip")
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600)
def carica_storico(div, n_stagioni, giorno):
    """Risultati, statistiche e quote delle ultime n stagioni. `giorno` fa scadere la cache ogni giorno."""
    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n_stagioni):
        y = anno - k
        codice = f"{y % 100:02d}{(y + 1) % 100:02d}"
        d = _leggi_csv(f"{BASE_URL}/mmz4281/{codice}/{div}.csv")
        if d is None or not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(d.columns):
            continue
        d["Stag"] = y
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def carica_fixtures(div):
    d = _leggi_csv(f"{BASE_URL}/fixtures.csv")
    if d is None or not {"Div", "HomeTeam", "AwayTeam"}.issubset(d.columns):
        return None
    d = d[d["Div"] == div].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
    return d.reset_index(drop=True)


@st.cache_data(ttl=3600)
def scarica_calendario_api(chiave, codice):
    if not chiave:
        return None
    try:
        res = requests.get(f"https://api.football-data.org/v4/competitions/{codice}/matches",
                           headers={"X-Auth-Token": chiave}, timeout=15)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def assegna_giornate(righe):
    """Giornata approssimata = numero della partita della squadra che ne ha già giocate di più + 1."""
    conta = {}
    for r in righe:
        g = max(conta.get(r["Casa"], 0), conta.get(r["Trasferta"], 0)) + 1
        r["Giornata"] = g
        conta[r["Casa"]] = conta.get(r["Casa"], 0) + 1
        conta[r["Trasferta"]] = conta.get(r["Trasferta"], 0) + 1
    return righe


def costruisci_calendario(div, df_storico, fx, chiave, squadre_modello):
    """Calendario completo da football-data.org (se c'è la API key), altrimenti ridotto:
    partite giocate dalla stagione corrente + prossimo turno da fixtures.csv."""
    dati = scarica_calendario_api(chiave, API_CODICI[div]) if chiave else None
    if dati and dati.get("matches"):
        nomi = {m["homeTeam"]["name"] for m in dati["matches"]} | {m["awayTeam"]["name"] for m in dati["matches"]}
        nomi = {n for n in nomi if n}
        mp = mappa_nomi(nomi, squadre_modello)
        righe = []
        for m in dati["matches"]:
            h, a = m["homeTeam"].get("name"), m["awayTeam"].get("name")
            if not h or not a:
                continue
            ft = (m.get("score") or {}).get("fullTime") or {}
            ts = pd.to_datetime(m["utcDate"], utc=True).tz_convert("Europe/Rome")
            righe.append({"Giornata": m.get("matchday") or 0, "Data": ts.strftime("%Y-%m-%d"), "Ora": ts.strftime("%H:%M"),
                          "Casa": mp.get(h, h), "Trasferta": mp.get(a, a),
                          "GolC": ft.get("home"), "GolT": ft.get("away"), "Stato": m.get("status", "")})
        return pd.DataFrame(righe), "football-data.org (calendario completo)"

    righe = []
    stag = df_storico[df_storico["Stag"] == df_storico["Stag"].max()]
    for _, r in stag.iterrows():
        righe.append({"Data": r["Date"].strftime("%Y-%m-%d"), "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "",
                      "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": int(r["FTHG"]), "GolT": int(r["FTAG"]),
                      "Stato": "FINISHED", "_d": r["Date"]})
    if fx is not None:
        for _, r in fx.iterrows():
            d = r["Date"] if "Date" in r and pd.notna(r["Date"]) else pd.Timestamp.max
            righe.append({"Data": d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "",
                          "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": None, "GolT": None, "Stato": "SCHEDULED", "_d": d})
    righe.sort(key=lambda r: r["_d"])
    righe = assegna_giornate(righe)
    return pd.DataFrame(righe).drop(columns="_d"), "football-data.co.uk (partite giocate + prossimo turno)"


# ======================= MODELLO =======================
def adatta_modello(df, data_rif, ridge, halflife, usa_dc=True):
    """Poisson con forza d'attacco e di difesa per squadra, vantaggio casa, shrinkage (ridge) verso la media,
    pesi che decadono col tempo e correzione Dixon-Coles sui punteggi bassi."""
    squadre = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(squadre)
    idx = {t: i for i, t in enumerate(squadre)}
    hi = df["HomeTeam"].map(idx).to_numpy()
    ai = df["AwayTeam"].map(idx).to_numpy()
    x = df["FTHG"].to_numpy(float)
    y = df["FTAG"].to_numpy(float)
    eta_gg = (data_rif - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-np.log(2) * eta_gg / halflife) if halflife and halflife > 0 else np.ones(len(df))

    def f(th):
        mu, hm = th[0], th[1]
        att, dfn = th[2:2 + n], th[2 + n:]
        eh = mu + hm + att[hi] + dfn[ai]
        ea = mu + att[ai] + dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        nll = -np.sum(w * (x * eh - lh + y * ea - la)) + ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (x - lh), w * (y - la)
        g = np.zeros_like(th)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
        g[2 + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * dfn
        return nll, g

    th0 = np.zeros(2 + 2 * n)
    th0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.3))
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    mu, hm, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]

    rho = 0.0
    if usa_dc:
        lh = np.exp(mu + hm + att[hi] + dfn[ai])
        la = np.exp(mu + att[ai] + dfn[hi])
        m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

        def neg(r):
            t = np.ones(len(x))
            t[m00] = 1 - lh[m00] * la[m00] * r
            t[m01] = 1 + lh[m01] * r
            t[m10] = 1 + la[m10] * r
            t[m11] = 1 - r
            return -np.sum(w * np.log(np.maximum(t, 1e-6)))

        rho = float(minimize_scalar(neg, bounds=(-0.3, 0.3), method="bounded").x)

    return {"squadre": squadre, "idx": idx, "mu": float(mu), "hm": float(hm), "att": att, "dfn": dfn, "rho": rho}


def lambda_partita(m, casa, trasf):
    ic, it = m["idx"].get(casa), m["idx"].get(trasf)
    ac, dc = (m["att"][ic], m["dfn"][ic]) if ic is not None else (0.0, 0.0)  # squadra senza storico = media
    at, dt = (m["att"][it], m["dfn"][it]) if it is not None else (0.0, 0.0)
    return math.exp(m["mu"] + m["hm"] + ac + dt), math.exp(m["mu"] + at + dc)


def matrice_punteggi(lh, la, rho, mg=10):
    k = np.arange(mg + 1)
    M = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    if rho != 0:
        M[0, 0] *= max(1 - lh * la * rho, 1e-6)
        M[0, 1] *= max(1 + lh * rho, 1e-6)
        M[1, 0] *= max(1 + la * rho, 1e-6)
        M[1, 1] *= max(1 - rho, 1e-6)
    return M / M.sum()


def mercati(M):
    tot = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    d = {"p1": float(np.tril(M, -1).sum()), "px": float(np.trace(M)), "p2": float(np.triu(M, 1).sum()),
         "btts": float(M[1:, 1:].sum())}
    for s in (1.5, 2.5, 3.5):
        d[f"over{s}"] = float(M[tot > s].sum())
    return d


# ======================= BACKTEST E PARAMETRI AUTOMATICI =======================
def esegui_backtest_df(df, ridge, halflife, usa_dc, passo):
    """Walk-forward: ogni blocco di `passo` giorni è previsto solo con le partite precedenti.
    La prima stagione serve da storico iniziale."""
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return pd.DataFrame()
    t = df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    righe = []
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[df["Date"] < t]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, usa_dc)
            esiti_tr = np.where(train["FTHG"] > train["FTAG"], 0, np.where(train["FTHG"] == train["FTAG"], 1, 2))
            freq = np.bincount(esiti_tr, minlength=3) / len(esiti_tr)
            for _, r in blocco.iterrows():
                lh, la = lambda_partita(m, r["HomeTeam"], r["AwayTeam"])
                mk = mercati(matrice_punteggi(lh, la, m["rho"]))
                esito = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
                qp, qc = prima_tripla(r, COL_PRESA), prima_tripla(r, COL_CHIUSURA)
                pm = senza_margine(qc) if qc else [np.nan] * 3
                righe.append({
                    "Date": r["Date"], "Partita": f"{r['HomeTeam']} - {r['AwayTeam']}", "y": esito,
                    "p1": mk["p1"], "px": mk["px"], "p2": mk["p2"], "f1": freq[0], "fx": freq[1], "f2": freq[2],
                    "q1": qp[0] if qp else np.nan, "qx": qp[1] if qp else np.nan, "q2": qp[2] if qp else np.nan,
                    "m1": pm[0], "mx": pm[1], "m2": pm[2],
                })
        t += pd.Timedelta(days=passo)
    return pd.DataFrame(righe)


def _ll_brier(P, y):
    P = np.asarray(P, float)
    ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))
    br = np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))
    return ll, br


def metriche_backtest(bt):
    y = bt["y"].to_numpy(int)
    out = {"n": len(bt)}
    out["modello"] = _ll_brier(bt[["p1", "px", "p2"]].to_numpy(), y)
    out["frequenze"] = _ll_brier(bt[["f1", "fx", "f2"]].to_numpy(), y)
    ok = bt[["m1", "mx", "m2"]].notna().all(axis=1).to_numpy()
    if ok.sum() > 30:
        out["mercato"] = _ll_brier(bt.loc[ok, ["m1", "mx", "m2"]].to_numpy(), y[ok])
        out["modello_su_mercato"] = _ll_brier(bt.loc[ok, ["p1", "px", "p2"]].to_numpy(), y[ok])
        out["n_mercato"] = int(ok.sum())
    return out


def tabella_scommesse(bt, soglie=(0, 2, 5, 8, 10)):
    ok = bt[["q1", "qx", "q2"]].notna().all(axis=1)
    b = bt[ok]
    if b.empty:
        return pd.DataFrame()
    P, Q, M = b[["p1", "px", "p2"]].to_numpy(), b[["q1", "qx", "q2"]].to_numpy(), b[["m1", "mx", "m2"]].to_numpy()
    y = b["y"].to_numpy(int)
    edge = P * Q - 1
    k = edge.argmax(axis=1)
    e_best = edge[np.arange(len(b)), k]
    q_k = Q[np.arange(len(b)), k]
    profitto = np.where(y == k, q_k - 1, -1.0)
    clv = q_k * M[np.arange(len(b)), k] - 1
    righe = []
    for s in soglie:
        sel = e_best >= s / 100
        n = int(sel.sum())
        if n == 0:
            righe.append({"Edge min %": s, "Scommesse": 0, "Vinte %": None, "ROI %": None, "± ROI %": None, "CLV medio %": None})
            continue
        pr, c = profitto[sel], clv[sel]
        righe.append({
            "Edge min %": s, "Scommesse": n, "Vinte %": round((y[sel] == k[sel]).mean() * 100, 1),
            "ROI %": round(pr.mean() * 100, 1),
            "± ROI %": round(pr.std(ddof=1) / math.sqrt(n) * 100, 1) if n > 1 else None,
            "CLV medio %": round(float(np.nanmean(c)) * 100, 1) if np.isfinite(c).any() else None,
        })
    return pd.DataFrame(righe)


def tabella_calibrazione(bt, bins=10):
    P = bt[["p1", "px", "p2"]].to_numpy().ravel()
    O = np.eye(3)[bt["y"].to_numpy(int)].ravel()
    cut = np.minimum((P * bins).astype(int), bins - 1)
    righe = []
    for b in range(bins):
        sel = cut == b
        if sel.sum() >= 10:
            righe.append({"Prob. prevista %": round(P[sel].mean() * 100, 1),
                          "Frequenza reale %": round(O[sel].mean() * 100, 1), "Casi": int(sel.sum())})
    return pd.DataFrame(righe)


@st.cache_data(ttl=6 * 3600, show_spinner="Ottimizzo i parametri del modello (solo la prima volta di giornata)...")
def parametri_auto(div, giorno):
    """Sceglie da solo shrinkage ed emivita: griglia piccola, valutata con il backtest walk-forward (log loss)."""
    df = carica_storico(div, N_STAGIONI, giorno)
    default = {"ridge": 2.0, "halflife": 365, "log_loss": None, "ricerca": [], "auto": False}
    if df is None or df["Stag"].nunique() < 2:
        return default
    ricerca = []
    for r_ in (0.5, 2.0, 6.0):
        for h_ in (180, 365, 730, 0):
            bt = esegui_backtest_df(df, r_, h_, True, 28)
            if not bt.empty:
                ricerca.append({"Shrinkage λ": r_, "Emivita (gg)": h_ if h_ else "nessuna",
                                "Log loss": round(metriche_backtest(bt)["modello"][0], 4), "_r": r_, "_h": h_})
    if not ricerca:
        return default
    best = min(ricerca, key=lambda z: z["Log loss"])
    return {"ridge": best["_r"], "halflife": best["_h"], "log_loss": best["Log loss"], "auto": True,
            "ricerca": [{k: v for k, v in z.items() if not k.startswith("_")} for z in ricerca]}


@st.cache_data(ttl=6 * 3600)
def modello_corrente(div, ridge, halflife, giorno):
    df = carica_storico(div, N_STAGIONI, giorno)
    if df is None or len(df) < 100:
        return None
    return adatta_modello(df, pd.Timestamp.now().normalize(), ridge, halflife, True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_cached(div, ridge, halflife, passo, giorno):
    df = carica_storico(div, N_STAGIONI, giorno)
    return None if df is None else esegui_backtest_df(df, ridge, halflife, True, passo)


# ======================= STATISTICHE, CLASSIFICA =======================
def stats_attese(df, casa, trasf, k=5):
    """Medie storiche reali (non formule inventate), con shrinkage verso la media del campionato."""
    out = {}
    for nome, (h, a) in STAT_COLONNE.items():
        if h not in df.columns or a not in df.columns:
            continue
        d = df.dropna(subset=[h, a])
        if d.empty:
            continue
        lega_h, lega_a = d[h].mean(), d[a].mean()

        def sm(serie, lega):
            return (serie.sum() + k * lega) / (len(serie) + k)

        dc, dt = d[d["HomeTeam"] == casa], d[d["AwayTeam"] == trasf]
        out[nome] = (round((sm(dc[h], lega_h) + sm(dt[h], lega_h)) / 2, 1),
                     round((sm(dt[a], lega_a) + sm(dc[a], lega_a)) / 2, 1))
    return out


def tabella_classifica(df_stag):
    righe = {}
    for r in df_stag.itertuples():
        for sq, gf, gs in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            d = righe.setdefault(sq, {"G": 0, "Pt": 0, "GF": 0, "GS": 0})
            d["G"] += 1
            d["GF"] += gf
            d["GS"] += gs
            d["Pt"] += 3 if gf > gs else (1 if gf == gs else 0)
    out = pd.DataFrame([{
        "Squadra": sq, "Giocate": v["G"], "Punti": v["Pt"],
        "Media GF": round(v["GF"] / v["G"], 2), "Media GS": round(v["GS"] / v["G"], 2),
        "DR": round((v["GF"] - v["GS"]) / v["G"], 2),
    } for sq, v in righe.items()])
    return out.sort_values(["Punti", "DR"], ascending=False).reset_index(drop=True) if not out.empty else out


# ======================= INTERFACCIA =======================
st.markdown("### 🏆 Seleziona Campionato (Mobile & Desktop)")
campionato = st.selectbox("Scegli il torneo da analizzare", list(LEAGUES.keys()), label_visibility="collapsed")
div = LEAGUES[campionato]

try:
    _chiave_default = st.secrets.get("FOOTBALL_DATA_API_KEY", "")
except Exception:
    _chiave_default = ""
st.sidebar.markdown("### ⚙ Configurazione API")
chiave_api = st.sidebar.text_input("🔑 API Key football-data.org (facoltativa)", value=_chiave_default, type="password")
st.sidebar.caption("Con la chiave il calendario mostra tutte le giornate. Senza, mostra le partite giocate e il prossimo turno.")
st.sidebar.markdown("---")
st.sidebar.info("💡 Il modello sceglie da solo i propri parametri (nessuna impostazione da fare). Da cellulare puoi cambiare campionato dal menu in alto.")

giorno = date.today().isoformat()
df_storico = carica_storico(div, N_STAGIONI, giorno)
if df_storico is None or len(df_storico) < 100:
    st.warning("Impossibile scaricare lo storico da football-data.co.uk (o dati insufficienti). Riprova tra poco.")
    st.stop()
par = parametri_auto(div, giorno)
modello = modello_corrente(div, par["ridge"], par["halflife"], giorno)
fx = carica_fixtures(div)
calendario, fonte_cal = costruisci_calendario(div, df_storico, fx, chiave_api, modello["squadre"])

# Value finder: partite del prossimo turno con quote reali
righe_val = []
if fx is not None:
    for _, r in fx.iterrows():
        q = prima_tripla(r, COL_PRESA)
        if not q:
            continue
        casa, trasf = r["HomeTeam"], r["AwayTeam"]
        lh, la = lambda_partita(modello, casa, trasf)
        mk = mercati(matrice_punteggi(lh, la, modello["rho"]))
        P = np.array([mk["p1"], mk["px"], mk["p2"]])
        pm = senza_margine(q)
        nomi = [f"1 ({casa})", "X (Pareggio)", f"2 ({trasf})"]
        k = int(np.argmax(P * np.array(q)))
        righe_val.append({
            "Data": r["Date"].strftime("%d/%m") if "Date" in r and pd.notna(r["Date"]) else "",
            "Partita": f"{casa} vs {trasf}", "Selezione": nomi[k],
            "P1 %": round(P[0] * 100, 1), "PX %": round(P[1] * 100, 1), "P2 %": round(P[2] * 100, 1),
            "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1),
            "Quota_Book": round(q[k], 2), "Edge": round((P[k] * q[k] - 1) * 100, 1),
            "Nota": "⚠ senza storico" if (casa not in modello["idx"] or trasf not in modello["idx"]) else "",
        })
df_value = pd.DataFrame(righe_val).sort_values("Edge", ascending=False) if righe_val else pd.DataFrame()

tab_calendario, tab_classifica, tab_value_finder, tab_ai_schedine, tab_monte_carlo, tab_backtest, tab_kelly = st.tabs([
    "📅 Calendario", "🏆 Classifica", "⚡ Value Finder", "🤖 Schedine AI", "🎲 Monte Carlo", "🧪 Backtest", "🧮 Value Bet",
])


def mostra_analisi(casa, trasf, data, ora):
    lh, la = lambda_partita(modello, casa, trasf)
    M = matrice_punteggi(lh, la, modello["rho"])
    mk = mercati(M)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 18px 25px; margin-top: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">📋 SCHEDA DINAMICA DELLA PARTITA</span>
            <span style="color: #94a3b8; font-size: 0.95rem;">📅 {data} {ora}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_sq, col_st = st.columns([1.2, 1.8])
    with col_sq:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 30px 20px; background: rgba(15, 23, 42, 0.6); border-radius: 14px; border: 1px solid #1e293b;">
                <h2 style="color: #38bdf8; margin: 0; font-size: 1.4rem;">{casa}</h2>
                <p style="color: #64748b; margin: 5px 0 20px 0; font-size: 0.85rem;">CASA</p>
                <h3 style="color: #a855f7; margin: 0; font-size: 1.2rem;">VS</h3>
                <h2 style="color: #f43f5e; margin: 20px 0 0 0; font-size: 1.4rem;">{trasf}</h2>
                <p style="color: #64748b; margin: 5px 0 0 0; font-size: 0.85rem;">TRASFERTA</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_st:
        st.markdown("<p style='text-align: center; font-weight: 700; color: #94a3b8; margin-bottom: 15px; letter-spacing: 0.5px;'>STATISTICHE ATTESE (MEDIE STORICHE REALI)</p>", unsafe_allow_html=True)
        righe_st = {"Gol Attesi (modello)": (round(lh, 2), round(la, 2))}
        righe_st.update(stats_attese(df_storico, casa, trasf))
        for label, (vc, vt) in righe_st.items():
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: #111827; padding: 12px 18px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 10px;">
                    <span style="font-weight: 700; color: #38bdf8; font-size: 1.1rem; width: 60px; text-align: left;">{vc}</span>
                    <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-align: center; flex-grow: 1;">{label}</span>
                    <span style="font-weight: 700; color: #f43f5e; font-size: 1.1rem; width: 60px; text-align: right;">{vt}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(f"<h3>🔬 Analisi Poisson / Dixon-Coles & Mercati: {casa} vs {trasf}</h3>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Segno 1 (Casa)", f"{mk['p1']*100:.1f}%", f"Quota equa: {1/mk['p1']:.2f}")
    a2.metric("Segno X (Pareggio)", f"{mk['px']*100:.1f}%", f"Quota equa: {1/mk['px']:.2f}")
    a3.metric("Segno 2 (Trasferta)", f"{mk['p2']*100:.1f}%", f"Quota equa: {1/mk['p2']:.2f}")
    st.markdown("#### ⚽ Analisi Under / Over & Goal / No Goal")
    u1, u2, u3, u4, u5 = st.columns(5)
    u1.metric("Over 2.5", f"{mk['over2.5']*100:.1f}%", f"Under: {(1-mk['over2.5'])*100:.1f}%")
    u2.metric("Over 1.5", f"{mk['over1.5']*100:.1f}%", f"Under: {(1-mk['over1.5'])*100:.1f}%")
    u3.metric("Over 3.5", f"{mk['over3.5']*100:.1f}%", f"Under: {(1-mk['over3.5'])*100:.1f}%")
    u4.metric("Goal (BTTS)", f"{mk['btts']*100:.1f}%", f"Quota: {1/mk['btts']:.2f}")
    u5.metric("No Goal", f"{(1-mk['btts'])*100:.1f}%", f"Quota: {1/(1-mk['btts']):.2f}")
    st.markdown("#### 🎯 Top 3 Risultati Esatti più Probabili")
    flat = sorted(((i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])), key=lambda t: t[2], reverse=True)[:3]
    for col, medaglia, nome, (i, j, p) in zip(st.columns(3), ["🥇", "🥈", "🥉"], ["1°", "2°", "3°"], flat):
        col.info(f"{medaglia} {nome} Esatto: **{i} - {j}** ({p*100:.1f}%)")
    if casa not in modello["idx"] or trasf not in modello["idx"]:
        st.caption("⚠ Una delle due squadre non ha storico (es. neopromossa): è trattata come squadra media.")


# --- CALENDARIO ---
with tab_calendario:
    st.caption(f"Fonte: {fonte_cal}")
    if calendario.empty:
        st.info("Nessuna partita disponibile.")
    else:
        giornate = sorted(g for g in calendario["Giornata"].unique() if g)
        non_giocate = calendario[calendario["Stato"] != "FINISHED"]
        g_def = int(non_giocate["Giornata"].min()) if not non_giocate.empty and non_giocate["Giornata"].min() else int(giornate[-1])
        cc1, cc2 = st.columns(2)
        cc1.selectbox("🏆 Torneo", [campionato], key="cal_torneo")
        giornata_sel = cc2.selectbox("📅 Giornata", giornate, index=giornate.index(g_def) if g_def in giornate else 0, key="cal_giornata")
        for idx, row in calendario[calendario["Giornata"] == giornata_sel].iterrows():
            gc = row["GolC"] if pd.notna(row["GolC"]) else "-"
            gt = row["GolT"] if pd.notna(row["GolT"]) else "-"
            gc = int(gc) if gc != "-" else gc
            gt = int(gt) if gt != "-" else gt
            st.markdown(
                f"""
                <div style="background-color: #111827; padding: 18px; border-radius: 12px; border: 1px solid #1f2937; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['Casa']}</span>
                        <span style="color: #94a3b8; margin: 0 8px;">vs</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">{row['Trasferta']}</span><br>
                        <span style="color: #64748b; font-size: 12px;">📅 {row['Data']} ore {row['Ora']}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="background-color: #1e293b; color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem;">{gc} - {gt}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("📊 Analizza Match", key=f"btn_{div}_{idx}"):
                st.session_state["match_attivo"] = {"div": div, "casa": row["Casa"], "trasf": row["Trasferta"], "data": row["Data"], "ora": row["Ora"]}
    att_m = st.session_state.get("match_attivo")
    if att_m and att_m.get("div") == div:
        mostra_analisi(att_m["casa"], att_m["trasf"], att_m["data"], att_m["ora"])

# --- CLASSIFICA ---
with tab_classifica:
    st.subheader("🏆 Classifica e Rendimento")
    df_cls = tabella_classifica(df_storico[df_storico["Stag"] == df_storico["Stag"].max()])
    if df_cls.empty:
        st.warning("Dati non disponibili.")
    else:
        st.caption("Calcolata dai risultati della stagione in corso (football-data.co.uk). Media GF/GS e DR sono per partita.")
        st.dataframe(df_cls, use_container_width=True, hide_index=True)
        st.download_button("📥 Scarica CSV", df_cls.to_csv(index=False).encode("utf-8"), "classifica.csv", "text/csv")
        fig = px.bar(df_cls, x="Squadra", y="Punti", color="Media GF", template="plotly_dark", title="Punti e Potenziale Offensivo")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 📊 Forza Squadre secondo il modello (attacco e difesa)")
    st.caption("Scala logaritmica rispetto alla squadra media (0). Attacco alto = segna di più; difesa alta = subisce meno.")
    df_f = pd.DataFrame({"Squadra": modello["squadre"], "Attacco": np.round(modello["att"], 3), "Difesa": np.round(-modello["dfn"], 3)})
    df_f["Totale"] = (df_f["Attacco"] + df_f["Difesa"]).round(3)
    k1, k2, k3 = st.columns(3)
    k1.metric("Vantaggio casa (gol ×)", f"{math.exp(modello['hm']):.2f}")
    k2.metric("Gol base (squadra media, in trasferta)", f"{math.exp(modello['mu']):.2f}")
    k3.metric("Rho Dixon-Coles", f"{modello['rho']:+.3f}")
    st.dataframe(df_f.sort_values("Totale", ascending=False), use_container_width=True, hide_index=True)

# --- VALUE FINDER ---
with tab_value_finder:
    st.subheader("⚡ Scanner Value Finder")
    if fx is None:
        st.warning("Impossibile scaricare il prossimo turno e le quote (fixtures.csv).")
    elif df_value.empty:
        st.info("Nessuna partita con quote nel prossimo turno (il file copre solo i prossimi giorni).")
    else:
        st.caption("Edge = prob. modello × quota − 1, sulla media dei bookmaker. Un edge alto di solito è un errore del modello: "
                   "controlla nel tab Backtest da quale soglia l'edge ha avuto CLV positivo.")
        min_edge = st.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
        st.dataframe(df_value[df_value["Edge"] >= min_edge], use_container_width=True, hide_index=True)
        if (df_value["Nota"] != "").any():
            st.caption("⚠ Squadre senza storico (es. neopromosse) sono trattate come squadra media: probabilmente sopravvalutate.")

# --- SCHEDINE AI ---
with tab_ai_schedine:
    st.subheader("🤖 Schedine Smart & Combo AI")
    base = df_value[df_value["Nota"] == ""] if not df_value.empty else df_value
    if base.empty or len(base) < 2:
        st.info("Servono almeno 2 partite del prossimo turno con quote reali e squadre con storico.")
    else:
        n_disp = len(base)
        num_ev = st.slider("Eventi in Multipla", 2, min(6, n_disp), min(3, n_disp)) if n_disp > 2 else 2
        subset = base.head(num_ev)
        quota_tot = float(np.prod(subset["Quota_Book"].values))
        prob_comb = float(np.prod(subset["Prob_Modello"].values / 100))
        s1, s2, s3 = st.columns(3)
        s1.metric("📈 Quota Totale Accumulatore", f"{quota_tot:.2f}")
        s2.metric("Prob. combinata (modello)", f"{prob_comb*100:.1f}%")
        s3.metric("Valore atteso (EV)", f"{(prob_comb*quota_tot-1)*100:+.1f}%")
        st.dataframe(subset, use_container_width=True, hide_index=True)
        st.caption("La multipla moltiplica sia le quote sia gli errori del modello e il margine del bookmaker: l'EV mostrato è ottimistico.")

# --- MONTE CARLO ---
with tab_monte_carlo:
    st.subheader("🎲 Simulatore Monte Carlo Avanzato")
    nomi = modello["squadre"]
    c1, c2, c3 = st.columns(3)
    sq_c = c1.selectbox("Casa", nomi, index=0, key="mc_casa")
    sq_t = c2.selectbox("Trasferta", nomi, index=min(1, len(nomi) - 1), key="mc_trasf")
    iterazioni = c3.slider("Iterazioni", 1000, 20000, 5000, step=1000, key="mc_iter")
    if st.button("🚀 Esegui Simulazione", key="btn_mc"):
        if sq_c == sq_t:
            st.info("Scegli due squadre diverse.")
        else:
            lh, la = lambda_partita(modello, sq_c, sq_t)
            M = matrice_punteggi(lh, la, modello["rho"])  # stessa distribuzione dell'analisi (con Dixon-Coles)
            rng = np.random.default_rng()
            scelta = rng.choice(M.size, size=iterazioni, p=M.ravel())
            gc_sim, gt_sim = scelta // M.shape[1], scelta % M.shape[1]
            pc = 100.0 / iterazioni
            v_c, v_p, v_t = np.sum(gc_sim > gt_sim) * pc, np.sum(gc_sim == gt_sim) * pc, np.sum(gc_sim < gt_sim) * pc
            tot_gol = gc_sim + gt_sim
            p_over15, p_over25 = np.sum(tot_gol > 1.5) * pc, np.sum(tot_gol > 2.5) * pc
            p_btts = np.sum((gc_sim > 0) & (gt_sim > 0)) * pc
            st.markdown("#### 📊 Esiti 1X2 dalle Simulazioni")
            m1, m2, m3 = st.columns(3)
            m1.metric("Vittoria Casa (1)", f"{v_c:.1f}%", f"Quota equa: {100/v_c:.2f}" if v_c > 0 else "N.D.")
            m2.metric("Pareggio (X)", f"{v_p:.1f}%", f"Quota equa: {100/v_p:.2f}" if v_p > 0 else "N.D.")
            m3.metric("Vittoria Trasferta (2)", f"{v_t:.1f}%", f"Quota equa: {100/v_t:.2f}" if v_t > 0 else "N.D.")
            st.markdown("#### ⚽ Mercati di Gol (Simulati)")
            uo1, uo2, uo3 = st.columns(3)
            uo1.metric("Over 1.5 Gol", f"{p_over15:.1f}%")
            uo2.metric("Over 2.5 Gol", f"{p_over25:.1f}%")
            uo3.metric("Goal (BTTS)", f"{p_btts:.1f}%")
            st.markdown("#### 🎯 Top 3 Risultati Esatti più Frequenti")
            conteggio = pd.Series(list(zip(gc_sim.tolist(), gt_sim.tolist()))).value_counts().head(3)
            for col, medaglia, ((rc, rt), count) in zip(st.columns(3), ["🥇", "🥈", "🥉"], conteggio.items()):
                col.info(f"{medaglia} **{rc} - {rt}** ({count / iterazioni * 100:.1f}% delle volte)")
            fig_mc = px.histogram(pd.DataFrame({"Gol Totali": tot_gol}), x="Gol Totali", nbins=int(tot_gol.max()) + 1,
                                  title="Distribuzione Frequenza Gol Totali (Iterazioni Monte Carlo)", template="plotly_dark")
            st.plotly_chart(fig_mc, use_container_width=True)

# --- BACKTEST ---
with tab_backtest:
    st.subheader("🧪 Backtest walk-forward")
    st.caption("Ogni blocco di partite è previsto da un modello addestrato solo sul passato; la prima stagione serve da storico iniziale. "
               "Si confronta con le frequenze storiche (modello banale) e con le quote di chiusura senza margine (il riferimento da battere).")
    if par["auto"]:
        st.caption(f"Parametri scelti automaticamente: shrinkage λ = {par['ridge']}, emivita = "
                   f"{par['halflife'] if par['halflife'] else 'nessun decadimento'}. Dixon-Coles sempre attivo.")
        with st.expander("Come sono stati scelti"):
            st.dataframe(pd.DataFrame(par["ricerca"]).sort_values("Log loss"), use_container_width=True, hide_index=True)
            st.caption("Scelta fatta sugli stessi dati del backtest: il log loss mostrato qui sotto è quindi un po' ottimistico.")
    if st.button("🚀 Esegui Backtest"):
        with st.spinner("Calcolo in corso..."):
            st.session_state["bt"] = backtest_cached(div, par["ridge"], par["halflife"], 14, giorno)
            st.session_state["bt_div"] = div
    bt = st.session_state.get("bt") if st.session_state.get("bt_div") == div else None
    if bt is not None and not bt.empty:
        mt = metriche_backtest(bt)
        st.markdown(f"#### 📏 Qualità delle probabilità 1X2 ({mt['n']} partite)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Log loss modello", f"{mt['modello'][0]:.4f}", f"Brier {mt['modello'][1]:.4f}", delta_color="off")
        r2.metric("Log loss frequenze storiche", f"{mt['frequenze'][0]:.4f}", f"Brier {mt['frequenze'][1]:.4f}", delta_color="off")
        if "mercato" in mt:
            r3.metric("Log loss mercato (chiusura)", f"{mt['mercato'][0]:.4f}", f"Brier {mt['mercato'][1]:.4f}", delta_color="off")
            gap = mt["modello_su_mercato"][0] - mt["mercato"][0]
            st.caption(f"Sulle {mt['n_mercato']} partite con quote di chiusura il modello fa {mt['modello_su_mercato'][0]:.4f} contro "
                       f"{mt['mercato'][0]:.4f} del mercato (scarto {gap:+.4f}; più basso è meglio). Se lo scarto è positivo, il mercato prevede meglio.")
        else:
            r3.metric("Log loss mercato", "N.D.")
        st.markdown("#### 💰 Scommesse simulate (1 unità sull'esito con edge più alto)")
        tb = tabella_scommesse(bt)
        if tb.empty:
            st.info("Quote non disponibili per simulare le scommesse.")
        else:
            st.dataframe(tb, use_container_width=True, hide_index=True)
            st.caption("Il ROI su poche centinaia di scommesse è molto rumoroso: guarda l'errore (± ROI) e il CLV. "
                       "Un CLV medio positivo e stabile è un segnale più affidabile del ROI.")
        st.markdown("#### 🎯 Calibrazione: probabilità prevista vs frequenza reale")
        cal = tabella_calibrazione(bt)
        if not cal.empty:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=cal["Prob. prevista %"], y=cal["Frequenza reale %"], mode="lines+markers", name="Modello",
                                      text=cal["Casi"], hovertemplate="Prevista %{x}%<br>Reale %{y}%<br>Casi %{text}"))
            figc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Calibrazione perfetta", line=dict(dash="dash")))
            figc.update_layout(template="plotly_dark", xaxis_title="Probabilità prevista (%)", yaxis_title="Frequenza reale (%)")
            st.plotly_chart(figc, use_container_width=True)
    elif bt is not None:
        st.warning("Backtest vuoto: servono almeno 2 stagioni di dati.")

# --- VALUE BET (KELLY) ---
with tab_kelly:
    st.subheader("🧮 Calcolatore Value Bet & Kelly Criterion")
    c1, c2, c3 = st.columns(3)
    p_stim = c1.slider("Probabilità Modello (%)", 1.0, 100.0, 45.0)
    q_book = c2.number_input("Quota Bookmaker", 1.01, 50.0, 2.30)
    bankroll = c3.number_input("Bankroll (€)", 10.0, 100000.0, 1000.0)
    q_equa = 100 / p_stim
    ev = (p_stim / 100) * q_book - 1
    b, p = q_book - 1, p_stim / 100
    kelly_fraction = max(0.0, (b * p - (1 - p)) / b) * 0.25
    m1, m2, m3 = st.columns(3)
    m1.metric("Quota Equa", f"{q_equa:.2f}")
    m2.metric("Valore Atteso (EV)", f"{ev*100:+.2f}%")
    m3.metric("Stake (Kelly 25%)", f"€{bankroll*kelly_fraction:.2f}")
    if q_book > q_equa:
        st.success("Edge positivo secondo la probabilità inserita (non verificato: dipende da quanto è calibrata).")
    else:
        st.error("❌ Nessun valore rilevato.")

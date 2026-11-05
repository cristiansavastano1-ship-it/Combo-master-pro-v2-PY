import difflib
import io
import json
import time
import math
import re
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

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
VERSIONE_TEST = 9  # cambia quando cambia la struttura dei risultati del test (invalida la cache)
N_STAGIONI_TEST = 8  # il test usa più stagioni del modello per avere più partite su cui misurare
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FINESTRA_GIORNI = 1000  # finestra di addestramento (circa 3 stagioni), come nell'app live
COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
COL_OU_PRESA = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")]
COL_OU_CHIUSURA = [("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5")]
STAT_COLONNE = {"Tiri": ("HS", "AS"), "Tiri in Porta": ("HST", "AST"), "Corner": ("HC", "AC"), "Falli": ("HF", "AF"), "Ammonizioni": ("HY", "AY")}

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


METODO_MARGINE = {"m": "proporzionale"}


def _devig_potenza(pi):
    lo, hi = 1.0, 10.0
    for _ in range(60):
        k = (lo + hi) / 2
        if (pi ** k).sum() > 1:
            lo = k
        else:
            hi = k
    p = pi ** ((lo + hi) / 2)
    return p / p.sum()


def _devig_shin(pi):
    B = pi.sum()

    def somma(z):
        return ((np.sqrt(z * z + 4 * (1 - z) * pi * pi / B) - z) / (2 * (1 - z))).sum()
    if somma(0.0) <= 1:
        return pi / B
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if somma(mid) > 1:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    p = (np.sqrt(z * z + 4 * (1 - z) * pi * pi / B) - z) / (2 * (1 - z))
    return p / p.sum()


def senza_margine(quote, metodo=None):
    """Probabilità implicite delle quote senza il margine del bookmaker. Il metodo (proporzionale, potenza o Shin) lo sceglie il backtest."""
    pi = 1.0 / np.asarray(quote, dtype=float)
    metodo = metodo or METODO_MARGINE["m"]
    if metodo == "potenza":
        return _devig_potenza(pi)
    if metodo == "shin":
        return _devig_shin(pi)
    return pi / pi.sum()


def scegli_metodo_margine(df):
    """Quale metodo per togliere il margine dà le probabilità più accurate? Confronto sul log loss delle quote di apertura (quelle usate
    dall'app) e di chiusura, su tutte le partite con quote. Si cambia dal proporzionale solo con un miglioramento chiaro."""
    metodi = ("proporzionale", "potenza", "shin")
    perdite = {m: {"ap": [], "ch": []} for m in metodi}
    for _, r in df.iterrows():
        es = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
        qa, qc = prima_tripla(r, COL_PRESA), prima_tripla(r, COL_CHIUSURA)
        for m in metodi:
            if qa:
                perdite[m]["ap"].append(-math.log(max(float(senza_margine(qa, m)[es]), 1e-9)))
            if qc:
                perdite[m]["ch"].append(-math.log(max(float(senza_margine(qc, m)[es]), 1e-9)))
    righe = []
    base = np.array(perdite["proporzionale"]["ap"])
    for m in metodi:
        ap, ch = np.array(perdite[m]["ap"]), np.array(perdite[m]["ch"])
        if len(ap) < 100:
            return "proporzionale", pd.DataFrame()
        righe.append({"Metodo": m, "Partite (apertura)": len(ap), "Log loss apertura": round(float(ap.mean()), 4),
                      "Differenza dal proporzionale (± err.)": "" if m == "proporzionale" else _diff_se(ap, base),
                      "Log loss chiusura": round(float(ch.mean()), 4) if len(ch) else None})
    best = min(metodi, key=lambda m: float(np.mean(perdite[m]["ap"])))
    d_best = np.array(perdite[best]["ap"]) - base
    scelto = best if (best != "proporzionale" and d_best.mean() < 0 and abs(d_best.mean()) >= 1.0 * d_best.std(ddof=1) / math.sqrt(len(d_best))) else "proporzionale"
    tab = pd.DataFrame(righe)
    tab["Usato dall'app"] = ["sì" if r["Metodo"] == scelto else "" for r in righe]
    return scelto, tab


def _leggi_csv(url):
    """Ritorna (DataFrame, None) oppure (None, motivo). Gestisce il BOM UTF-8 e riprova (fino a 3 volte) se il server risponde con un errore 5xx o non risponde."""
    ultimo = "errore sconosciuto"
    for tentativo in range(3):
        try:
            res = requests.get(url, timeout=20, headers=UA)
        except Exception as e:
            ultimo = f"connessione fallita ({type(e).__name__})"
        else:
            if res.status_code == 200:
                for enc in ("utf-8-sig", "latin-1"):
                    try:
                        d = pd.read_csv(io.BytesIO(res.content), encoding=enc, on_bad_lines="skip")
                        d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
                        return d, None
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        return None, f"file non leggibile ({type(e).__name__})"
                return None, "codifica non riconosciuta"
            ultimo = f"risposta HTTP {res.status_code}"
            if res.status_code < 500 and res.status_code != 429:
                return None, ultimo
        if tentativo < 2:
            time.sleep(2 + 3 * tentativo)
    return None, ultimo


@st.cache_data(ttl=6 * 3600)
def carica_storico(div, n_stagioni, giorno):
    """Risultati, statistiche e quote delle ultime n stagioni. `giorno` fa scadere la cache ogni giorno."""
    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n_stagioni):
        y = anno - k
        codice = f"{y % 100:02d}{(y + 1) % 100:02d}"
        d, _ = _leggi_csv(f"{BASE_URL}/mmz4281/{codice}/{div}.csv")
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
    """Ritorna (DataFrame, None) oppure (None, motivo)."""
    d, err = _leggi_csv(f"{BASE_URL}/fixtures.csv")
    if d is None:
        return None, err
    if not {"Div", "HomeTeam", "AwayTeam"}.issubset(d.columns):
        return None, "colonne attese non trovate nel file"
    d = d[d["Div"] == div].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
    return d.reset_index(drop=True), None


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
    giocate = {(r["Data"], r["Casa"], r["Trasferta"]) for r in righe}
    if fx is not None:
        for _, r in fx.iterrows():
            d = r["Date"] if "Date" in r and pd.notna(r["Date"]) else pd.Timestamp.max
            if (d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", r["HomeTeam"], r["AwayTeam"]) in giocate:
                continue
            righe.append({"Data": d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "",
                          "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": None, "GolT": None, "Stato": "SCHEDULED", "_d": d})
    righe.sort(key=lambda r: r["_d"])
    righe = assegna_giornate(righe)
    return pd.DataFrame(righe).drop(columns="_d"), "football-data.co.uk (partite giocate + prossimo turno)"


# ======================= MODELLO =======================
def adatta_modello(df, data_rif, ridge, halflife, usa_dc=True, elo_diff=None):
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

    use_elo = elo_diff is not None
    de = np.asarray(elo_diff, float) if use_elo else np.zeros(len(df))

    def f(th):
        mu, hm = th[0], th[1]
        att, dfn = th[2:2 + n], th[2 + n:2 + 2 * n]
        beta = th[2 + 2 * n] if use_elo else 0.0
        eh = mu + hm + att[hi] + dfn[ai] + beta * de
        ea = mu + att[ai] + dfn[hi] - beta * de
        lh, la = np.exp(eh), np.exp(ea)
        nll = -np.sum(w * (x * eh - lh + y * ea - la)) + ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (x - lh), w * (y - la)
        g = np.zeros_like(th)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
        g[2 + n:2 + 2 * n] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * dfn
        if use_elo:
            g[2 + 2 * n] = -np.sum(de * (rh - ra))  # beta non è penalizzato
        return nll, g

    th0 = np.zeros(2 + 2 * n + (1 if use_elo else 0))
    th0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.3))
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    mu, hm, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:2 + 2 * n]
    beta = float(th[2 + 2 * n]) if use_elo else 0.0

    rho = 0.0
    if usa_dc:
        lh = np.exp(mu + hm + att[hi] + dfn[ai] + beta * de)
        la = np.exp(mu + att[ai] + dfn[hi] - beta * de)
        m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

        def neg(r):
            t = np.ones(len(x))
            t[m00] = 1 - lh[m00] * la[m00] * r
            t[m01] = 1 + lh[m01] * r
            t[m10] = 1 + la[m10] * r
            t[m11] = 1 - r
            return -np.sum(w * np.log(np.maximum(t, 1e-6)))

        rho = float(minimize_scalar(neg, bounds=(-0.3, 0.3), method="bounded").x)

    return {"squadre": squadre, "idx": idx, "mu": float(mu), "hm": float(hm), "att": att, "dfn": dfn, "rho": rho, "beta": beta}


def lambda_partita(m, casa, trasf, d_elo=0.0):
    ic, it = m["idx"].get(casa), m["idx"].get(trasf)
    ac, dc = (m["att"][ic], m["dfn"][ic]) if ic is not None else (0.0, 0.0)  # squadra senza storico = media
    at, dt = (m["att"][it], m["dfn"][it]) if it is not None else (0.0, 0.0)
    b = m.get("beta", 0.0) * d_elo
    return math.exp(m["mu"] + m["hm"] + ac + dt + b), math.exp(m["mu"] + at + dc - b)


_K = np.arange(11)
_LOGFACT = gammaln(_K + 1)


def _pmf(lam):
    return np.exp(-lam + _K * np.log(lam) - _LOGFACT)


def matrice_punteggi(lh, la, rho, mg=10):
    M = np.outer(_pmf(lh), _pmf(la))
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


PROB_MIN_DEFAULT = 30  # sotto questa probabilità di mercato le selezioni (sfavorite) rendono molto meno del margine


def migliori_per_partita(df, min_prob):
    """Per ogni partita, la selezione con l'edge più alto tra quelle con probabilità di mercato almeno min_prob."""
    if df is None or df.empty:
        return df
    d = df[df["Prob_Mercato"] >= min_prob]
    if d.empty:
        return d
    return d.loc[d.groupby("Partita")["Edge"].idxmax()].sort_values("Edge", ascending=False)


def tabella_scommesse_fasce(bt, soglia_edge=3.0, prob_min=PROB_MIN_DEFAULT):
    """Scommesse simulate (esito con edge più alto) divise per probabilità di mercato dell'esito scelto, e con il filtro dell'app."""
    ok = bt[["q1", "qx", "q2"]].notna().all(axis=1) & bt[["m1", "mx", "m2"]].notna().all(axis=1)
    b = bt[ok]
    if len(b) < 50:
        return pd.DataFrame()
    P, Q, M = b[["p1", "px", "p2"]].to_numpy(), b[["q1", "qx", "q2"]].to_numpy(), b[["m1", "mx", "m2"]].to_numpy()
    y = b["y"].to_numpy(int)
    PF = np.array([senza_margine(q_) for q_ in Q])
    edge = P * Q - 1
    n = len(b)
    righe = []

    def riga(nome, sel, k):
        if sel.sum() < 5:
            return
        idx = np.where(sel)[0]
        kk = k[idx]
        qk = Q[idx, kk]
        pr = np.where(y[idx] == kk, qk - 1, -1.0)
        clv = qk * M[idx, kk] - 1
        righe.append({"Gruppo": nome, "Scommesse": int(len(idx)), "Vinte %": round(float((y[idx] == kk).mean() * 100), 1),
                      "ROI %": round(float(pr.mean() * 100), 1), "± ROI %": round(float(pr.std(ddof=1) / math.sqrt(len(idx)) * 100), 1),
                      "CLV medio %": round(float(clv.mean() * 100), 1)})
    k0 = edge.argmax(axis=1)
    e0 = edge[np.arange(n), k0]
    pf0 = PF[np.arange(n), k0]
    base = e0 >= soglia_edge / 100
    for nome, lo, hi in [("Esito con prob. di mercato fino a 25%", 0, 0.25), ("25-35%", 0.25, 0.35), ("35-45%", 0.35, 0.45), ("oltre 45%", 0.45, 1.01)]:
        riga(nome, base & (pf0 >= lo) & (pf0 < hi), k0)
    edge_f = np.where(PF >= prob_min / 100, edge, -np.inf)
    k1 = edge_f.argmax(axis=1)
    e1 = edge_f[np.arange(n), k1]
    riga(f"CON IL FILTRO DELL'APP (prob. di mercato ≥ {prob_min}%)", np.isfinite(e1) & (e1 >= soglia_edge / 100), k1)
    return pd.DataFrame(righe)


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


def riassunto_clv(bt, soglia_pct):
    """CLV medio (e suo errore) delle scommesse simulate con edge >= soglia, dal backtest 1X2."""
    ok = bt[["q1", "qx", "q2"]].notna().all(axis=1) & bt[["m1", "mx", "m2"]].notna().all(axis=1)
    b = bt[ok]
    if b.empty:
        return 0, float("nan"), float("nan")
    P, Q, M = b[["p1", "px", "p2"]].to_numpy(), b[["q1", "qx", "q2"]].to_numpy(), b[["m1", "mx", "m2"]].to_numpy()
    edge = P * Q - 1
    k = edge.argmax(axis=1)
    sel = edge[np.arange(len(b)), k] >= soglia_pct / 100
    if sel.sum() < 2:
        return int(sel.sum()), float("nan"), float("nan")
    clv = (Q[np.arange(len(b)), k] * M[np.arange(len(b)), k] - 1)[sel]
    return int(sel.sum()), float(clv.mean() * 100), float(clv.std(ddof=1) / math.sqrt(len(clv)) * 100)


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


def scegli_parametri(df):
    """Sceglie shrinkage ed emivita con una griglia piccola, valutata col backtest walk-forward (log loss 1X2)
    usando SOLO i dati passati in df."""
    default = {"ridge": 2.0, "halflife": 365, "log_loss": None, "ricerca": [], "auto": False}
    if df is None or df["Stag"].nunique() < 2:
        return default
    ricerca = []
    for r_ in (0.5, 2.0, 6.0):
        for h_ in (180, 365, 730, 0):
            bt = esegui_backtest_df(df, r_, h_, True, 28)
            if not bt.empty:
                ricerca.append({"Shrinkage λ": r_, "Emivita (gg)": str(h_) if h_ else "nessuna",
                                "Log loss": round(metriche_backtest(bt)["modello"][0], 4), "_r": r_, "_h": h_})
    if not ricerca:
        return default
    best = min(ricerca, key=lambda z: z["Log loss"])
    return {"ridge": best["_r"], "halflife": best["_h"], "log_loss": best["Log loss"], "auto": True,
            "ricerca": [{k: v for k, v in z.items() if not k.startswith("_")} for z in ricerca]}


@st.cache_data(ttl=6 * 3600, show_spinner="Ottimizzo i parametri del modello (solo la prima volta di giornata)...")
def parametri_auto(div, giorno):
    return scegli_parametri(carica_storico(div, N_STAGIONI, giorno))


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


# ======================= TEST COMPLETO (GOAL, OVER/UNDER, RISULTATI ESATTI) =======================
_I, _J = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
_MASK = {"p1": _I > _J, "px": _I == _J, "p2": _I < _J, "over1.5": (_I + _J) > 1.5, "over2.5": (_I + _J) > 2.5,
         "over3.5": (_I + _J) > 3.5, "btts": (_I > 0) & (_J > 0)}
VARIANTI_TIRI = ["Gol + Tiri 50/50", "Tiri in porta"]
VARIANTI_MODELLO = ["Modello", "Senza Dixon-Coles", "Ancorato al mercato", "Misto 50/50"] + VARIANTI_TIRI
VARIANTI = VARIANTI_MODELLO + ["Frequenze storiche"]


def prima_coppia(riga, colonne):
    for a, b in colonne:
        try:
            v = [float(riga[a]), float(riga[b])]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v):
            return v
    return None


def matrice_ancorata(p1x2, p_over25, rho, start):
    """Gol attesi ricavati dalle quote (1X2 ed eventualmente Over/Under 2.5), poi distribuzione completa dei punteggi."""
    def err(th):
        mk = mercati(matrice_punteggi(math.exp(th[0]), math.exp(th[1]), rho))
        e = (mk["p1"] - p1x2[0]) ** 2 + (mk["px"] - p1x2[1]) ** 2 + (mk["p2"] - p1x2[2]) ** 2
        if p_over25 is not None:
            e += (mk["over2.5"] - p_over25) ** 2
        return e
    res = minimize(err, np.log(np.asarray(start, float)), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 250})
    return matrice_punteggi(math.exp(res.x[0]), math.exp(res.x[1]), rho)


def esegui_test_completo(df, ridge, halflife, passo, solo_base=False, inizio=None, leggero=False):
    """Walk-forward come il backtest, ma conserva l'intera distribuzione dei punteggi di ogni variante.
    Con solo_base=True calcola solo Modello e Frequenze storiche (più veloce: serve a scegliere i pesi)."""
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return None
    t = inizio if inizio is not None else df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    nan_m = np.full((11, 11), np.nan)
    rec = {"Partita": [], "xs": [], "ys": [], "ok": [], "ou_ok": [], "m1x2": [], "mou": [], "min_storia": [], "tiri_ok": [], "date": [], "q": [], "ht": [], "split": []}
    Ms = {v: [] for v in VARIANTI}
    ha_tiri = {"HST", "AST"}.issubset(df.columns)
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=FINESTRA_GIORNI))]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, True)
            cnt = pd.concat([train["HomeTeam"], train["AwayTeam"]]).value_counts()
            m_freq = matrice_frequenze(train)
            S_blocco = tabella_split(train)
            m_t, conv = None, None
            if ha_tiri and not solo_base:
                tr_t = train.dropna(subset=["HST", "AST"])
                if len(tr_t) >= 150 and tr_t["HST"].sum() > 0 and tr_t["AST"].sum() > 0:
                    d_t = tr_t.copy()
                    d_t["FTHG"], d_t["FTAG"] = tr_t["HST"].astype(int), tr_t["AST"].astype(int)
                    scala = (tr_t["HST"] + tr_t["AST"]).mean() / max((tr_t["FTHG"] + tr_t["FTAG"]).mean(), 1e-9)
                    m_t = adatta_modello(d_t, t, ridge * scala, halflife, False)  # shrinkage rapportata alla scala dei conteggi
                    conv = (tr_t["FTHG"].sum() / tr_t["HST"].sum(), tr_t["FTAG"].sum() / tr_t["AST"].sum())  # gol per tiro in porta
            for _, r in blocco.iterrows():
                h, a = r["HomeTeam"], r["AwayTeam"]
                lh, la = lambda_partita(m, h, a)
                Mb = matrice_punteggi(lh, la, m["rho"])
                Mgt = Mt = nan_m
                tiri_ok = False
                if m_t is not None:
                    lth, lta = lambda_partita(m_t, h, a)
                    ls_h, ls_a = lth * conv[0], lta * conv[1]
                    Mt = matrice_punteggi(ls_h, ls_a, m["rho"])
                    Mgt = matrice_punteggi(math.sqrt(lh * ls_h), math.sqrt(la * ls_a), m["rho"])
                    tiri_ok = True
                if solo_base:
                    Mn = Ma = Mx = nan_m
                    qp = None
                    qc, qoc = prima_tripla(r, COL_CHIUSURA), prima_coppia(r, COL_OU_CHIUSURA)
                else:
                    Mn = nan_m if leggero else matrice_punteggi(lh, la, 0.0)
                    qp, qo = prima_tripla(r, COL_PRESA), prima_coppia(r, COL_OU_PRESA)
                    if qp:
                        p_o = float(senza_margine(qo)[0]) if qo else None
                        Ma = matrice_ancorata(senza_margine(qp), p_o, m["rho"], (lh, la))
                        Mx = 0.5 * (Mb + Ma)
                    else:
                        Ma = Mx = nan_m
                    qc, qoc = prima_tripla(r, COL_CHIUSURA), prima_coppia(r, COL_OU_CHIUSURA)
                rec["Partita"].append(f"{h} - {a}")
                rec["date"].append(r["Date"])
                rec["xs"].append(min(int(r["FTHG"]), 10))
                rec["ys"].append(min(int(r["FTAG"]), 10))
                rec["q"].append(np.array(qp, float) if qp else np.full(3, np.nan))
                rec["split"].append(S_blocco)
                rec["ht"].append((float(r["HTHG"]), float(r["HTAG"])) if ("HTHG" in r.index and pd.notna(r["HTHG"]) and pd.notna(r["HTAG"])) else (np.nan, np.nan))
                rec["ok"].append(bool(qp))
                rec["ou_ok"].append(bool(qp and qoc))
                rec["m1x2"].append(senza_margine(qc) if qc else np.full(3, np.nan))
                rec["mou"].append(float(senza_margine(qoc)[0]) if qoc else np.nan)
                rec["min_storia"].append(int(min(cnt.get(h, 0), cnt.get(a, 0))))
                rec["tiri_ok"].append(tiri_ok)
                for nome, Mv in {"Modello": Mb, "Senza Dixon-Coles": Mn, "Ancorato al mercato": Ma, "Misto 50/50": Mx,
                                 "Gol + Tiri 50/50": Mgt, "Tiri in porta": Mt, "Frequenze storiche": m_freq}.items():
                    Ms[nome].append(Mv)
        t += pd.Timedelta(days=passo)
    if not rec["xs"]:
        return None
    out = {k: np.array(v) for k, v in rec.items() if k != "Partita"}
    out["Partita"] = rec["Partita"]
    out["Ms"] = {k: np.array(v) for k, v in Ms.items()}
    return out


def matrice_frequenze(df):
    """Distribuzione empirica dei punteggi del campionato (modello banale), lievemente lisciata."""
    cm = np.zeros((11, 11))
    np.add.at(cm, (np.minimum(df["FTHG"].to_numpy(), 10), np.minimum(df["FTAG"].to_numpy(), 10)), 1)
    return (cm + 0.05) / (cm.sum() + 0.05 * 121)


def _p_mercato(Ms, chiave):
    return (Ms * _MASK[chiave]).sum(axis=(1, 2))


def _ll_bin(p, o):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def _ll_1x2(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def _vettore_perdita(Mv, chiave, xs, ys, outc, esito):
    if chiave == "1x2":
        P = np.stack([_p_mercato(Mv, "p1"), _p_mercato(Mv, "px"), _p_mercato(Mv, "p2")], axis=1)
        return -np.log(np.clip(P[np.arange(len(esito)), esito], 1e-9, 1))
    if chiave == "exact":
        return -np.log(np.clip(Mv[np.arange(len(Mv)), xs, ys], 1e-12, 1))
    p = np.clip(_p_mercato(Mv, chiave), 1e-9, 1 - 1e-9)
    o = outc[chiave]
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def _diff_se(a, b):
    d = a - b
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.4f} ± {se:.4f}"


def pesi_ottimi(res, nome_gol="Modello", nome_es="Modello"):
    """Peso del modello (resto: frequenze storiche) che minimizza l'errore: uno per i mercati gol, uno per i risultati esatti."""
    xs, ys = res["xs"], res["ys"]
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {"over1.5": (xs + ys > 1.5).astype(float), "over2.5": (xs + ys > 2.5).astype(float),
            "over3.5": (xs + ys > 3.5).astype(float), "btts": ((xs > 0) & (ys > 0)).astype(float)}
    Mb_g, Mb_e, Mf = res["Ms"][nome_gol], res["Ms"][nome_es], res["Ms"]["Frequenze storiche"]
    gol, esatto = {}, {}
    for w in W_GRID:
        Mw = w * Mb_g + (1 - w) * Mf
        gol[w] = float(np.mean([_vettore_perdita(Mw, k_, xs, ys, outc, esito).mean() for k_ in ("over1.5", "over2.5", "over3.5", "btts")]))
        esatto[w] = float(_vettore_perdita(w * Mb_e + (1 - w) * Mf, "exact", xs, ys, outc, esito).mean())
    return {"gol": min(gol, key=gol.get), "esatto": min(esatto, key=esatto.get), "n": len(xs)}


def tabella_roi(res, ok_mask, esito):
    """Scommettere sempre un esito a una certa fascia di probabilità di mercato: frequenza reale e rendimento, con errore."""
    if "q" not in res:
        return pd.DataFrame()
    Q = res["q"]
    sel = ok_mask & ~np.isnan(Q).any(axis=1)
    if sel.sum() < 50:
        return pd.DataFrame()
    Qs, es = Q[sel], esito[sel]
    pf = np.array([senza_margine(q_) for q_ in Qs])
    righe = []
    for i, nome in enumerate(["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]):
        win = es == i
        profitto = np.where(win, Qs[:, i] - 1, -1.0)
        for etich, lo, hi in [("Tutte", 0, 1.01), ("fino a 25%", 0, 0.25), ("25-35%", 0.25, 0.35), ("35-45%", 0.35, 0.45),
                              ("45-55%", 0.45, 0.55), ("oltre 55%", 0.55, 1.01)]:
            m_ = (pf[:, i] >= lo) & (pf[:, i] < hi)
            if m_.sum() >= 30:
                pr_ = profitto[m_]
                righe.append({"Esito": nome, "Fascia prob. mercato": etich, "Partite": int(m_.sum()), "Prob. mercato %": round(float(pf[m_, i].mean() * 100), 1),
                              "Reale %": round(float(win[m_].mean() * 100), 1), "ROI %": round(float(pr_.mean() * 100), 1),
                              "± ROI %": round(float(pr_.std(ddof=1) / math.sqrt(m_.sum()) * 100), 1)})
    return pd.DataFrame(righe)


def _base_app(res, pesi, chiave):
    """Matrici del modello come le usa l'app senza quote (con i tiri in porta dove il backtest li ha giustificati)."""
    usa = bool(pesi and pesi.get(chiave) and "tiri_ok" in res and res["tiri_ok"].mean() >= 0.5)
    if usa:
        return np.where(res["tiri_ok"][:, None, None], res["Ms"]["Gol + Tiri 50/50"], res["Ms"]["Modello"])
    return res["Ms"]["Modello"]


def tabella_tempi_test(res, ok_mask, pesi=None):
    """Primo e secondo tempo: probabilità previste contro frequenze reali (media), per le due fonti dell'app."""
    if "ht" not in res or "split" not in res:
        return pd.DataFrame()
    ht = res["ht"]
    ha_ht = ~np.isnan(ht).any(axis=1)
    if ha_ht.sum() < 100:
        return pd.DataFrame()
    xs, ys = res["xs"], res["ys"]
    h1 = ht[:, 0] + ht[:, 1]
    h2 = (xs - ht[:, 0]) + (ys - ht[:, 1])
    Mb, Mf, Ma = _base_app(res, pesi, "tiri_gol"), res["Ms"]["Frequenze storiche"], res["Ms"]["Ancorato al mercato"]
    w = pesi["gol"] if pesi else 0.5
    mercati_t = [("1° tempo almeno 1 gol", lambda p1, p2, pm: 1 - p1[0], lambda a, b: a >= 1),
                 ("1° tempo almeno 2 gol", lambda p1, p2, pm: 1 - p1[0] - p1[1], lambda a, b: a >= 2),
                 ("2° tempo almeno 1 gol", lambda p1, p2, pm: 1 - p2[0], lambda a, b: b >= 1),
                 ("2° tempo almeno 2 gol", lambda p1, p2, pm: 1 - p2[0] - p2[1], lambda a, b: b >= 2),
                 ("1° tempo 1-2 gol", lambda p1, p2, pm: p1[1:3].sum(), lambda a, b: (a >= 1) & (a <= 2)),
                 ("2° tempo 1-2 gol", lambda p1, p2, pm: p2[1:3].sum(), lambda a, b: (b >= 1) & (b <= 2)),
                 ("Primo tempo più gol", lambda p1, p2, pm: np.tril(pm, -1).sum(), lambda a, b: a > b),
                 ("Secondo tempo più gol", lambda p1, p2, pm: np.triu(pm, 1).sum(), lambda a, b: a < b)]
    righe = []
    for nome_f, sel, costruisci in (
            ("APP senza quote", ha_ht, lambda i: w * Mb[i] + (1 - w) * Mf[i]),
            ("APP con quote", ha_ht & ok_mask, lambda i: Ma[i])):
        idx = np.where(sel)[0]
        if len(idx) < 50:
            continue
        pred = {m[0]: [] for m in mercati_t}
        for i in idx:
            p1, p2, pm = probs_tempi(costruisci(i), res["split"][i])
            for nome, fp, _ in mercati_t:
                pred[nome].append(float(fp(p1, p2, pm)))
        for nome, _, fa in mercati_t:
            pr_, re_ = np.array(pred[nome]), fa(h1[idx], h2[idx]).astype(float)
            righe.append({"Mercato": nome, "Fonte": nome_f, "Partite": len(idx), "Prevista %": round(float(pr_.mean() * 100), 1),
                          "Reale %": round(float(re_.mean() * 100), 1), "Differenza (punti)": round(float((pr_.mean() - re_.mean()) * 100), 1)})
    return pd.DataFrame(righe)


def tabelle_test(res, pesi=None):
    # Le varianti con i tiri si confrontano solo se i tiri in porta sono disponibili per (quasi) tutte le partite
    usa_tiri = res.get("tiri_ok") is not None and bool(res["tiri_ok"].mean() >= 0.5)
    VARIANTI_A = VARIANTI if usa_tiri else [v for v in VARIANTI if v not in VARIANTI_TIRI]
    VARIANTI_MOD_A = [v for v in VARIANTI_MODELLO if v in VARIANTI_A]
    xs, ys = res["xs"], res["ys"]
    ok = res["ok"] & res["tiri_ok"] if usa_tiri else res["ok"]
    ou_ok = res["ou_ok"] & res["tiri_ok"] if usa_tiri else res["ou_ok"]
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {"over1.5": (xs + ys > 1.5).astype(float), "over2.5": (xs + ys > 2.5).astype(float),
            "over3.5": (xs + ys > 3.5).astype(float), "btts": ((xs > 0) & (ys > 0)).astype(float)}
    mercati_def = [("1X2", "1x2"), ("Over/Under 1.5", "over1.5"), ("Over/Under 2.5", "over2.5"),
                   ("Over/Under 3.5", "over3.5"), ("Goal / No Goal", "btts"), ("Risultato esatto (log score)", "exact")]
    righe, migliore, peso_mk = [], {}, {}
    for etichetta, mk in mercati_def:
        sel = ok.copy()
        if mk == "over2.5":
            sel = ou_ok.copy()
        if mk == "1x2":
            sel = ok & ~np.isnan(res["m1x2"]).any(axis=1)
        if sel.sum() < 10:
            continue
        xs_s, ys_s = xs[sel], ys[sel]
        outc_s = {k_: v[sel] for k_, v in outc.items()}
        es_s = esito[sel]
        L = {v: _vettore_perdita(res["Ms"][v][sel], mk, xs_s, ys_s, outc_s, es_s) for v in VARIANTI_A}
        r = {"Mercato": etichetta, "Partite": int(sel.sum())}
        for v in VARIANTI_A:
            r[v] = round(float(L[v].mean()), 4)
        Mb, Mf = res["Ms"]["Modello"][sel], res["Ms"]["Frequenze storiche"][sel]
        lw = {w: float(_vettore_perdita(w * Mb + (1 - w) * Mf, mk, xs_s, ys_s, outc_s, es_s).mean()) for w in W_GRID}
        w_best = min(lw, key=lw.get)
        r["Modello+Frequenze"] = round(lw[w_best], 4)
        r["Peso modello %"] = int(w_best * 100)
        if mk == "1x2":
            r["Mercato (chiusura)"] = round(float(-np.mean(np.log(np.clip(res["m1x2"][sel][np.arange(sel.sum()), es_s], 1e-9, 1)))), 4)
        elif mk == "over2.5":
            r["Mercato (chiusura)"] = round(_ll_bin(res["mou"][sel], outc_s["over2.5"]), 4)
        else:
            r["Mercato (chiusura)"] = None
        r["Modello − Frequenze (± err.)"] = _diff_se(L["Modello"], L["Frequenze storiche"])
        r["Ancorato − Modello (± err.)"] = _diff_se(L["Ancorato al mercato"], L["Modello"])
        L_mix = _vettore_perdita(w_best * Mb + (1 - w_best) * Mf, mk, xs_s, ys_s, outc_s, es_s)
        r["Modello+Freq − Frequenze (± err.)"] = _diff_se(L_mix, L["Frequenze storiche"])
        if usa_tiri:
            r["Gol+Tiri − Modello (± err.)"] = _diff_se(L["Gol + Tiri 50/50"], L["Modello"])
            r["Tiri − Modello (± err.)"] = _diff_se(L["Tiri in porta"], L["Modello"])
        best = min(VARIANTI_MOD_A, key=lambda v: r[v])
        r["Migliore"] = best
        migliore[mk] = best
        peso_mk[mk] = w_best
        righe.append(r)
    tab_a = pd.DataFrame(righe)

    # Presi e non presi per soglia di fiducia
    righe_b = []
    sel = ok
    for etichetta, mk, lato in [("Goal", "btts", 1), ("No Goal", "btts", 0), ("Over 1.5", "over1.5", 1), ("Under 1.5", "over1.5", 0),
                                ("Over 2.5", "over2.5", 1), ("Under 2.5", "over2.5", 0), ("Over 3.5", "over3.5", 1), ("Under 3.5", "over3.5", 0)]:
        v = migliore.get(mk, "Modello")
        p = _p_mercato(res["Ms"][v][sel], mk)
        p = p if lato == 1 else 1 - p
        hit = (outc[mk][sel] == lato)
        for thr in (55, 65, 75):
            s_ = p >= thr / 100
            if s_.sum() >= 5:
                righe_b.append({"Mercato": etichetta, "Soglia %": thr, "Variante": v, "Selezioni": int(s_.sum()),
                                "Presi %": round(hit[s_].mean() * 100, 1), "Prob. dichiarata %": round(p[s_].mean() * 100, 1)})
    v1 = migliore.get("1x2", "Modello")
    P1 = np.stack([_p_mercato(res["Ms"][v1][sel], k_) for k_ in ("p1", "px", "p2")], axis=1)
    for i, nome in enumerate(["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]):
        for thr in (45, 55, 65):
            s_ = P1[:, i] >= thr / 100
            if s_.sum() >= 5:
                righe_b.append({"Mercato": nome, "Soglia %": thr, "Variante": v1, "Selezioni": int(s_.sum()),
                                "Presi %": round((esito[sel][s_] == i).mean() * 100, 1), "Prob. dichiarata %": round(P1[s_, i].mean() * 100, 1)})
    tab_b = pd.DataFrame(righe_b)

    # Risultati esatti: presi e non presi
    righe_c = []
    for v in VARIANTI_A:
        Mv = res["Ms"][v][ok].reshape(ok.sum(), -1)
        ordine = np.argsort(-Mv, axis=1)[:, :3]
        reale = xs[ok] * 11 + ys[ok]
        righe_c.append({"Variante": v, "Log score": round(float(-np.mean(np.log(np.clip(Mv[np.arange(len(Mv)), reale], 1e-12, 1)))), 4),
                        "1° esatto preso %": round((ordine[:, 0] == reale).mean() * 100, 1),
                        "Nei primi 3 %": round((ordine == reale[:, None]).any(axis=1).mean() * 100, 1),
                        "Prob. media del 1° %": round(float(np.take_along_axis(Mv, ordine[:, :1], 1).mean() * 100), 1)})
    tab_c = pd.DataFrame(righe_c)
    best_ex = min(VARIANTI_MOD_A, key=lambda v: float(tab_c.loc[tab_c["Variante"] == v, "Log score"].iloc[0]))
    idx = np.where(ok)[0][-25:]
    Mv = res["Ms"][best_ex]
    righe_d = []
    for i in idx[::-1]:
        flat = Mv[i].ravel()
        top = np.argsort(-flat)[:3]
        reale = xs[i] * 11 + ys[i]
        stato = "✅ preso (1°)" if top[0] == reale else ("🟡 nei primi 3" if reale in top else "❌ non preso")
        righe_d.append({"Partita": res["Partita"][i],
                        "1°": f"{top[0]//11}-{top[0]%11} ({flat[top[0]]*100:.0f}%)", "2°": f"{top[1]//11}-{top[1]%11} ({flat[top[1]]*100:.0f}%)",
                        "3°": f"{top[2]//11}-{top[2]%11} ({flat[top[2]]*100:.0f}%)",
                        "Risultato": f"{xs[i]}-{ys[i]}", "Esito": stato})
    tab_d = pd.DataFrame(righe_d)

    # Squadre con poca storia (neopromosse): Modello, Ancorato e mercato sulle stesse partite
    righe_e = []
    base_ok = ok & ~np.isnan(res["m1x2"]).any(axis=1)
    for nome, gruppo in [("Squadra con meno di 15 partite di storico", res["min_storia"] < 15),
                         ("Tutte le altre", res["min_storia"] >= 15)]:
        sel_e = gruppo & base_ok
        if sel_e.sum() >= 5:
            es_e = esito[sel_e]
            riga = {"Gruppo": nome, "Partite": int(sel_e.sum())}
            for v in ("Modello", "Ancorato al mercato"):
                riga[f"Log loss 1X2 ({v})"] = round(float(_vettore_perdita(res["Ms"][v][sel_e], "1x2", xs[sel_e], ys[sel_e], {}, es_e).mean()), 4)
            riga["Log loss 1X2 (Mercato chiusura)"] = round(float(-np.mean(np.log(np.clip(res["m1x2"][sel_e][np.arange(sel_e.sum()), es_e], 1e-9, 1)))), 4)
            righe_e.append(riga)
    # Calibrazione di Goal e Over 2.5: previsto vs reale per fascia, per le tre fonti usate dall'app
    righe_cal = []
    for nome_m, mk in (("Goal", "btts"), ("Over 2.5", "over2.5")):
        sel_c = ou_ok if mk == "over2.5" else ok
        if sel_c.sum() < 30 or mk not in peso_mk:
            continue
        Mb_c, Mf_c = res["Ms"]["Modello"][sel_c], res["Ms"]["Frequenze storiche"][sel_c]
        w_c = peso_mk[mk]
        fonti = {"Modello da solo": Mb_c, f"Modello+Frequenze (peso modello {int(w_c * 100)}%)": w_c * Mb_c + (1 - w_c) * Mf_c,
                 "Ancorato al mercato": res["Ms"]["Ancorato al mercato"][sel_c]}
        if pesi is not None:
            w_l = pesi["gol"]
            Mb_app = _base_app(res, pesi, "tiri_gol")[sel_c]
            fonti[f"APP senza quote (peso modello {int(w_l * 100)}%)"] = w_l * Mb_app + (1 - w_l) * Mf_c
        o_c = outc[mk][sel_c]
        for nome_f, Mv in fonti.items():
            pm = _p_mercato(Mv, mk)
            righe_cal.append({"Mercato": nome_m, "Fonte": nome_f, "Fascia": "TUTTE", "Partite": len(pm),
                              "Prevista %": round(float(pm.mean() * 100), 1), "Reale %": round(float(o_c.mean() * 100), 1),
                              "Differenza (punti)": round(float((pm.mean() - o_c.mean()) * 100), 1)})
            for lo, hi in [(0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)]:
                m_ = (pm >= lo) & (pm < hi)
                if m_.sum() >= 15:
                    righe_cal.append({"Mercato": nome_m, "Fonte": nome_f, "Fascia": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", "Partite": int(m_.sum()),
                                      "Prevista %": round(float(pm[m_].mean() * 100), 1), "Reale %": round(float(o_c[m_].mean() * 100), 1),
                                      "Differenza (punti)": round(float((pm[m_].mean() - o_c[m_].mean()) * 100), 1)})
    return tab_a, tab_b, tab_c, tab_d, pd.DataFrame(righe_e), best_ex, pd.DataFrame(righe_cal), tabella_roi(res, ok, esito), tabella_tempi_test(res, ok, pesi)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def test_completo_cached(div, ridge, halflife, passo, giorno, versione):
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    return None if df is None else esegui_test_completo(df, ridge, halflife, passo)


def _campioni(M1x2, Mgol, esito, outc):
    """Coppie (probabilità prevista, è successo) per ogni esito/mercato mostrato dall'app, raggruppate per tipo."""
    p1, px, p2 = (_p_mercato(M1x2, k) for k in ("p1", "px", "p2"))
    dc = np.stack([p1 + px, px + p2, p1 + p2], axis=1)
    hit_dc = np.stack([esito <= 1, esito >= 1, esito != 1], axis=1).astype(float)
    pg = _p_mercato(Mgol, "btts")
    pred_ou, hit_ou = [], []
    for mk in ("over1.5", "over2.5", "over3.5"):
        po = _p_mercato(Mgol, mk)
        pred_ou += [po, 1 - po]
        hit_ou += [outc[mk], 1 - outc[mk]]
    return {"1x2": (np.stack([p1, px, p2], axis=1).ravel(), np.eye(3)[esito].ravel()),
            "doppia": (dc.ravel(), hit_dc.ravel()),
            "goal": (np.concatenate([pg, 1 - pg]), np.concatenate([outc["btts"], 1 - outc["btts"]])),
            "ou": (np.concatenate(pred_ou), np.concatenate(hit_ou))}


def costruisci_curve(res, w_gol, nome_gol="Modello"):
    """Affidabilità storica delle percentuali nelle due situazioni dell'app: senza quote e con quote."""
    esito, outc = _esiti(res["xs"], res["ys"])
    Mb, Mf, Ma = res["Ms"]["Modello"], res["Ms"]["Frequenze storiche"], res["Ms"]["Ancorato al mercato"]
    Mw = w_gol * res["Ms"][nome_gol] + (1 - w_gol) * Mf
    out = {"senza": _campioni(Mb, Mw, esito, outc)}
    sel = res["ou_ok"]
    if sel.sum() >= 100:
        out["con"] = _campioni(Ma[sel], Ma[sel], esito[sel], {k: v[sel] for k, v in outc.items()})
    return out


def cerca_fiducia(campioni, p, n_min=40):
    """Tra i casi storici con probabilità prevista vicina a p, quanti sono davvero usciti. Ritorna (frequenza, casi, previsto medio) o None."""
    if campioni is None:
        return None
    pred, hit = campioni
    for w_ in (0.02, 0.03, 0.05, 0.08):
        m = np.abs(pred - p) <= w_
        if m.sum() >= n_min:
            return float(hit[m].mean()), int(m.sum()), float(pred[m].mean())
    return None


def tabella_affidabilita(curve):
    righe = []
    nomi = {"1x2": "1X2 (ogni esito)", "doppia": "Doppia chance", "goal": "Goal / No Goal", "ou": "Over/Under (1.5, 2.5, 3.5)"}
    for fonte, nome_f in (("senza", "Senza quote"), ("con", "Con quote")):
        if fonte not in curve:
            continue
        for k, nome in nomi.items():
            pred, hit = curve[fonte][k]
            for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
                m = (pred >= lo) & (pred < hi)
                if m.sum() >= 30:
                    righe.append({"Situazione": nome_f, "Mercato": nome, "Fascia prevista": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", "Casi": int(m.sum()),
                                  "Prevista %": round(float(pred[m].mean() * 100), 1), "Uscita davvero %": round(float(hit[m].mean() * 100), 1)})
    return pd.DataFrame(righe)


def decidi_tiri(res):
    """I tiri in porta migliorano il modello in questo campionato? Si usano solo se, sul backtest, il miglioramento supera 1.5 volte l'errore."""
    out = {"gol": False, "esatto": False, "tab": pd.DataFrame()}
    if "tiri_ok" not in res or res["tiri_ok"].sum() < 300:
        return out
    sel = res["tiri_ok"]
    xs, ys = res["xs"][sel], res["ys"][sel]
    esito, outc = _esiti(xs, ys)
    Mm, Mt = res["Ms"]["Modello"][sel], res["Ms"]["Gol + Tiri 50/50"][sel]
    d_gol = np.mean([_vettore_perdita(Mt, k_, xs, ys, outc, esito) - _vettore_perdita(Mm, k_, xs, ys, outc, esito)
                     for k_ in ("over1.5", "over2.5", "over3.5", "btts")], axis=0)
    d_es = _vettore_perdita(Mt, "exact", xs, ys, outc, esito) - _vettore_perdita(Mm, "exact", xs, ys, outc, esito)
    righe = []
    for nome, d_, chiave in (("Goal, Over/Under (media dei 4 mercati)", d_gol, "gol"), ("Risultato esatto", d_es, "esatto")):
        se = float(d_.std(ddof=1) / math.sqrt(len(d_)))
        out[chiave] = bool(d_.mean() < 0 and -d_.mean() >= 1.5 * se)
        righe.append({"Mercato": nome, "Partite": int(len(d_)), "Gol+Tiri − Modello (± err.)": f"{d_.mean():+.4f} ± {se:.4f}",
                      "Usato dall'app": "sì" if out[chiave] else "no"})
    out["tab"] = pd.DataFrame(righe)
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def modello_tiri_corrente(div, ridge, halflife, giorno):
    """Modello degli attacchi/difese stimato sui tiri in porta, per le previsioni dal vivo (se i tiri sono disponibili)."""
    df = carica_storico(div, N_STAGIONI, giorno)
    if df is None or not {"HST", "AST"}.issubset(df.columns):
        return None
    tr_t = df.dropna(subset=["HST", "AST"])
    if len(tr_t) < 150 or tr_t["HST"].sum() <= 0 or tr_t["AST"].sum() <= 0:
        return None
    d_t = tr_t.copy()
    d_t["FTHG"], d_t["FTAG"] = tr_t["HST"].astype(int), tr_t["AST"].astype(int)
    scala = (tr_t["HST"] + tr_t["AST"]).mean() / max((tr_t["FTHG"] + tr_t["FTAG"]).mean(), 1e-9)
    m_t = adatta_modello(d_t, pd.Timestamp.now().normalize(), ridge * scala, halflife, False)
    conv = (tr_t["FTHG"].sum() / tr_t["HST"].sum(), tr_t["FTAG"].sum() / tr_t["AST"].sum())
    return {"modello": m_t, "conv": conv}


@st.cache_data(ttl=6 * 3600, show_spinner="Calcolo pesi e affidabilità dal backtest (solo la prima volta di giornata)...")
def pesi_blend_auto(div, ridge, halflife, giorno):
    """Dal backtest: metodo per togliere il margine dalle quote, peso del modello senza quote (resto: frequenze storiche),
    soglia per la X e affidabilità storica delle percentuali."""
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    metodo, tab_margine = scegli_metodo_margine(df) if df is not None else ("proporzionale", pd.DataFrame())
    METODO_MARGINE["m"] = metodo
    res = esegui_test_completo(df, ridge, halflife, 28, leggero=True) if df is not None else None
    if res is None or len(res["xs"]) < 100:
        return {"gol": 0.5, "esatto": 0.75, "n": 0, "auto": False, "soglia_x": {"modello": None, "ancorato": None}, "curve": {},
                "metodo_margine": metodo, "tab_margine": tab_margine, "tiri_gol": False, "tiri_esatto": False, "tab_tiri": pd.DataFrame(),
                "nome_gol": "Modello", "nome_esatto": "Modello"}
    dt = decidi_tiri(res)
    res["Ms"]["Modello+Tiri"] = np.where(res["tiri_ok"][:, None, None], res["Ms"]["Gol + Tiri 50/50"], res["Ms"]["Modello"])
    nome_g = "Modello+Tiri" if dt["gol"] else "Modello"
    nome_e = "Modello+Tiri" if dt["esatto"] else "Modello"
    w = pesi_ottimi(res, nome_g, nome_e)
    w["auto"] = True
    w["tiri_gol"], w["tiri_esatto"], w["tab_tiri"] = dt["gol"], dt["esatto"], dt["tab"]
    w["nome_gol"], w["nome_esatto"] = nome_g, nome_e
    w["metodo_margine"], w["tab_margine"] = metodo, tab_margine
    es_, _ = _esiti(res["xs"], res["ys"])
    P_mod = np.stack([_p_mercato(res["Ms"]["Modello"], k_) for k_ in ("p1", "px", "p2")], axis=1)
    sx = {"modello": trova_soglia_x(P_mod, es_), "ancorato": None}
    if res["ok"].sum() >= 150:
        P_anc = np.stack([_p_mercato(res["Ms"]["Ancorato al mercato"][res["ok"]], k_) for k_ in ("p1", "px", "p2")], axis=1)
        sx["ancorato"] = trova_soglia_x(P_anc, es_[res["ok"]])
    w["soglia_x"] = sx
    w["curve"] = costruisci_curve(res, w["gol"], nome_g)
    return w


# ======================= TEST SU STAGIONI MAI VISTE =======================
MERCATI_GOL = [("Over/Under 1.5", "over1.5"), ("Over/Under 2.5", "over2.5"), ("Over/Under 3.5", "over3.5"),
               ("Goal / No Goal", "btts"), ("Risultato esatto (log score)", "exact")]


def _esiti(xs, ys):
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {"over1.5": (xs + ys > 1.5).astype(float), "over2.5": (xs + ys > 2.5).astype(float),
            "over3.5": (xs + ys > 3.5).astype(float), "btts": ((xs > 0) & (ys > 0)).astype(float)}
    return esito, outc


def test_stagioni_nuove(df, ridge_att, halflife_att, passo=14):
    """Per ogni stagione (dalla terza in poi) parametri e pesi sono scelti usando SOLO le stagioni precedenti;
    poi si misura la qualità sulla stagione stessa, mai vista in fase di scelta."""
    stagioni = sorted(df["Stag"].unique())
    righe, tutte = [], {"new": [], "att": [], "mkt": [], "freq": []}
    perd = {mk: {"Modello": [], "Frequenze": [], "Misto": []} for _, mk in MERCATI_GOL}
    pesi_usati = {"gol": [], "esatto": []}
    for k in stagioni[2:]:
        t0 = df.loc[df["Stag"] == k, "Date"].min()
        prima = df[df["Date"] < t0]
        pk = scegli_parametri(prima)
        res_prev = esegui_test_completo(prima, pk["ridge"], pk["halflife"], 28, solo_base=True)
        if res_prev is None or len(res_prev["xs"]) < 100:
            continue
        w = pesi_ottimi(res_prev)
        dk = df[df["Stag"] <= k]
        rn = esegui_test_completo(dk, pk["ridge"], pk["halflife"], passo, solo_base=True, inizio=t0)
        ra = esegui_test_completo(dk, ridge_att, halflife_att, passo, solo_base=True, inizio=t0)
        if rn is None or len(rn["xs"]) < 20:
            continue
        xs, ys = rn["xs"], rn["ys"]
        esito, outc = _esiti(xs, ys)
        n = len(xs)
        Ln = _vettore_perdita(rn["Ms"]["Modello"], "1x2", xs, ys, outc, esito)
        Lf = _vettore_perdita(rn["Ms"]["Frequenze storiche"], "1x2", xs, ys, outc, esito)
        La = _vettore_perdita(ra["Ms"]["Modello"], "1x2", xs, ys, outc, esito) if ra is not None and ra["Partita"] == rn["Partita"] else np.full(n, np.nan)
        ok_m = ~np.isnan(rn["m1x2"]).any(axis=1)
        Lm = np.full(n, np.nan)
        Lm[ok_m] = -np.log(np.clip(rn["m1x2"][ok_m][np.arange(ok_m.sum()), esito[ok_m]], 1e-9, 1))
        tutte["new"].append(Ln); tutte["att"].append(La); tutte["mkt"].append(Lm); tutte["freq"].append(Lf)
        etichetta = f"{int(k)}/{(int(k) + 1) % 100:02d}" + (" (parziale)" if k == stagioni[-1] else "")
        riga = {"Stagione": etichetta, "Parametri scelti (λ / emivita)": f"{pk['ridge']} / {pk['halflife'] if pk['halflife'] else 'nessuna'}",
                "Partite": n, "Modello (parametri scelti prima)": round(float(Ln.mean()), 4),
                "Modello (parametri attuali)": round(float(np.nanmean(La)), 4) if np.isfinite(La).any() else None,
                "Frequenze storiche": round(float(Lf.mean()), 4),
                "Mercato (chiusura)": round(float(np.nanmean(Lm)), 4) if ok_m.any() else None}
        riga["Modello − Mercato (± err.)"] = _diff_se(Ln[ok_m], Lm[ok_m]) if ok_m.sum() > 5 else None
        riga["Scelti prima − Attuali (± err.)"] = _diff_se(Ln, La) if np.isfinite(La).all() else None
        righe.append(riga)
        # mercati gol/esatti: peso scelto sulle stagioni precedenti
        Mb, Mf = rn["Ms"]["Modello"], rn["Ms"]["Frequenze storiche"]
        for _, mk in MERCATI_GOL:
            wk = w["esatto"] if mk == "exact" else w["gol"]
            perd[mk]["Modello"].append(_vettore_perdita(Mb, mk, xs, ys, outc, esito))
            perd[mk]["Frequenze"].append(_vettore_perdita(Mf, mk, xs, ys, outc, esito))
            perd[mk]["Misto"].append(_vettore_perdita(wk * Mb + (1 - wk) * Mf, mk, xs, ys, outc, esito))
        pesi_usati["gol"].append(f"{etichetta.split(' ')[0]}: {int(w['gol'] * 100)}")
        pesi_usati["esatto"].append(f"{etichetta.split(' ')[0]}: {int(w['esatto'] * 100)}")
    if not righe:
        return None, None
    Ln, La, Lm, Lf = (np.concatenate(tutte[k_]) for k_ in ("new", "att", "mkt", "freq"))
    ok_m = np.isfinite(Lm)
    riga = {"Stagione": "TUTTE", "Parametri scelti (λ / emivita)": "", "Partite": len(Ln),
            "Modello (parametri scelti prima)": round(float(Ln.mean()), 4),
            "Modello (parametri attuali)": round(float(np.nanmean(La)), 4) if np.isfinite(La).any() else None,
            "Frequenze storiche": round(float(Lf.mean()), 4), "Mercato (chiusura)": round(float(Lm[ok_m].mean()), 4) if ok_m.any() else None,
            "Modello − Mercato (± err.)": _diff_se(Ln[ok_m], Lm[ok_m]) if ok_m.sum() > 5 else None,
            "Scelti prima − Attuali (± err.)": _diff_se(Ln, La) if np.isfinite(La).all() else None}
    tab1 = pd.DataFrame(righe + [riga])
    righe2 = []
    for etichetta, mk in MERCATI_GOL:
        M_, F_, X_ = (np.concatenate(perd[mk][v]) for v in ("Modello", "Frequenze", "Misto"))
        pesi_txt = " · ".join(pesi_usati["esatto" if mk == "exact" else "gol"])
        righe2.append({"Mercato": etichetta, "Partite": len(M_), "Modello": round(float(M_.mean()), 4), "Frequenze storiche": round(float(F_.mean()), 4),
                       "Misto (pesi scelti prima)": round(float(X_.mean()), 4), "Peso modello % per stagione": pesi_txt,
                       "Misto − Frequenze (± err.)": _diff_se(X_, F_), "Misto − Modello (± err.)": _diff_se(X_, M_)})
    return tab1, pd.DataFrame(righe2)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def stagioni_nuove_cached(div, ridge, halflife, passo, giorno, versione):
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    return (None, None) if df is None else test_stagioni_nuove(df, ridge, halflife, passo)


# ======================= TEST CLUBELO =======================
ELO_BASI = ["http://api.clubelo.com", "https://api.clubelo.com"]
ELO_PAESE = {"I1": "ITA", "E0": "ENG", "SP1": "ESP", "D1": "GER", "F1": "FRA", "N1": "NED"}
_ALIAS_ELO = {
    "man city": "mancity", "man united": "manunited", "nott m forest": "forest", "wolves": "wolves", "ath madrid": "atletico",
    "ath bilbao": "athletic", "espanol": "espanyol", "ein frankfurt": "frankfurt", "m gladbach": "gladbach",
    "bayern munich": "bayern", "paris sg": "psg", "fc koln": "koeln", "for sittard": "sittard", "st etienne": "stetienne",
    "st pauli": "stpauli", "union berlin": "unionberlin", "inter": "inter", "milan": "milan",
}


def _compatta_elo(nome):
    n = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = " ".join(t for t in n.split() if t not in _STOP_NOMI and not t.isdigit())
    return _ALIAS_ELO.get(n, n).replace(" ", "")


def _sim_elo(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()


def abbina_elo(squadre, nomi_elo):
    comp = {n: _compatta_elo(n) for n in nomi_elo}
    out = {}
    for t in squadre:
        ct = _compatta_elo(t)
        best, bs = None, 0.0
        for n, c in comp.items():
            sc = _sim_elo(ct, c)
            if sc > bs:
                best, bs = n, sc
        if best is not None and bs >= 0.88:
            out[t] = best
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def carica_elo(div, squadre, giorno):
    """Elo storico (ClubElo, gratuito) delle squadre. Ritorna (serie per squadra, squadre non trovate, errore)."""
    paese = ELO_PAESE.get(div)

    base_ok = {"b": None}

    def leggi_elo(percorso):
        err_ = None
        for base in ([base_ok["b"]] if base_ok["b"] else ELO_BASI):  # prima http, poi https; dopo il primo successo si usa solo quella
            d_, err_ = _leggi_csv(f"{base}/{percorso}")
            if d_ is not None:
                base_ok["b"] = base
                return d_, None
        return None, err_

    snap, err = leggi_elo(giorno)
    if snap is None or not {"Club", "Country"}.issubset(snap.columns):
        return {}, list(squadre), err or "formato inatteso"
    nomi = snap.loc[snap["Country"] == paese, "Club"].astype(str).unique().tolist()
    abb = abbina_elo(squadre, nomi)
    serie = {}
    for t, nome in abb.items():
        d, _ = leggi_elo(nome)
        if d is None or not {"Elo", "From", "To"}.issubset(d.columns):
            continue
        d["From"] = pd.to_datetime(d["From"], errors="coerce")
        d["To"] = pd.to_datetime(d["To"], errors="coerce")
        d = d.dropna(subset=["From", "To", "Elo"]).sort_values("From")
        if not d.empty:
            serie[t] = d[["From", "To", "Elo"]].reset_index(drop=True)
    return serie, [t for t in squadre if t not in serie], None


def aggiungi_elo(df, serie):
    """Elo di ogni squadra il giorno prima della partita (mai quello dopo)."""
    d = df.copy()
    d["EloH"], d["EloA"] = np.nan, np.nan
    prima = (d["Date"] - pd.Timedelta(days=1)).to_numpy()
    for t, srs in serie.items():
        froms, tos, elos = srs["From"].to_numpy(), srs["To"].to_numpy(), srs["Elo"].to_numpy(float)
        for col, tc in (("EloH", "HomeTeam"), ("EloA", "AwayTeam")):
            m = (d[tc] == t).to_numpy()
            if not m.any():
                continue
            dates = prima[m]
            idx = np.searchsorted(froms, dates, side="right") - 1
            ok = (idx >= 0) & (dates <= tos[np.clip(idx, 0, None)])
            val = np.full(m.sum(), np.nan)
            val[ok] = elos[idx[ok]]
            d.loc[m, col] = val
    d["EloDiff"] = (d["EloH"] - d["EloA"]) / 400.0
    return d


def test_elo(df, ridge, halflife, passo=14):
    """Walk-forward: stesso modello con e senza la differenza Elo come covariata, sulle stesse partite."""
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return None
    t = df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    M0s, M1s, xs, ys, m1x2, ms, betas = [], [], [], [], [], [], []
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=FINESTRA_GIORNI))]
        if len(blocco) and len(train) >= 150:
            m0 = adatta_modello(train, t, ridge, halflife, True)
            m1 = adatta_modello(train, t, ridge, halflife, True, elo_diff=train["EloDiff"].to_numpy())
            betas.append(m1["beta"])
            cnt = pd.concat([train["HomeTeam"], train["AwayTeam"]]).value_counts()
            for _, r in blocco.iterrows():
                h, a = r["HomeTeam"], r["AwayTeam"]
                lh0, la0 = lambda_partita(m0, h, a)
                lh1, la1 = lambda_partita(m1, h, a, r["EloDiff"])
                M0s.append(matrice_punteggi(lh0, la0, m0["rho"]))
                M1s.append(matrice_punteggi(lh1, la1, m1["rho"]))
                xs.append(min(int(r["FTHG"]), 10)); ys.append(min(int(r["FTAG"]), 10))
                qc = prima_tripla(r, COL_CHIUSURA)
                m1x2.append(senza_margine(qc) if qc else np.full(3, np.nan))
                ms.append(int(min(cnt.get(h, 0), cnt.get(a, 0))))
        t += pd.Timedelta(days=passo)
    if not xs:
        return None
    xs, ys, ms, m1x2 = np.array(xs), np.array(ys), np.array(ms), np.array(m1x2)
    M0, M1 = np.array(M0s), np.array(M1s)
    esito, outc = _esiti(xs, ys)
    righe = []
    for etichetta, mk in [("1X2", "1x2"), ("Over/Under 1.5", "over1.5"), ("Over/Under 2.5", "over2.5"), ("Over/Under 3.5", "over3.5"),
                          ("Goal / No Goal", "btts"), ("Risultato esatto (log score)", "exact")]:
        L0 = _vettore_perdita(M0, mk, xs, ys, outc, esito)
        L1 = _vettore_perdita(M1, mk, xs, ys, outc, esito)
        righe.append({"Mercato": etichetta, "Partite": len(xs), "Modello": round(float(L0.mean()), 4), "Modello + Elo": round(float(L1.mean()), 4),
                      "Modello+Elo − Modello (± err.)": _diff_se(L1, L0)})
    tab_a = pd.DataFrame(righe)
    L0 = _vettore_perdita(M0, "1x2", xs, ys, outc, esito)
    L1 = _vettore_perdita(M1, "1x2", xs, ys, outc, esito)
    ok_m = ~np.isnan(m1x2).any(axis=1)
    Lm = np.full(len(xs), np.nan)
    Lm[ok_m] = -np.log(np.clip(m1x2[ok_m][np.arange(ok_m.sum()), esito[ok_m]], 1e-9, 1))
    righe_b = []
    for nome, g in [("Squadra con meno di 15 partite di storico (neopromosse)", ms < 15), ("Tutte le altre", ms >= 15)]:
        gm = g & ok_m
        if gm.sum() >= 5:
            righe_b.append({"Gruppo": nome, "Partite": int(gm.sum()), "Modello": round(float(L0[gm].mean()), 4),
                            "Modello + Elo": round(float(L1[gm].mean()), 4), "Mercato (chiusura)": round(float(Lm[gm].mean()), 4),
                            "Modello+Elo − Modello (± err.)": _diff_se(L1[gm], L0[gm]),
                            "Modello+Elo − Mercato (± err.)": _diff_se(L1[gm], Lm[gm])})
    return tab_a, pd.DataFrame(righe_b), {"beta": float(np.mean(betas)) if betas else float("nan"), "n": len(xs)}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def elo_test_cached(div, ridge, halflife, passo, giorno, versione):
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    if df is None:
        return {"errore": "storico non disponibile"}
    squadre = tuple(sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])))
    serie, non_trovate, err = carica_elo(div, squadre, giorno)
    if err or not serie:
        return {"errore": f"impossibile scaricare gli Elo da ClubElo ({err or 'nessuna squadra trovata'})"}
    d2 = aggiungi_elo(df, serie)
    copertura = float(np.isfinite(d2["EloDiff"]).mean())
    d2 = d2[np.isfinite(d2["EloDiff"])].reset_index(drop=True)
    if len(d2) < 400:
        return {"errore": f"copertura Elo insufficiente ({len(d2)} partite con Elo per entrambe le squadre)", "non_trovate": non_trovate}
    out = test_elo(d2, ridge, halflife, passo)
    if out is None:
        return {"errore": "dati insufficienti per il test"}
    return {"tab_a": out[0], "tab_b": out[1], "info": out[2], "non_trovate": non_trovate, "copertura": copertura, "n_squadre": len(serie)}


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


# ======================= PRONOSTICO, DOPPIA CHANCE, COMBO, MULTIGOL =======================
_TOT = _I + _J
_ESITI_MASK = {"1": _MASK["p1"], "X": _MASK["px"], "2": _MASK["p2"],
               "1X": _MASK["p1"] | _MASK["px"], "X2": _MASK["px"] | _MASK["p2"], "12": _MASK["p1"] | _MASK["p2"]}
_MERC_MASK = {"Goal": _MASK["btts"], "No Goal": ~_MASK["btts"],
              "Over 1.5": _MASK["over1.5"], "Under 1.5": ~_MASK["over1.5"],
              "Over 2.5": _MASK["over2.5"], "Under 2.5": ~_MASK["over2.5"],
              "Over 3.5": _MASK["over3.5"], "Under 3.5": ~_MASK["over3.5"]}
LINEE_OU = ("Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5")
NOMI_ESITO = {"1": "1 (Casa)", "X": "X (Pareggio)", "2": "2 (Trasferta)"}


def matrice_congiunta(M_1x2, M_gol):
    """Distribuzione dei punteggi che ha i marginali 1X2 di M_1x2 e la struttura dei gol di M_gol."""
    out = M_gol.copy()
    for chiave in ("p1", "px", "p2"):
        mask = _MASK[chiave]
        pg, po = M_gol[mask].sum(), M_1x2[mask].sum()
        if pg > 0:
            out[mask] *= po / pg
    return out / out.sum()


def _riga_prob(etichetta, p, freq=None):
    r = {"Mercato": etichetta, "Probabilità %": round(float(p) * 100, 1), "Quota equa": round(1 / p, 2) if p > 1e-6 else None}
    if freq is not None and etichetta in freq:
        r["Media storica %"] = round(freq[etichetta] * 100, 1)
        r["Differenza (punti)"] = round(float(p) * 100 - freq[etichetta] * 100, 1)
    return r


def frequenze_combo(df):
    """Frequenza reale (ultime stagioni del campionato) di esiti, combo e multigol: serve da confronto con le probabilità del modello."""
    gh, ga = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    esiti = {"1": gh > ga, "X": gh == ga, "2": gh < ga}
    esiti["1X"], esiti["X2"], esiti["12"] = esiti["1"] | esiti["X"], esiti["X"] | esiti["2"], esiti["1"] | esiti["2"]
    gol = (gh > 0) & (ga > 0)
    merc = {"Goal": gol, "No Goal": ~gol}
    for linea in (1.5, 2.5, 3.5):
        ov_ = (gh + ga) > linea
        merc[f"Over {linea}"], merc[f"Under {linea}"] = ov_, ~ov_
    out = {e: float(m.mean()) for e, m in esiti.items()}
    out.update({f"{e} + {m}": float((esiti[e] & merc[m]).mean()) for e in esiti for m in merc})
    out.update({f"{g_} + {o_}": float((merc[g_] & merc[o_]).mean()) for g_ in ("Goal", "No Goal") for o_ in LINEE_OU})
    tot = gh + ga
    out.update({f"{a}-{b} gol": float(((tot >= a) & (tot <= b)).mean()) for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]})
    for nome, g_ in (("Casa", gh), ("Trasferta", ga)):
        for k in (0, 1, 2):
            out[f"{nome} {k} gol"] = float((g_ == k).mean())
        out[f"{nome} 3+ gol"] = float((g_ >= 3).mean())
        for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]:
            out[f"{nome} {a}-{b} gol"] = float(((g_ >= a) & (g_ <= b)).mean())
    if {"HTHG", "HTAG"}.issubset(df.columns):
        d2 = df.dropna(subset=["HTHG", "HTAG"])
        if len(d2) >= 100:
            h1 = (d2["HTHG"] + d2["HTAG"]).to_numpy()
            h2 = ((d2["FTHG"] - d2["HTHG"]) + (d2["FTAG"] - d2["HTAG"])).to_numpy()
            for nome, h in (("1° tempo", h1), ("2° tempo", h2)):
                for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]:
                    out[f"{nome} {a}-{b} gol"] = float(((h >= a) & (h <= b)).mean())
                out[f"{nome} 0 gol"] = float((h == 0).mean())
                out[f"{nome} 1 gol"] = float((h == 1).mean())
                out[f"{nome} almeno 1 gol"] = float((h >= 1).mean())
                out[f"{nome} almeno 2 gol"] = float((h >= 2).mean())
            out["Primo tempo più gol"] = float((h1 > h2).mean())
            out["Gol uguali nei due tempi"] = float((h1 == h2).mean())
            out["Secondo tempo più gol"] = float((h1 < h2).mean())
    return out


def quota_primo_tempo(df):
    """Quota dei gol segnati nel 1° tempo (casa, trasferta) nello storico del campionato; riserva 45%."""
    if not {"HTHG", "HTAG"}.issubset(df.columns):
        return 0.45, 0.45
    d = df.dropna(subset=["HTHG", "HTAG"])
    if len(d) < 100 or d["FTHG"].sum() == 0 or d["FTAG"].sum() == 0:
        return 0.45, 0.45
    return (float(min(max(d["HTHG"].sum() / d["FTHG"].sum(), 0.2), 0.7)), float(min(max(d["HTAG"].sum() / d["FTAG"].sum(), 0.2), 0.7)))


def tabella_split(df, s=10.0):
    """Come si dividono davvero i gol tra 1° e 2° tempo, dato il totale della partita (frequenze storiche, lisciate verso la divisione binomiale).
    S[t][k] = probabilità che in una partita con t gol ne siano stati segnati k nel 1° tempo."""
    S = np.zeros((11, 11))
    pi = 0.45
    d = None
    if {"HTHG", "HTAG"}.issubset(df.columns):
        d = df.dropna(subset=["HTHG", "HTAG"])
        if len(d) < 100:
            d = None
    if d is not None:
        ft = (d["FTHG"] + d["FTAG"]).to_numpy(int)
        h1 = (d["HTHG"] + d["HTAG"]).to_numpy(int)
        pi = float(min(max(h1.sum() / max(ft.sum(), 1), 0.3), 0.6))
    for t in range(11):
        prior = np.array([math.comb(t, k) * pi ** k * (1 - pi) ** (t - k) for k in range(t + 1)])
        if d is None:
            S[t, :t + 1] = prior
            continue
        m = ft == t
        counts = np.bincount(np.clip(h1[m], 0, t), minlength=t + 1)[:t + 1].astype(float)
        S[t, :t + 1] = (counts + s * prior) / (m.sum() + s)
    return S


def probs_tempi(M_gol, S):
    """Distribuzione dei gol del 1° e 2° tempo: il totale della partita viene da M_gol, la divisione tra i tempi dallo storico."""
    tot = np.add.outer(np.arange(M_gol.shape[0]), np.arange(M_gol.shape[1]))
    pt = np.array([M_gol[tot == t].sum() for t in range(11)])
    pt = pt / pt.sum()
    joint = np.zeros((11, 11))
    for t in range(11):
        for k in range(t + 1):
            joint[k, t - k] += pt[t] * S[t, k]
    return joint.sum(axis=1), joint.sum(axis=0), joint


def _finestra_sopra_media(finestre, freq, nome):
    """Tra le finestre di multigol sceglie quella in cui la probabilità del modello è più sopra la media storica del campionato
    (la più probabile in assoluto è quasi sempre la stessa); senza media storica ripiega sulla più probabile.
    Restituisce (finestra, probabilità, media storica oppure None)."""
    medie = {k: freq.get(f"{nome} {k} gol") for k in finestre} if freq else {}
    if medie and all(v is not None for v in medie.values()):
        k = max(finestre, key=lambda x: finestre[x] - medie[x])
        return k, finestre[k], medie[k]
    k = max(finestre, key=finestre.get)
    return k, finestre[k], None


# ======================= CHI FARÀ DI PIÙ: TIRI, TIRI IN PORTA, ANGOLI, AMMONIZIONI =======================
# Per ogni statistica: medie attese delle due squadre (stessa formula delle statistiche attese: casa/trasferta + subito dall'avversario, con shrinkage)
# e due conteggi binomiali negativi indipendenti, da cui "casa di più", "pari" e "trasferta di più". Nessuna quota su questi mercati.
CHI_STAT = [("Tiri", "HS", "AS"), ("Tiri in porta", "HST", "AST"), ("Corner", "HC", "AC"), ("Ammonizioni", "HY", "AY")]
CHI_K = 5            # shrinkage verso la media del campionato (partite virtuali); il test di taratura confronta altri valori
CHI_SCALA_PHI = 1.0  # fattore sulla sovradispersione stimata; il test di taratura confronta altri valori


def _prob_piu_di(mu_h, mu_a, phi):
    """Probabilità che la casa abbia di più, che sia pari, che abbia di più la trasferta. phi = sovradispersione (0 = Poisson)."""
    def pmf(mu, n):
        k = np.arange(n)
        if phi < 1e-6:
            return poisson.pmf(k, mu)
        r = 1.0 / phi
        return nbinom.pmf(k, r, r / (r + mu))
    m = max(mu_h, mu_a)
    n = int(m + 8 * math.sqrt(m * (1 + phi * m)) + 6)
    M = np.outer(pmf(mu_h, n), pmf(mu_a, n))
    M = M / M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def _segno_chi(pc, pp_, pt):
    """Segno 1/X/2 se un esito supera il 55%, altrimenti la doppia chance migliore se arriva almeno al 72%, altrimenti 'equilibrata'. Ritorna (segno, probabilità)."""
    p = {"1": pc, "X": pp_, "2": pt}
    top = max(p, key=p.get)
    if p[top] >= 0.55:
        return top, p[top]
    dc = {"1X": pc + pp_, "X2": pp_ + pt, "12": pc + pt}
    best = max(dc, key=dc.get)
    return (best, dc[best]) if dc[best] >= 0.72 else (None, p[top])


def _medie_incrementali(df, h, a, k=CHI_K, min_lega=100):
    """Per ogni partita, le medie attese (casa, trasferta) di una statistica calcolate solo con i giorni precedenti, e la sovradispersione stimata
    sui residui passati. Ritorna (df ordinato, mu_casa, mu_trasf, phi per partita, phi finale)."""
    d = df.sort_values("Date").reset_index(drop=True)
    n = len(d)
    mu_h, mu_a, phi_r = np.full(n, np.nan), np.full(n, np.nan), np.full(n, 0.05)
    xh, xa = d[h].to_numpy(float), d[a].to_numpy(float)
    casa, trasf, giorni = d["HomeTeam"].to_numpy(), d["AwayTeam"].to_numpy(), d["Date"].to_numpy()
    S = {"hh": {}, "ha": {}, "aa": {}, "ah": {}}
    N_ = {"c": {}, "t": {}}
    lh = la = 0.0
    nl = 0
    num = den = 0.0
    nres = 0
    phi = 0.05
    i = 0
    while i < n:
        j = i
        while j < n and giorni[j] == giorni[i]:
            j += 1
        for r in range(i, j):
            phi_r[r] = phi
            if nl >= min_lega:
                c, t = casa[r], trasf[r]
                lega_h, lega_a = lh / nl, la / nl
                nc, nt = N_["c"].get(c, 0), N_["t"].get(t, 0)
                mu_h[r] = ((S["hh"].get(c, 0.0) + k * lega_h) / (nc + k) + (S["ah"].get(t, 0.0) + k * lega_h) / (nt + k)) / 2
                mu_a[r] = ((S["aa"].get(t, 0.0) + k * lega_a) / (nt + k) + (S["ha"].get(c, 0.0) + k * lega_a) / (nc + k)) / 2
        for r in range(i, j):
            if np.isnan(xh[r]) or np.isnan(xa[r]):
                continue
            c, t = casa[r], trasf[r]
            if not np.isnan(mu_h[r]):
                for x_, m_ in ((xh[r], mu_h[r]), (xa[r], mu_a[r])):
                    num += (x_ - m_) ** 2 - m_
                    den += m_ ** 2
                    nres += 1
            S["hh"][c] = S["hh"].get(c, 0.0) + xh[r]
            S["ha"][c] = S["ha"].get(c, 0.0) + xa[r]
            S["aa"][t] = S["aa"].get(t, 0.0) + xa[r]
            S["ah"][t] = S["ah"].get(t, 0.0) + xh[r]
            N_["c"][c] = N_["c"].get(c, 0) + 1
            N_["t"][t] = N_["t"].get(t, 0) + 1
            lh += xh[r]
            la += xa[r]
            nl += 1
        if nres >= 200 and den > 0:
            phi = float(np.clip(num / den, 0.001, 1.0))
        i = j
    return d, mu_h, mu_a, phi_r, phi


def _perdite_chi(df, h, a, k=CHI_K, scala_phi=CHI_SCALA_PHI, pre=None):
    """Walk-forward per una statistica: ritorna (perdite modello, perdite frequenze, probabilità n×3, esiti, phi finale) oppure None.
    'pre' permette di riusare le medie già calcolate per lo stesso k (serve alla taratura)."""
    d, mu_h, mu_a, phi, phi_fin = pre if pre is not None else _medie_incrementali(df, h, a, k=k)
    xh, xa = d[h].to_numpy(float), d[a].to_numpy(float)
    esito = np.where(xh > xa, 0, np.where(xh == xa, 1, 2))
    valido = ~(np.isnan(mu_h) | np.isnan(xh) | np.isnan(xa))
    cnt = np.ones(3)
    Lm, Lf, P, E = [], [], [], []
    for r in range(len(d)):
        if valido[r]:
            p3 = np.array(_prob_piu_di(mu_h[r], mu_a[r], phi[r] * scala_phi))
            Lm.append(-math.log(max(p3[esito[r]], 1e-9)))
            Lf.append(-math.log(cnt[esito[r]] / cnt.sum()))
            P.append(p3)
            E.append(esito[r])
        if not (np.isnan(xh[r]) or np.isnan(xa[r])):
            cnt[esito[r]] += 1
    if len(P) < 300:
        return None
    return np.array(Lm), np.array(Lf), np.array(P), np.array(E), phi_fin


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def parametri_chi_fa_di_piu(div, giorno):
    """Per ogni statistica: sovradispersione, differenza di log score dal test a ritroso contro le sole frequenze e il suo z (negativo = il modello è meglio)."""
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    out = {}
    if df is None:
        return out
    for nome, h, a in CHI_STAT:
        if h in df.columns and a in df.columns:
            r = _perdite_chi(df, h, a)
            if r is not None:
                dd = r[0] - r[1]
                se = dd.std(ddof=1) / math.sqrt(len(dd))
                out[nome] = {"phi": float(r[4]), "z": float(dd.mean() / se) if se > 0 else 0.0, "d": float(dd.mean())}
    return out


def chi_fa_di_piu(df, casa, trasf, par):
    """Righe per la scheda partita: chi avrà di più per tiri, tiri in porta, angoli e ammonizioni (con il pareggio). Se nel test a ritroso il modello non batte
    le frequenze di almeno 0.02 di log score (z ≤ −2), le percentuali sono le frequenze storiche del campionato e lo dice."""
    righe = []
    for nome, h, a in CHI_STAT:
        if h not in df.columns or a not in df.columns or nome not in par:
            continue
        d = df.dropna(subset=[h, a])
        if len(d) < 100:
            continue
        lega_h, lega_a = d[h].mean(), d[a].mean()
        sm = lambda serie, lega: (serie.sum() + CHI_K * lega) / (len(serie) + CHI_K)
        dc, dt = d[d["HomeTeam"] == casa], d[d["AwayTeam"] == trasf]
        mh = (sm(dc[h], lega_h) + sm(dt[h], lega_h)) / 2
        ma = (sm(dt[a], lega_a) + sm(dc[a], lega_a)) / 2
        ok_modello = par[nome]["z"] <= -2 and par[nome]["d"] <= -0.02
        if ok_modello:
            pc, pp_, pt = _prob_piu_di(mh, ma, par[nome]["phi"] * CHI_SCALA_PHI)
        else:
            pc, pp_, pt = float((d[h] > d[a]).mean()), float((d[h] == d[a]).mean()), float((d[h] < d[a]).mean())
        segno, ps = _segno_chi(pc, pp_, pt)
        righe.append({"Statistica": nome, "Attesi (casa - trasf.)": f"{mh:.1f} - {ma:.1f}", "Casa di più %": round(pc * 100, 1), "Pari %": round(pp_ * 100, 1),
                      "Trasferta di più %": round(pt * 100, 1), "Segno": f"{segno} ({ps * 100:.0f}%)" if segno else "equilibrata",
                      "Basato su": "✅ modello (batte le frequenze)" if ok_modello else "⚠️ solo frequenza storica del campionato", "_segno": segno, "_modello": ok_modello})
    return righe


def test_chi_fa_di_piu(df):
    """Walk-forward: per ogni partita le medie attese usano solo i giorni precedenti. Confronta il modello con le sole frequenze (casa di più / pari / trasferta di più).
    Ritorna (tabella log score, tabella calibrazione) oppure None."""
    righe, cal = [], []
    for nome, h, a in CHI_STAT:
        if h not in df.columns or a not in df.columns:
            continue
        r = _perdite_chi(df, h, a)
        if r is None:
            continue
        Lm, Lf, pm, ac, _phi = r
        reale = [float((ac == i).mean()) * 100 for i in range(3)]
        righe.append({"Statistica": nome, "Partite": len(Lm), "Modello": round(float(Lm.mean()), 4), "Frequenze": round(float(Lf.mean()), 4),
                      "Modello − Frequenze (± err.)": _diff_se(Lm, Lf),
                      "Casa di più: previsto %": round(float(pm[:, 0].mean()) * 100, 1), "Casa di più: reale %": round(reale[0], 1),
                      "Pari: previsto %": round(float(pm[:, 1].mean()) * 100, 1), "Pari: reale %": round(reale[1], 1),
                      "Trasferta di più: previsto %": round(float(pm[:, 2].mean()) * 100, 1), "Trasferta di più: reale %": round(reale[2], 1)})
        top, atteso = pm.max(axis=1), pm.argmax(axis=1)
        for lo, hi, et in ((0, 0.45, "sotto 45%"), (0.45, 0.55, "45-55%"), (0.55, 0.65, "55-65%"), (0.65, 1.01, "oltre 65%")):
            m = (top >= lo) & (top < hi)
            if m.sum() >= 20:
                cal.append({"Statistica": nome, "Fascia del pronostico": et, "Partite": int(m.sum()), "Previsto %": round(float(top[m].mean()) * 100, 1),
                            "Reale %": round(float((atteso[m] == ac[m]).mean()) * 100, 1),
                            "Differenza (punti)": round((float(top[m].mean()) - float((atteso[m] == ac[m]).mean())) * 100, 1)})
    if not righe:
        return None
    return pd.DataFrame(righe), pd.DataFrame(cal)


def taratura_chi_fa_di_piu(df, ks=(2, 3, 5, 8), scale=(1.0, 0.7)):
    """Confronta shrinkage (k) e fattore sulla sovradispersione. Ogni riga è confrontata con la configurazione attuale (k=5, fattore 1.0) con l'errore statistico,
    e mostra pari e fascia 'oltre 65%' previsti e reali per vedere se la sottostima dei pronostici sicuri si riduce."""
    righe = []
    for nome, h, a in CHI_STAT:
        if h not in df.columns or a not in df.columns:
            continue
        base = _perdite_chi(df, h, a, k=5, scala_phi=1.0)
        if base is None:
            continue
        for k in ks:
            pre = _medie_incrementali(df, h, a, k=k)
            for sc in scale:
                r = _perdite_chi(df, h, a, k=k, scala_phi=sc, pre=pre)
                if r is None or len(r[0]) != len(base[0]):
                    continue
                Lm, Lf, pm, ac, _ = r
                top, atteso = pm.max(axis=1), pm.argmax(axis=1)
                m = top >= 0.65
                righe.append({"Statistica": nome, "k": k, "Fattore dispersione": sc, "Partite": len(Lm), "Log score": round(float(Lm.mean()), 4),
                              "Differenza dall'attuale (± err.)": _diff_se(Lm, base[0]),
                              "Pari previsto %": round(float(pm[:, 1].mean()) * 100, 1), "Pari reale %": round(float((ac == 1).mean()) * 100, 1),
                              "Oltre 65%: partite": int(m.sum()),
                              "Oltre 65%: previsto %": round(float(top[m].mean()) * 100, 1) if m.sum() else None,
                              "Oltre 65%: reale %": round(float((atteso[m] == ac[m]).mean()) * 100, 1) if m.sum() else None})
    return pd.DataFrame(righe) if righe else None


_NOMI_SCELTA = {"1": "1 (vince la casa)", "X": "X (pareggio)", "2": "2 (vince la trasferta)",
                "1X": "1X (la casa non perde)", "X2": "X2 (la trasferta non perde)", "12": "12 (non pareggia)"}


def quote_grezze_partita(casa, trasf):
    """Quote 1X2 del bookmaker per la partita, se sono nel file delle partite del turno (altrimenti None)."""
    try:
        if fx is None:
            return None
        m = fx[(fx["HomeTeam"] == casa) & (fx["AwayTeam"] == trasf)]
        if m.empty:
            return None
        q = prima_tripla(m.iloc[0], COL_PRESA)
        return tuple(float(x) for x in q) if q else None
    except Exception:
        return None


def scheda_semplice(casa, trasf, pr, d, curve, senza_storico, grezze, gol_casa, gol_trasf, gol_lega):
    """Riquadro in cima alla scheda: com'è la partita, la giocata, a quale quota ha senso, cosa evitare. Usa i numeri già calcolati e le curve del backtest
    (quante volte è davvero uscito un esito con quella probabilità prevista; livelli: alta da 70%, media da 55%).
    Ritorna (livello 'alta'/'media'/'bassa', testo in markdown). Non è un consiglio di profitto: il modello non batte le quote."""
    pp = pr["p"]
    top = pr["pronostico"]
    p_top = pp[top]
    equilibrata = top != "X" and abs(pp["1"] - pp["2"]) * 100 <= 8 and pp["X"] >= 0.25
    scelta, chiave, note = pr["doppia_best"], "doppia", []
    if p_top >= 0.50 and top != "X" and not equilibrata:
        f_s = cerca_fiducia(curve.get("1x2") if curve else None, p_top)
        if (f_s[0] if f_s else p_top) >= 0.55:
            scelta, chiave = top, "1x2"
        else:
            note.append("l'esito secco non supera il 55% nei backtest, quindi ti propongo la doppia chance")
    p_sc = pp[scelta]
    f_ = cerca_fiducia(curve.get(chiave) if curve else None, p_sc)
    hit = f_[0] if f_ else None
    livello = "alta" if (hit if hit is not None else p_sc) >= 0.70 else ("media" if (hit if hit is not None else p_sc) >= 0.55 else "bassa")
    motivi = []
    if senza_storico:
        motivi.append("una delle due squadre non ha storico (neopromossa o inizio stagione)")
        if livello != "bassa":
            livello = "media" if livello == "alta" else "bassa"
    if livello == "bassa" and not motivi:
        motivi.append("nessun esito è abbastanza probabile da fidarsi")
    if hit is None:
        note.append("pochi casi simili nel backtest, quindi la fiducia non è verificata")
    # com'è la partita, a parole
    if top == "X" or p_top < 0.42:
        partita_ = f"Partita aperta: nessuna delle due è davvero favorita (1 {pp['1'] * 100:.0f}%, X {pp['X'] * 100:.0f}%, 2 {pp['2'] * 100:.0f}%)."
    else:
        forza = "nettamente favorita" if p_top >= 0.65 else ("favorita" if p_top >= 0.52 else "leggermente favorita")
        partita_ = f"**{casa if top == '1' else trasf}** {forza}: vince circa {p_top * 10:.0f} volte su 10 ({p_top * 100:.0f}%)."
    tot = gol_casa + gol_trasf
    gol_ = "più gol del solito" if tot >= gol_lega + 0.3 else ("meno gol del solito" if tot <= gol_lega - 0.3 else "gol nella media")
    partita_ += f" Il pareggio vale {pp['X'] * 100:.0f}%. Si attendono circa {tot:.1f} gol ({gol_casa:.1f}-{gol_trasf:.1f}), {gol_}."
    righe = []
    if livello == "bassa":
        righe.append(f"**🎯 GIOCATA: PASSA.** Il meglio che c'è è {_NOMI_SCELTA[scelta]} al {p_sc * 100:.0f}%, ma la fiducia è **BASSA**.")
        righe.append("**⏭️ Perché passo:** " + "; ".join(motivi) + ".")
    else:
        titolo = f"**🎯 GIOCATA: {_NOMI_SCELTA[scelta]}** — fiducia **{'ALTA' if livello == 'alta' else 'MEDIA'}**"
        titolo += f": con questa probabilità è uscita circa {hit * 100:.0f} volte su 100 nei test ({f_[1]} casi)." if hit is not None else "."
        righe.append(titolo)
    righe.append("**📖 La partita:** " + partita_)
    if livello != "bassa":
        q_equa = 1 / p_sc
        riga_q = f"**💶 Quota minima: {q_equa:.2f}** (quota equa). "
        if grezze:
            qs = dict(zip(("1", "X", "2"), grezze))
            if scelta in qs:
                q_b, appr = qs[scelta], ""
            else:
                a_, b_ = {"1X": ("1", "X"), "X2": ("X", "2"), "12": ("1", "2")}[scelta]
                q_b, appr = 1 / (1 / qs[a_] + 1 / qs[b_]), "circa "
            ev = (p_sc * q_b - 1) * 100
            riga_q += f"Il bookmaker paga {appr}{q_b:.2f}: ogni 100 € giocati ne tornano in media {100 + ev:.0f} €. "
            riga_q += ("Con le quote reali il modello coincide con il mercato: il costo è il margine del bookmaker, non c'è un vantaggio da cercare."
                       if d["ancorata"] else "Il vantaggio, se c'è, è solo sulla carta: il modello non ha dimostrato di battere le quote.")
        else:
            riga_q += "Giocala solo se la trovi a questa quota o più alta: sotto, ti costa."
        righe.append(riga_q)
    outsider = [("il 1" if k == "1" else "il 2") + f" al {pp[k] * 100:.0f}%" for k in ("1", "2") if pp[k] < 0.25]
    righe.append("**⛔ Evita:** gli outsider" + (f" (qui {', '.join(outsider)})" if outsider else "") +
                 ", i multigol, i risultati esatti e le combo: nei nostri test le quote alte sugli outsider hanno reso peggio e questi mercati hanno margini più alti.")
    if note:
        righe.append("**Nota:** " + "; ".join(note) + ".")
    righe.append("_Il modello non batte le quote: la fiducia dice quante volte esce, non se conviene giocarla._")
    return livello, "\n\n".join(righe)


def _testo_multigol(p, f):
    return f"{p * 100:.1f}% (media {f * 100:.1f}%)" if f is not None else f"{p * 100:.1f}%"


def tabelle_tempi(M_gol, S, freq=None):
    p1, p2, pm = probs_tempi(M_gol, S)
    out = {}

    def righe(p, nome):
        rows = [_riga_prob(f"{nome} 0 gol", p[0], freq), _riga_prob(f"{nome} 1 gol", p[1], freq)]
        rows += [_riga_prob(f"{nome} {a}-{b} gol", p[a:b + 1].sum(), freq) for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]]
        rows.append(_riga_prob(f"{nome} almeno 1 gol", 1 - p[0], freq))
        rows.append(_riga_prob(f"{nome} almeno 2 gol", 1 - p[0] - p[1], freq))
        return pd.DataFrame(rows)
    out["t1"], out["t2"] = righe(p1, "1° tempo"), righe(p2, "2° tempo")
    out["piu_gol"] = pd.DataFrame([_riga_prob("Primo tempo più gol", np.tril(pm, -1).sum(), freq),
                                   _riga_prob("Gol uguali nei due tempi", np.trace(pm), freq),
                                   _riga_prob("Secondo tempo più gol", np.triu(pm, 1).sum(), freq)])
    for chiave, p, nome_t in (("t1", p1, "1° tempo"), ("t2", p2, "2° tempo")):
        w2 = {f"{a}-{b}": float(p[a:b + 1].sum()) for a, b in [(0, 1), (1, 2), (2, 3)]}
        out[chiave + "_best"], out[chiave + "_best_p"], out[chiave + "_best_f"] = _finestra_sopra_media(w2, freq, nome_t)
    piu = {"1° tempo": float(np.tril(pm, -1).sum()), "Pari": float(np.trace(pm)), "2° tempo": float(np.triu(pm, 1).sum())}
    out["piu_best"], out["piu_best_p"] = max(piu, key=piu.get), max(piu.values())
    return out


def tabelle_squadre(M_gol, freq=None):
    out = {}
    for chiave, marg, nome in (("casa", M_gol.sum(axis=1), "Casa"), ("trasf", M_gol.sum(axis=0), "Trasferta")):
        rows = [_riga_prob(f"{nome} {k} gol", marg[k], freq) for k in (0, 1, 2)] + [_riga_prob(f"{nome} 3+ gol", marg[3:].sum(), freq)]
        rows += [_riga_prob(f"{nome} {a}-{b} gol", marg[a:b + 1].sum(), freq) for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]]
        out[chiave] = pd.DataFrame(rows)
        w2 = {f"{a}-{b}": float(marg[a:b + 1].sum()) for a, b in [(0, 1), (1, 2), (2, 3)]}
        out[chiave + "_best"], out[chiave + "_best_p"], out[chiave + "_best_f"] = _finestra_sopra_media(w2, freq, nome)
    return out


def trova_soglia_x(P, esito, n_min=50):
    """Soglia di probabilità di pareggio da cui in poi puntare la X azzecca più che puntare la squadra favorita tra 1 e 2.
    Prudente: il vantaggio deve superare 1.5 volte l'errore statistico e valere per ogni soglia più alta con abbastanza partite."""
    P = np.asarray(P)
    lato = np.where(P[:, 0] >= P[:, 2], 0, 2)
    d_tutti = (esito == 1).astype(float) - (esito == lato).astype(float)
    validi = []
    for t in np.arange(0.24, 0.371, 0.01):
        m = P[:, 1] >= t - 1e-9
        n = int(m.sum())
        if n < n_min:
            break
        d = d_tutti[m]
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
        validi.append((round(float(t), 2), float(d.mean()) >= 1.5 * se))
    soglia = None
    for t, ok_ in reversed(validi):
        if not ok_:
            break
        soglia = t
    return soglia


def tabella_dove_x(bt):
    px = bt["px"].to_numpy()
    y = bt["y"].to_numpy()
    righe = []
    for etichetta, lo, hi in [("X prevista sotto 22%", 0, 0.22), ("22-25%", 0.22, 0.25), ("25-28%", 0.25, 0.28), ("28-31%", 0.28, 0.31),
                              ("31-34%", 0.31, 0.34), ("34% e oltre", 0.34, 1.01)]:
        m = (px >= lo) & (px < hi)
        if m.sum() >= 10:
            righe.append({"Fascia": etichetta, "Partite": int(m.sum()), "1 reale %": round(float((y[m] == 0).mean() * 100), 1),
                          "X reale %": round(float((y[m] == 1).mean() * 100), 1), "2 reale %": round(float((y[m] == 2).mean() * 100), 1)})
    return pd.DataFrame(righe)


def tabelle_pronostico(M_1x2, M_gol, freq=None, soglia_x=None):
    Mj = matrice_congiunta(M_1x2, M_gol)
    p = {k: float(M_1x2[m].sum()) for k, m in _ESITI_MASK.items()}
    out = {"p": p}
    best = max(("1", "X", "2"), key=lambda k: p[k])
    out["x_regola"] = False
    if soglia_x is not None and p["X"] >= soglia_x and best != "X":
        best, out["x_regola"] = "X", True
    out["pronostico"] = best
    out["doppia_x_best"] = max(("1X", "X2"), key=lambda k: p[k])
    out["doppie"] = pd.DataFrame([_riga_prob(k, p[k], freq) for k in ("1X", "X2", "12")]).sort_values("Probabilità %", ascending=False)
    out["doppia_best"] = max(("1X", "X2", "12"), key=lambda k: p[k])

    def combo(esiti, mercati):
        righe = [(f"{e} + {m}", float(Mj[_ESITI_MASK[e] & _MERC_MASK[m]].sum())) for e in esiti for m in mercati]
        return pd.DataFrame([_riga_prob(l, v, freq) for l, v in sorted(righe, key=lambda r: -r[1])])
    out["combo_goal"] = combo(("1", "X", "2"), ("Goal", "No Goal"))
    out["combo_ou"] = combo(("1", "X", "2"), ("Over 2.5", "Under 2.5"))
    out["combo_doppie"] = combo(("1X", "X2", "12"), ("Goal", "No Goal", "Over 2.5", "Under 2.5"))
    out["combo_ou_linee"] = combo(("1", "X", "2"), LINEE_OU)
    out["combo_doppie_goal"] = combo(("1X", "X2", "12"), ("Goal", "No Goal"))
    out["combo_doppie_ou"] = combo(("1X", "X2", "12"), LINEE_OU)
    righe_go = [(f"{g_} + {o_}", float(Mj[_MERC_MASK[g_] & _MERC_MASK[o_]].sum())) for g_ in ("Goal", "No Goal") for o_ in LINEE_OU]
    out["combo_goal_ou"] = pd.DataFrame([_riga_prob(l, v, freq) for l, v in sorted(righe_go, key=lambda r: -r[1])])
    # Multigol (totale e per squadra) dalla distribuzione dei gol
    out["multigol_tot"] = pd.DataFrame([_riga_prob(f"{a}-{b} gol", M_gol[(_TOT >= a) & (_TOT <= b)].sum(), freq)
                                        for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]])
    larghi3 = {f"{a}-{b}": float(M_gol[(_TOT >= a) & (_TOT <= b)].sum()) for a, b in [(1, 3), (2, 4), (3, 5)]}
    out["multigol_best"] = max(larghi3, key=larghi3.get)
    out["multigol_best_p"] = larghi3[out["multigol_best"]]
    out["multigol_casa"] = pd.DataFrame([_riga_prob(f"Casa {a}-{b} gol", M_gol[(_I >= a) & (_I <= b)].sum()) for a, b in [(1, 2), (1, 3), (2, 3), (2, 4)]])
    out["multigol_trasf"] = pd.DataFrame([_riga_prob(f"Trasferta {a}-{b} gol", M_gol[(_J >= a) & (_J <= b)].sum()) for a, b in [(1, 2), (1, 3), (2, 3), (2, 4)]])
    return out


def diagnostica_pareggi(bt):
    """Il modello sottostima i pareggi? Confronto per fasce di equilibrio tra 1 e 2."""
    gap = (bt["p1"] - bt["p2"]).abs().to_numpy()
    reale = (bt["y"].to_numpy() == 1).astype(float)
    px = bt["px"].to_numpy()
    mkt = bt["mx"].to_numpy()
    righe = []
    for etichetta, sel in [("Differenza 1-2 fino a 4 punti", gap <= 0.04), ("fino a 8 punti", gap <= 0.08), ("fino a 12 punti", gap <= 0.12),
                           ("fino a 20 punti", gap <= 0.20), ("Tutte le partite", gap >= 0)]:
        if sel.sum() >= 10:
            righe.append({"Fascia": etichetta, "Partite": int(sel.sum()), "X prevista dal modello %": round(float(px[sel].mean() * 100), 1),
                          "X reale %": round(float(reale[sel].mean() * 100), 1),
                          "X prevista dal mercato %": round(float(np.nanmean(mkt[sel]) * 100), 1) if np.isfinite(mkt[sel]).any() else None})
    top = np.argmax(bt[["p1", "px", "p2"]].to_numpy(), axis=1)
    n_top_x = int((top == 1).sum())
    return pd.DataFrame(righe), n_top_x


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
try:
    _chiave_apif = st.secrets.get("API_FOOTBALL_KEY", "")
except Exception:
    _chiave_apif = ""
chiave_apif = st.sidebar.text_input("🔑 API Key API-Football (facoltativa)", value=_chiave_apif, type="password")
st.sidebar.caption("Serve per le quote e il calendario delle coppe: la verifica è nella scheda 🌍 Coppe. La chiave non viene salvata.")
st.sidebar.markdown("---")
st.sidebar.info("💡 Il modello sceglie da solo i propri parametri (nessuna impostazione da fare). Da cellulare puoi cambiare campionato dal menu in alto.")

giorno = date.today().isoformat()
df_storico = carica_storico(div, N_STAGIONI, giorno)
if df_storico is None or len(df_storico) < 100:
    st.warning("Impossibile scaricare lo storico da football-data.co.uk (o dati insufficienti). Riprova tra poco.")
    st.stop()
par = parametri_auto(div, giorno)
modello = modello_corrente(div, par["ridge"], par["halflife"], giorno)
fx, fx_err = carica_fixtures(div)
oggi_ts = pd.Timestamp.now().normalize()
calendario, fonte_cal = costruisci_calendario(div, df_storico, fx, chiave_api, modello["squadre"])

# Quote del prossimo turno (1X2 + Over/Under 2.5 senza margine) e distribuzione dei punteggi "migliore disponibile"
pesi = pesi_blend_auto(div, par["ridge"], par["halflife"], giorno)
METODO_MARGINE["m"] = pesi.get("metodo_margine", "proporzionale")
modello_tiri = modello_tiri_corrente(div, par["ridge"], par["halflife"], giorno) if (pesi.get("tiri_gol") or pesi.get("tiri_esatto")) else None
M_freq_live = matrice_frequenze(df_storico)
freq_combo = frequenze_combo(df_storico)
quota_pt = quota_primo_tempo(df_storico)
split_live = tabella_split(df_storico)
quote_fixtures = {}
if fx is not None:
    for _, _r in fx.iterrows():
        _q, _qo = prima_tripla(_r, COL_PRESA), prima_coppia(_r, COL_OU_PRESA)
        if _q and _qo:
            quote_fixtures[(_r["HomeTeam"], _r["AwayTeam"])] = (senza_margine(_q), float(senza_margine(_qo)[0]))


def distribuzione(casa, trasf):
    """Con quote reali (1X2 + Over/Under 2.5): gol attesi ancorati alle quote, usati per tutti i mercati.
    Senza quote: 1X2 dal modello; Goal/Over/Esatti da modello + frequenze storiche (pesi scelti dal backtest)."""
    lh, la = lambda_partita(modello, casa, trasf)
    Mb = matrice_punteggi(lh, la, modello["rho"])
    q = quote_fixtures.get((casa, trasf))
    if q is not None:
        Ma = matrice_ancorata(q[0], q[1], modello["rho"], (lh, la))
        return {"M_1x2": Ma, "M_gol": Ma, "M_es": Ma, "ancorata": True,
                "fonte": "ancorata alle quote reali del bookmaker (1X2 + Over/Under 2.5)"}
    Mt = Mb
    if modello_tiri is not None and (pesi.get("tiri_gol") or pesi.get("tiri_esatto")):
        lth, lta = lambda_partita(modello_tiri["modello"], casa, trasf)
        Mt = matrice_punteggi(math.sqrt(lh * lth * modello_tiri["conv"][0]), math.sqrt(la * lta * modello_tiri["conv"][1]), modello["rho"])
    Mb_g = Mt if (modello_tiri is not None and pesi.get("tiri_gol")) else Mb
    Mb_e = Mt if (modello_tiri is not None and pesi.get("tiri_esatto")) else Mb
    Mg = pesi["gol"] * Mb_g + (1 - pesi["gol"]) * M_freq_live
    Me = pesi["esatto"] * Mb_e + (1 - pesi["esatto"]) * M_freq_live
    usa_t = [n_ for n_, f_ in (("Goal/Over", pesi.get("tiri_gol")), ("risultati esatti", pesi.get("tiri_esatto"))) if f_ and modello_tiri is not None]
    return {"M_1x2": Mb, "M_gol": Mg, "M_es": Me, "ancorata": False,
            "fonte": f"modello + frequenze storiche (peso del modello: {pesi['gol']*100:.0f}% per Goal/Over, {pesi['esatto']*100:.0f}% per i risultati esatti)"
                     + (f", con i tiri in porta per {' e '.join(usa_t)}" if usa_t else "") + " – nessuna quota disponibile"}


def gol_attesi(M):
    return float((M.sum(axis=1) * np.arange(M.shape[0])).sum()), float((M.sum(axis=0) * np.arange(M.shape[1])).sum())

# Value finder: partite del prossimo turno con quote reali
righe_val = []
if fx is not None:
    for _, r in fx.iterrows():
        if "Date" in r and pd.notna(r["Date"]) and r["Date"] < oggi_ts:
            continue  # partita già giocata
        q = prima_tripla(r, COL_PRESA)
        if not q:
            continue
        casa, trasf = r["HomeTeam"], r["AwayTeam"]
        lh, la = lambda_partita(modello, casa, trasf)
        mk = mercati(matrice_punteggi(lh, la, modello["rho"]))
        P = np.array([mk["p1"], mk["px"], mk["p2"]])
        pm = senza_margine(q)
        nomi = [f"1 ({casa})", "X (Pareggio)", f"2 ({trasf})"]
        for k in range(3):
            righe_val.append({
                "Data": r["Date"].strftime("%d/%m") if "Date" in r and pd.notna(r["Date"]) else "",
                "Partita": f"{casa} vs {trasf}", "Selezione": nomi[k],
                "P1 %": round(P[0] * 100, 1), "PX %": round(P[1] * 100, 1), "P2 %": round(P[2] * 100, 1),
                "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1),
                "Quota_Book": round(q[k], 2), "Edge": round((P[k] * q[k] - 1) * 100, 1),
                "Nota": "⚠ senza storico" if (casa not in modello["idx"] or trasf not in modello["idx"]) else "",
            })
df_value = pd.DataFrame(righe_val)  # tutte le selezioni (3 per partita)

tab_calendario, tab_pronostici, tab_classifica, tab_value_finder, tab_ai_schedine, tab_monte_carlo, tab_backtest, tab_kelly, tab_coppe = st.tabs([
    "📅 Calendario", "📋 Pronostici", "🏆 Classifica", "⚡ Value Finder", "🤖 Schedine AI", "🎲 Monte Carlo", "🧪 Backtest", "🧮 Value Bet", "🌍 Coppe",
])


def mostra_analisi(casa, trasf, data, ora):
    d = distribuzione(casa, trasf)
    lh, la = gol_attesi(d["M_gol"])
    M = d["M_es"]
    mk = mercati(d["M_1x2"])
    mg = mercati(d["M_gol"])
    soglia_x = pesi.get("soglia_x", {}).get("ancorato" if d["ancorata"] else "modello")
    pr = tabelle_pronostico(d["M_1x2"], d["M_gol"], freq_combo, soglia_x)
    pt = tabelle_tempi(d["M_gol"], split_live, freq_combo)
    ps = tabelle_squadre(d["M_gol"], freq_combo)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 18px 25px; margin-top: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">📋 SCHEDA DINAMICA DELLA PARTITA</span>
            <span style="color: #94a3b8; font-size: 0.95rem;">📅 {data} {ora}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    curve_v = pesi.get("curve", {}).get("con" if d["ancorata"] else "senza")
    gl_c, gl_t = gol_attesi(d["M_gol"])
    liv_v, testo_v = scheda_semplice(casa, trasf, pr, d, curve_v, casa not in modello["idx"] or trasf not in modello["idx"],
                                     quote_grezze_partita(casa, trasf), gl_c, gl_t, sum(gol_attesi(M_freq_live)))
    {"alta": st.success, "media": st.info, "bassa": st.warning}[liv_v](testo_v)
    cf_righe = chi_fa_di_piu(df_storico, casa, trasf, parametri_chi_fa_di_piu(div, giorno))
    if cf_righe:
        st.markdown("#### ⚔️ Chi farà di più")
        st.markdown("**In breve:** " + " · ".join(f"{r['Statistica']} **{r['_segno'] or '—'}**" + ("" if r["_modello"] else " ⚠️") for r in cf_righe))
        st.dataframe(pd.DataFrame(cf_righe).drop(columns=["_segno", "_modello"]), use_container_width=True, hide_index=True)
        st.caption("1 = più la casa, X = pari, 2 = più la trasferta; 1X, X2 o 12 sono doppie chance quando nessun esito supera il 55%. Medie storiche reali delle due squadre (in casa o in trasferta e subite dall'avversario). "
                   "Non ci sono quote su questi mercati: sono indicazioni, non valore. ⚠️ = nel test a ritroso il modello non batte abbastanza le sole frequenze, quindi uso la frequenza storica del campionato.")
    with st.expander("🎯 Tutti i pronostici e le combo", expanded=False):
        pp = pr["p"]
        st.markdown("#### 🎯 Il pronostico")
        r1, r2, r3, r4 = st.columns(4)
        dx = (pp[pr["pronostico"]] - pp["X"]) * 100
        gap12 = abs(pp["1"] - pp["2"]) * 100
        equilibrata = pr["pronostico"] != "X" and gap12 <= 8 and pp["X"] >= 0.25
        r1.metric("Pronostico 1X2", NOMI_ESITO[pr["pronostico"]], f"{pp[pr['pronostico']]*100:.1f}%" + (" · partita equilibrata" if equilibrata else "") + (" · X scelta dalla soglia del backtest" if pr["x_regola"] else ""), delta_color="off")
        r2.metric("Doppia chance più probabile", pr["doppia_best"], f"{pp[pr['doppia_best']]*100:.1f}%", delta_color="off")
        cg = pr["combo_goal"].iloc[0]
        r3.metric("Combo più probabile (1X2 + Goal)", cg["Mercato"], f"{cg['Probabilità %']:.1f}%", delta_color="off")
        r4.metric("Multigol più probabile (3 gol)", pr["multigol_best"] + " gol", f"{pr['multigol_best_p']*100:.1f}%", delta_color="off")
        if equilibrata:
            st.info(f"⚖️ Partita equilibrata: 1 e 2 sono a {gap12:.1f} punti e la X vale {pp['X']*100:.1f}% (a {dx:.1f} punti dal pronostico). "
                    f"Il pareggio qui è un esito molto concreto: nel campionato esce circa il {freq_combo['X']*100:.0f}% delle volte e di più nelle partite equilibrate "
                    "(nei backtest circa il 30%). Un solo esito resta più probabile, ma 1X / X2 coprono bene il pareggio.")
        s1, s2, s3 = st.columns(3)
        cs = pr["combo_goal"].sort_values("Differenza (punti)", ascending=False).iloc[0]
        s1.metric("Più sopra la media (1X2 + Goal)", cs["Mercato"], f"{cs['Differenza (punti)']:+.1f} punti rispetto alla media storica", delta_color="off")
        co1 = pr["combo_ou"].iloc[0]
        s2.metric("Combo più probabile (1X2 + Over/Under 2.5)", co1["Mercato"], f"{co1['Probabilità %']:.1f}%", delta_color="off")
        cs2 = pr["combo_ou"].sort_values("Differenza (punti)", ascending=False).iloc[0]
        s3.metric("Più sopra la media (1X2 + Over/Under 2.5)", cs2["Mercato"], f"{cs2['Differenza (punti)']:+.1f} punti rispetto alla media storica", delta_color="off")
        k1_, k2_, k3_ = st.columns(3)
        c25 = lambda df_: df_[df_["Mercato"].str.contains("2.5")].iloc[0]
        kd = c25(pr["combo_doppie_ou"])
        kg = c25(pr["combo_goal_ou"])
        kdg = pr["combo_doppie_goal"].iloc[0]
        k1_.metric("Doppia chance + Over/Under 2.5 più probabile", kd["Mercato"], f"{kd['Probabilità %']:.1f}%", delta_color="off")
        k2_.metric("Goal + Over/Under 2.5 più probabile", kg["Mercato"], f"{kg['Probabilità %']:.1f}%", delta_color="off")
        k3_.metric("Doppia chance + Goal più probabile", kdg["Mercato"], f"{kdg['Probabilità %']:.1f}%", delta_color="off")
        st.caption(f"Fonte Goal / Over-Under / Risultati esatti: {d['fonte']}.")
        if not d["ancorata"]:
            st.warning("⚠ Senza le quote del bookmaker, Goal/No Goal e Over/Under hanno poco potere predittivo: nel backtest su circa 1200 partite "
                       "non hanno battuto in modo netto le semplici frequenze storiche. Considerali indicativi, vicini alla media del campionato.")
    with st.expander("📊 Analisi completa: squadre, mercati, tempi, multigol", expanded=False):
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
            righe_st = {"Gol Attesi": (round(lh, 2), round(la, 2))}
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
        curve_ = pesi.get("curve", {}).get("con" if d["ancorata"] else "senza")
        st.markdown("#### 📏 Quanto fidarsi di queste percentuali")
        pg_ = mg["btts"]
        side_g = ("Goal", pg_) if pg_ >= 0.5 else ("No Goal", 1 - pg_)
        po_ = mg["over2.5"]
        side_o = ("Over 2.5", po_) if po_ >= 0.5 else ("Under 2.5", 1 - po_)
        voci = [("Pronostico 1X2: " + NOMI_ESITO[pr["pronostico"]], pp[pr["pronostico"]], "1x2"),
                ("Doppia chance " + pr["doppia_best"], pp[pr["doppia_best"]], "doppia"),
                ("Doppia chance con la X: " + pr["doppia_x_best"], pp[pr["doppia_x_best"]], "doppia"),
                (side_g[0], side_g[1], "goal"), (side_o[0], side_o[1], "ou")]
        righe_f = []
        for nome_v, pv, chiave in voci:
            f_ = cerca_fiducia(curve_.get(chiave) if curve_ else None, pv)
            righe_f.append({"Mercato": nome_v, "Previsto %": round(pv * 100, 1),
                            "Nei backtest è uscito %": round(f_[0] * 100, 1) if f_ else None, "Casi simili": f_[1] if f_ else None})
        st.dataframe(pd.DataFrame(righe_f), use_container_width=True, hide_index=True)
        st.caption("Tra le partite del backtest in cui l'app dava una percentuale simile, quante volte è davvero uscito. Un 70% previsto è uscito circa 7 volte su 10: "
                   "anche la percentuale più alta perde spesso. Se i casi simili sono pochi o mancano, la percentuale è poco verificata.")
        u1_, u2_, u3_ = st.columns(3)
        u1_.metric("Doppia chance con la X più probabile", pr["doppia_x_best"], f"{pp[pr['doppia_x_best']]*100:.1f}%", delta_color="off")
        u2_.metric("Multigol casa più sopra la media (2 gol)", ps["casa_best"] + " gol", _testo_multigol(ps["casa_best_p"], ps["casa_best_f"]), delta_color="off")
        u3_.metric("Multigol trasferta più sopra la media (2 gol)", ps["trasf_best"] + " gol", _testo_multigol(ps["trasf_best_p"], ps["trasf_best_f"]), delta_color="off")
        v1_, v2_, v3_ = st.columns(3)
        v1_.metric("Multigol 1° tempo più sopra la media (2 gol)", pt["t1_best"] + " gol", _testo_multigol(pt["t1_best_p"], pt["t1_best_f"]), delta_color="off")
        v2_.metric("Multigol 2° tempo più sopra la media (2 gol)", pt["t2_best"] + " gol", _testo_multigol(pt["t2_best_p"], pt["t2_best_f"]), delta_color="off")
        v3_.metric("Tempo con più gol", pt["piu_best"], f"{pt['piu_best_p']*100:.1f}%", delta_color="off")
        st.caption("Multigol da 2 gol (0-1, 1-2, 2-3): la finestra mostrata è quella in cui il modello sta più sopra la media storica del campionato, "
                   "non la più probabile in assoluto, che sarebbe quasi sempre la stessa. Sopra la media non significa valore: non c'è una quota con cui confrontarla. "
                   "Nelle tabelle in basso trovi tutte le finestre e le percentuali di 0, 1 e 2+ gol con la media accanto.")
        gap = abs(pp["1"] - pp["2"]) * 100
        st.caption(f"La X vale {pp['X']*100:.1f}% (media storica del campionato: {freq_combo['X']*100:.1f}%) e la differenza tra 1 e 2 è di {gap:.1f} punti. "
                   "La X è quasi mai l'esito singolo più probabile, anche con le quote del mercato: nelle partite equilibrate guarda la doppia chance. "
                   "'Più sopra la media' mostra dove il modello vede qualcosa di diverso dal solito, mentre 'più probabile' dice soprattutto chi è la favorita. "
                   "Nel Backtest, sezioni 'Il pareggio' e 'Calibrazione Goal', trovi se il modello sbaglia sistematicamente.")
        a1, a2, a3 = st.columns(3)
        a1.metric("Segno 1 (Casa)", f"{mk['p1']*100:.1f}%", f"Quota equa: {1/mk['p1']:.2f}")
        a2.metric("Segno X (Pareggio)", f"{mk['px']*100:.1f}%", f"Quota equa: {1/mk['px']:.2f}")
        a3.metric("Segno 2 (Trasferta)", f"{mk['p2']*100:.1f}%", f"Quota equa: {1/mk['p2']:.2f}")
        st.markdown("#### ⚽ Analisi Under / Over & Goal / No Goal")
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric("Over 2.5", f"{mg['over2.5']*100:.1f}%", f"Under: {(1-mg['over2.5'])*100:.1f}%")
        u2.metric("Over 1.5", f"{mg['over1.5']*100:.1f}%", f"Under: {(1-mg['over1.5'])*100:.1f}%")
        u3.metric("Over 3.5", f"{mg['over3.5']*100:.1f}%", f"Under: {(1-mg['over3.5'])*100:.1f}%")
        u4.metric("Goal (BTTS)", f"{mg['btts']*100:.1f}%", f"Quota: {1/mg['btts']:.2f}")
        u5.metric("No Goal", f"{(1-mg['btts'])*100:.1f}%", f"Quota: {1/(1-mg['btts']):.2f}")
        st.markdown("#### 🎯 Top 3 Risultati Esatti più Probabili")
        flat = sorted(((i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])), key=lambda t: t[2], reverse=True)[:3]
        for col, medaglia, nome, (i, j, p) in zip(st.columns(3), ["🥇", "🥈", "🥉"], ["1°", "2°", "3°"], flat):
            col.info(f"{medaglia} {nome} Esatto: **{i} - {j}** ({p*100:.1f}%)")
        st.markdown("#### 🛡️ Doppia chance")
        st.dataframe(pr["doppie"], use_container_width=True, hide_index=True)
        st.markdown("#### 🔗 Combo")
        ct1, ct2, ct3, ct4, ct5 = st.tabs(["1X2 + Goal", "1X2 + Over/Under", "Doppia chance + Goal", "Doppia chance + Over/Under", "Goal + Over/Under"])
        ct1.dataframe(pr["combo_goal"], use_container_width=True, hide_index=True)
        ct2.dataframe(pr["combo_ou_linee"], use_container_width=True, hide_index=True)
        ct3.dataframe(pr["combo_doppie_goal"], use_container_width=True, hide_index=True)
        ct4.dataframe(pr["combo_doppie_ou"], use_container_width=True, hide_index=True)
        ct5.dataframe(pr["combo_goal_ou"], use_container_width=True, hide_index=True)
        st.caption("Over/Under sulle linee 1.5, 2.5 e 3.5. Le righe sono ordinate per probabilità: le combo con 'Over 1.5' o 'Under 3.5' sono quasi sempre in cima perché sono molto probabili, "
                   "ma pagano poco. La colonna 'Differenza' dice dove il modello si discosta dalla media storica.")
        st.markdown("#### 🎰 Multigol totale")
        st.dataframe(pr["multigol_tot"], use_container_width=True, hide_index=True)
        st.markdown("#### 👥 Gol e multigol delle due squadre")
        mm1, mm2 = st.columns(2)
        mm1.markdown(f"**{casa}**")
        mm1.dataframe(ps["casa"], use_container_width=True, hide_index=True)
        mm2.markdown(f"**{trasf}**")
        mm2.dataframe(ps["trasf"], use_container_width=True, hide_index=True)
        st.markdown("#### 🕐 Primo e secondo tempo")
        tt1, tt2 = st.columns(2)
        tt1.markdown("**1° tempo**")
        tt1.dataframe(pt["t1"], use_container_width=True, hide_index=True)
        tt2.markdown("**2° tempo**")
        tt2.dataframe(pt["t2"], use_container_width=True, hide_index=True)
        st.markdown("**Quale tempo ha più gol**")
        st.dataframe(pt["piu_gol"], use_container_width=True, hide_index=True)
        st.caption(f"Combo, multigol e tempi sono calcolati dalla stessa distribuzione dei punteggi (1X2 come mostrato sopra, gol secondo la fonte indicata). "
                   f"I tempi usano il totale dei gol della partita e come i gol si sono divisi davvero tra primo e secondo tempo nello storico del campionato "
                   f"(circa il {quota_pt[0]*100:.0f}% dei gol in casa e il {quota_pt[1]*100:.0f}% in trasferta arriva nel 1° tempo): è un'approssimazione, verificata nel Backtest. Un multigol più largo è sempre più probabile ma paga meno: confronta la quota equa con quella offerta.")
        if d["ancorata"]:
            st.caption("Con le quote reali le probabilità 1X2 coincidono con quelle del mercato (senza margine): non c'è edge da cercare qui.")
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

# --- PRONOSTICI DEL TURNO ---
with tab_pronostici:
    st.subheader("📋 Pronostici delle prossime partite")
    st.caption("Tutte le partite dei prossimi 10 giorni con il pronostico dell'app e quanto, storicamente, quella percentuale è uscita. "
               "Le partite con quote reali hanno la parte Goal/Over-Under ancorata al mercato; senza quote è indicativa.")
    limite_ = (oggi_ts + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    righe_p = []
    for _, riga in calendario.iterrows():
        if riga["Stato"] == "FINISHED" or not riga["Data"] or not (oggi_ts.strftime("%Y-%m-%d") <= riga["Data"] <= limite_):
            continue
        c_, t_ = riga["Casa"], riga["Trasferta"]
        d_ = distribuzione(c_, t_)
        sg_ = pesi.get("soglia_x", {}).get("ancorato" if d_["ancorata"] else "modello")
        prp = tabelle_pronostico(d_["M_1x2"], d_["M_gol"], None, sg_)
        pp_ = prp["p"]
        mg_ = mercati(d_["M_gol"])
        cv_ = pesi.get("curve", {}).get("con" if d_["ancorata"] else "senza", {})
        top_p = pp_[prp["pronostico"]]
        db_ = prp["doppia_best"]
        fid_d = cerca_fiducia(cv_.get("doppia"), pp_[db_])
        fid_1 = cerca_fiducia(cv_.get("1x2"), top_p)
        g_ = ("Goal", mg_["btts"]) if mg_["btts"] >= 0.5 else ("No Goal", 1 - mg_["btts"])
        o_ = ("Over 2.5", mg_["over2.5"]) if mg_["over2.5"] >= 0.5 else ("Under 2.5", 1 - mg_["over2.5"])
        es_ = d_["M_es"].ravel()
        ie = int(np.argmax(es_))
        righe_p.append({
            "Data": riga["Data"][5:], "Partita": f"{c_} - {t_}", "Quote": "sì" if d_["ancorata"] else "no",
            "Pronostico 1X2": f"{NOMI_ESITO[prp['pronostico']]} {top_p*100:.0f}%",
            "Uscito storicamente (1X2) %": round(fid_1[0] * 100, 1) if fid_1 else None,
            "Doppia chance": f"{db_} {pp_[db_]*100:.0f}%", "Quota equa doppia": round(1 / pp_[db_], 2),
            "Uscita storica (doppia) %": round(fid_d[0] * 100, 1) if fid_d else None,
            "Goal/No Goal": f"{g_[0]} {g_[1]*100:.0f}%", "Over/Under 2.5": f"{o_[0]} {o_[1]*100:.0f}%",
            "Esatto più probabile": f"{ie // 11}-{ie % 11} ({es_[ie]*100:.0f}%)",
            "_ordine": pp_[db_],
        })
    if not righe_p:
        st.info("Nessuna partita nei prossimi 10 giorni nel calendario. Con la API key di football-data.org (barra laterale) il calendario è completo; "
                "senza, vedi solo il prossimo turno quando football-data.co.uk lo pubblica.")
    else:
        ordina = st.radio("Ordina per", ["Doppia chance più sicura", "Data"], horizontal=True, key="ord_pron")
        dfp = pd.DataFrame(righe_p)
        dfp = dfp.sort_values("_ordine", ascending=False) if ordina.startswith("Doppia") else dfp.sort_values("Data")
        st.dataframe(dfp.drop(columns="_ordine"), use_container_width=True, hide_index=True)
        st.caption("'Uscita storica' è la frequenza con cui, nei backtest, una percentuale simile è davvero uscita (se manca, i casi storici sono pochi). "
                   "Non è una garanzia: anche al 75% l'esito non esce una volta su quattro.")

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
        st.warning(f"Impossibile scaricare il prossimo turno e le quote (fixtures.csv): {fx_err}.")
    elif df_value.empty:
        st.info("Nessuna partita futura con quote: il file fixtures.csv viene aggiornato pochi giorni prima di ogni turno "
                "e finché non c'è il turno successivo contiene solo partite già giocate.")
    else:
        st.caption("Edge = prob. modello × quota − 1, sulla media dei bookmaker. Qui si usa il modello puro: con le probabilità ancorate alle quote "
                   "l'edge sarebbe sempre negativo (il margine del bookmaker). Un edge alto di solito è un errore del modello: "
                   "controlla nel tab Backtest da quale soglia l'edge ha avuto CLV positivo.")
        min_edge = st.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
        min_prob = st.slider("Probabilità di mercato minima della selezione (%)", 0, 60, PROB_MIN_DEFAULT)
        st.caption("Le sfavorite (probabilità di mercato bassa) nei backtest perdono molto più del margine: in tre campionati le vittorie in casa sotto il 25% "
                   "hanno reso circa −35%. Per questo di default non le mostro: abbassa il filtro solo se sai cosa fai.")
        bt_b = backtest_cached(div, par["ridge"], par["halflife"], 14, giorno)
        if bt_b is not None and not bt_b.empty:
            n_b, clv_m, clv_se = riassunto_clv(bt_b, min_edge)
            if n_b < 30 or not np.isfinite(clv_m):
                st.info(f"Backtest: solo {n_b} scommesse simulate con edge ≥ {min_edge:.0f}%, troppo poche per giudicare.")
            elif clv_m + 2 * clv_se < 0:
                st.warning(f"⚠ Edge NON dimostrato. Nel backtest, le {n_b} scommesse simulate con edge ≥ {min_edge:.0f}% hanno avuto CLV medio "
                           f"{clv_m:+.1f}% (± {clv_se:.1f}): in passato questo segnale ha preso quote peggiori del mercato di chiusura.")
            elif clv_m - 2 * clv_se > 0:
                st.success(f"Nel backtest le {n_b} scommesse con edge ≥ {min_edge:.0f}% hanno avuto CLV medio {clv_m:+.1f}% (± {clv_se:.1f}). "
                           "È un buon segnale, ma va confermato su partite nuove.")
            else:
                st.info(f"Backtest: CLV medio {clv_m:+.1f}% (± {clv_se:.1f}) su {n_b} scommesse con edge ≥ {min_edge:.0f}%: "
                        "non distinguibile da zero, quindi nessun vantaggio dimostrato.")
        vis_ = migliori_per_partita(df_value, min_prob)
        vis_ = vis_[vis_["Edge"] >= min_edge] if not vis_.empty else vis_
        if vis_.empty:
            st.info("Nessuna selezione con questi filtri.")
        else:
            st.dataframe(vis_, use_container_width=True, hide_index=True)
        if (df_value["Nota"] != "").any():
            st.caption("⚠ Squadre senza storico (es. neopromosse) sono trattate come squadra media: probabilmente sopravvalutate.")

# --- SCHEDINE AI ---
with tab_ai_schedine:
    st.subheader("🤖 Schedine Smart & Combo AI")
    base = migliori_per_partita(df_value[df_value["Nota"] == ""], PROB_MIN_DEFAULT) if not df_value.empty else df_value
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
            d_mc = distribuzione(sq_c, sq_t)
            M = d_mc["M_gol"]  # stessa distribuzione dei mercati gol dell'analisi (con Dixon-Coles)
            st.caption(f"Distribuzione simulata: {d_mc['fonte']}.")
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

# ======================= NUOVE FUNZIONI: TEMPI PER STAGIONE E TEST SU TUTTI I CAMPIONATI =======================
def tabella_tempi_per_stagione(df):
    """Come si dividono davvero i gol dei due tempi, stagione per stagione: quota di partite con 0, 1, 2+ gol e con 0-1 / 1-2 gol."""
    if df is None or not {"HTHG", "HTAG", "Stag", "FTHG", "FTAG"}.issubset(df.columns):
        return pd.DataFrame()
    d = df.dropna(subset=["HTHG", "HTAG"])
    if len(d) < 100:
        return pd.DataFrame()
    h1 = (d["HTHG"] + d["HTAG"]).to_numpy().astype(int)
    h2 = ((d["FTHG"] - d["HTHG"]) + (d["FTAG"] - d["HTAG"])).to_numpy().astype(int)
    stag = d["Stag"].to_numpy().astype(int)
    oggi = date.today()
    anno_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1

    def riga(etichetta, tempo, h):
        n = len(h)
        p2 = float((h >= 2).mean())
        return {"Stagione": etichetta, "Tempo": tempo, "Partite": n,
                "0 gol %": round(float((h == 0).mean()) * 100, 1), "1 gol %": round(float((h == 1).mean()) * 100, 1),
                "2+ gol %": round(p2 * 100, 1), "± 2+ gol": round(math.sqrt(p2 * (1 - p2) / n) * 100, 1),
                "0-1 gol %": round(float((h <= 1).mean()) * 100, 1), "1-2 gol %": round(float(((h >= 1) & (h <= 2)).mean()) * 100, 1)}
    righe = []
    for tempo, h in (("1° tempo", h1), ("2° tempo", h2)):
        for y in sorted(set(stag.tolist())):
            m = stag == y
            etichetta = f"{y}/{(y + 1) % 100:02d}" + (" (in corso)" if y == anno_corrente else "")
            righe.append(riga(etichetta, tempo, h[m]))
        righe.append(riga("Tutte", tempo, h))
    return pd.DataFrame(righe)


def tabella_punteggi_bassi(res):
    """Stagione per stagione: frequenza reale dei punteggi bassi (0-0, 1-0, 0-1, 1-1) contro quella prevista dal modello nel test a ritroso.
    Solo un controllo: non cambia nessuna previsione."""
    if res is None or "Ms" not in res or "date" not in res:
        return pd.DataFrame()
    d = pd.to_datetime(pd.Series(res["date"]))
    stag = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1).astype(int)
    xs, ys = np.asarray(res["xs"]).astype(int), np.asarray(res["ys"]).astype(int)
    oggi = date.today()
    anno_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1
    g = np.add.outer(np.arange(11), np.arange(11))

    def prev(nome, f):
        M = res["Ms"].get(nome)
        if M is None or len(M) != len(xs):
            return np.full(len(xs), np.nan)
        return f(M)
    p00 = prev("Modello", lambda M: M[:, 0, 0])
    p10 = prev("Modello", lambda M: M[:, 1, 0] + M[:, 0, 1])
    p11 = prev("Modello", lambda M: M[:, 1, 1])
    p_mod = prev("Modello", lambda M: M[:, :2, :2].sum(axis=(1, 2)))
    p_sdc = prev("Senza Dixon-Coles", lambda M: M[:, :2, :2].sum(axis=(1, 2)))
    p_anc = prev("Ancorato al mercato", lambda M: M[:, :2, :2].sum(axis=(1, 2)))
    gol_prev = prev("Modello", lambda M: (M * g).sum(axis=(1, 2)))
    r00 = (xs == 0) & (ys == 0)
    r10 = ((xs == 1) & (ys == 0)) | ((xs == 0) & (ys == 1))
    r11 = (xs == 1) & (ys == 1)
    rtot = (xs <= 1) & (ys <= 1)
    gol = (xs + ys).astype(float)

    def riga(etichetta, m):
        n = int(m.sum())
        if n < 20 or np.isnan(p_mod[m]).all():
            return None
        pr = float(rtot[m].mean())
        riga_ = {"Periodo": etichetta, "Partite": n,
                 "0-0 reale %": round(float(r00[m].mean()) * 100, 1), "0-0 previsto %": round(float(np.nanmean(p00[m])) * 100, 1),
                 "1-0 o 0-1 reale %": round(float(r10[m].mean()) * 100, 1), "1-0 o 0-1 previsto %": round(float(np.nanmean(p10[m])) * 100, 1),
                 "1-1 reale %": round(float(r11[m].mean()) * 100, 1), "1-1 previsto %": round(float(np.nanmean(p11[m])) * 100, 1),
                 "Bassi reali %": round(pr * 100, 1), "± err.": round(math.sqrt(pr * (1 - pr) / n) * 100, 1),
                 "Bassi previsti % (Modello)": round(float(np.nanmean(p_mod[m])) * 100, 1),
                 "Differenza Modello (punti)": round((float(np.nanmean(p_mod[m])) - pr) * 100, 1),
                 "Bassi previsti % (senza Dixon-Coles)": round(float(np.nanmean(p_sdc[m])) * 100, 1) if not np.isnan(p_sdc[m]).all() else None}
        sub = m & ~np.isnan(p_anc)
        riga_["Differenza Ancorato (punti)"] = round((float(p_anc[sub].mean()) - float(rtot[sub].mean())) * 100, 1) if sub.sum() >= 20 else None
        riga_["Gol reali"] = round(float(gol[m].mean()), 2)
        riga_["Gol previsti"] = round(float(np.nanmean(gol_prev[m])), 2)
        return riga_
    righe = []
    for y in sorted(set(stag.tolist())):
        r_ = riga(f"{y}/{(y + 1) % 100:02d}" + (" (in corso)" if y == anno_corrente else ""), stag == y)
        if r_:
            righe.append(r_)
    r_ = riga("Tutte", np.ones(len(xs), dtype=bool))
    if r_:
        righe.append(r_)
    ultime = np.zeros(len(xs), dtype=bool)
    ultime[np.argsort(d.to_numpy(), kind="stable")[-100:]] = True
    r_ = riga("Ultime 100 partite", ultime)
    if r_:
        righe.append(r_)
    return pd.DataFrame(righe)


def riepilogo_punteggi_bassi(risultati):
    """Punteggi bassi con i campionati sommati (più partite, errore statistico più piccolo)."""
    tabs = [r.get("punteggi_bassi") for r in risultati.values()]
    tabs = [t for t in tabs if t is not None and not t.empty]
    if not tabs:
        return pd.DataFrame()
    T = pd.concat(tabs)
    periodi = sorted(set(T["Periodo"]), key=lambda x: (2 if x.startswith("Ultime") else 1 if x == "Tutte" else 0, x))
    righe = []
    for periodo in periodi:
        gr = T[T["Periodo"] == periodo]
        w = gr["Partite"].to_numpy().astype(float)
        n = int(w.sum())
        pr = float(np.average(gr["Bassi reali %"], weights=w)) / 100
        pm = float(np.average(gr["Bassi previsti % (Modello)"], weights=w))
        righe.append({"Periodo": periodo, "Partite": n, "Bassi reali %": round(pr * 100, 1), "± err.": round(math.sqrt(pr * (1 - pr) / n) * 100, 1),
                      "Bassi previsti % (Modello)": round(pm, 1), "Differenza Modello (punti)": round(pm - pr * 100, 1),
                      "Gol reali": round(float(np.average(gr["Gol reali"], weights=w)), 2), "Gol previsti": round(float(np.average(gr["Gol previsti"], weights=w)), 2)})
    return pd.DataFrame(righe)


NOMI_TABELLE_TEST = ["1_qualita_per_mercato", "2_calibrazione_goal_over", "3_tempi", "4_rendimento_roi",
                     "5_presi_e_non_presi", "6_risultati_esatti", "7_ultime_25_esatti", "8_squadre_poca_storia"]


def esegui_test_campionato(div, giorno):
    """Stesso test del pulsante 'Esegui test completo', per un campionato qualsiasi (parametri e pesi propri di quel campionato).
    Restituisce le 8 tabelle, oppure None se i dati non bastano."""
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    if df is None or len(df) < 100:
        return None
    par_ = parametri_auto(div, giorno)
    pesi_ = pesi_blend_auto(div, par_["ridge"], par_["halflife"], giorno)
    METODO_MARGINE["m"] = pesi_.get("metodo_margine", "proporzionale")
    res = esegui_test_completo(df, par_["ridge"], par_["halflife"], 14)
    if res is None:
        return None
    tab_a, tab_b, tab_c, tab_d, tab_e, best_ex, tab_cal, tab_roi, tab_tempi = tabelle_test(res, pesi_)
    return {"tabelle": [tab_a, tab_cal, tab_tempi, tab_roi, tab_b, tab_c, tab_d, tab_e], "tempi_stagione": tabella_tempi_per_stagione(df), "punteggi_bassi": tabella_punteggi_bassi(res), "chi_di_piu": test_chi_fa_di_piu(df), "taratura_chi": taratura_chi_fa_di_piu(df)}


def _cella_cal(cal, mercato, fonte, fascia):
    """Differenza prevista − reale (punti) dalla tabella di calibrazione; fonte può essere un prefisso."""
    if cal is None or cal.empty:
        return None
    m = cal[(cal["Mercato"] == mercato) & (cal["Fascia"] == fascia) & cal["Fonte"].astype(str).str.startswith(fonte)]
    return None if m.empty else float(m.iloc[0]["Differenza (punti)"])


def riepilogo_campionati(risultati):
    """Una riga per campionato con i numeri che servono a confrontarli (stessi controlli letti a mano dopo ogni test)."""
    righe = []
    for nome, r in risultati.items():
        tab_a, tab_cal, tab_e = r["tabelle"][0], r["tabelle"][1], r["tabelle"][7]
        riga = {"Campionato": nome}
        x = tab_a[tab_a["Mercato"] == "1X2"] if tab_a is not None and not tab_a.empty else pd.DataFrame()
        if not x.empty:
            x = x.iloc[0]
            riga["Partite"] = int(x["Partite"])
            riga["1X2 modello − mercato"] = round(float(x["Modello"]) - float(x["Mercato (chiusura)"]), 4)
            riga["1X2 ancorato − mercato"] = round(float(x["Ancorato al mercato"]) - float(x["Mercato (chiusura)"]), 4)
        riga["Goal ancorato (prev. − reale)"] = _cella_cal(tab_cal, "Goal", "Ancorato al mercato", "TUTTE")
        riga["Over 2.5 modello (prev. − reale)"] = _cella_cal(tab_cal, "Over 2.5", "Modello da solo", "TUTTE")
        riga["Over 2.5 modello, fascia 60-100%"] = _cella_cal(tab_cal, "Over 2.5", "Modello da solo", "60-100%")
        if tab_e is not None and not tab_e.empty:
            for gruppo, etichetta in (("Squadra con meno di 15", "Poca storia: modello − mercato"), ("Tutte le altre", "Altre squadre: modello − mercato")):
                g_ = tab_e[tab_e["Gruppo"].astype(str).str.startswith(gruppo)]
                if not g_.empty:
                    g_ = g_.iloc[0]
                    riga[etichetta] = round(float(g_["Log loss 1X2 (Modello)"]) - float(g_["Log loss 1X2 (Mercato chiusura)"]), 4)
        righe.append(riga)
    return pd.DataFrame(righe)


def zip_test_campionati(risultati):
    """ZIP con le 8 tabelle di ogni campionato in CSV (una cartella per campionato)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nome, r in risultati.items():
            cartella = re.sub(r"[^A-Za-z0-9]+", "_", nome.split(" (")[0]).strip("_")
            for nome_t, tab in zip(NOMI_TABELLE_TEST, r["tabelle"]):
                if tab is not None and not tab.empty:
                    z.writestr(f"{cartella}/{nome_t}.csv", "\ufeff" + tab.to_csv(index=False))
            ts_ = r.get("tempi_stagione")
            if ts_ is not None and not ts_.empty:
                z.writestr(f"{cartella}/9_tempi_per_stagione.csv", "\ufeff" + ts_.to_csv(index=False))
            pb_ = r.get("punteggi_bassi")
            if pb_ is not None and not pb_.empty:
                z.writestr(f"{cartella}/10_punteggi_bassi.csv", "\ufeff" + pb_.to_csv(index=False))
            ch_ = r.get("chi_di_piu")
            if ch_ is not None:
                z.writestr(f"{cartella}/11_chi_fa_di_piu_log_score.csv", "\ufeff" + ch_[0].to_csv(index=False))
                z.writestr(f"{cartella}/12_chi_fa_di_piu_calibrazione.csv", "\ufeff" + ch_[1].to_csv(index=False))
            ta_ = r.get("taratura_chi")
            if ta_ is not None:
                z.writestr(f"{cartella}/13_taratura_chi_fa_di_piu.csv", "\ufeff" + ta_.to_csv(index=False))
    return buf.getvalue()


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
            tf_ = tabella_scommesse_fasce(bt)
            if not tf_.empty:
                st.markdown("**Le stesse scommesse (edge ≥ 3%) divise per probabilità di mercato dell'esito scelto**")
                st.dataframe(tf_, use_container_width=True, hide_index=True)
                st.caption("Se le fasce basse perdono molto più delle alte, il filtro dell'app (ultima riga) migliora il rendimento. Il ROI resta rumoroso: guarda l'errore.")
            st.caption("Il ROI su poche centinaia di scommesse è molto rumoroso: guarda l'errore (± ROI) e il CLV. "
                       "Un CLV medio positivo e stabile è un segnale più affidabile del ROI.")
        st.markdown("#### 🎲 Il pareggio (X): il modello lo sottostima?")
        dg, n_top_x = diagnostica_pareggi(bt)
        if not dg.empty:
            st.dataframe(dg, use_container_width=True, hide_index=True)
            st.caption(f"In {n_top_x} partite su {len(bt)} la X era l'esito più probabile secondo il modello. Se 'X reale' è molto più alto di 'X prevista dal modello' "
                       "nelle partite equilibrate, il modello sottostima i pareggi; se è vicino (o vicino a quella del mercato), la X non compare perché non è quasi mai "
                       "l'esito più probabile, non per un errore.")
        tt_ = pesi.get("tab_tiri")
        if tt_ is not None and not tt_.empty:
            st.markdown("#### 🎯 Tiri in porta: l'app li usa in questo campionato?")
            st.dataframe(tt_, use_container_width=True, hide_index=True)
            st.caption("Valori negativi = i tiri in porta migliorano il modello. L'app li usa per Goal/Over-Under e risultati esatti (senza quote) solo dove il miglioramento "
                       "supera 1.5 volte l'errore; per l'1X2 non li usa mai.")
        tm = pesi.get("tab_margine")
        if tm is not None and not tm.empty:
            st.markdown("#### 🧮 Come togliere il margine dalle quote")
            st.dataframe(tm, use_container_width=True, hide_index=True)
            st.caption("Le quote includono il margine del bookmaker: per ricavare probabilità vere ci sono più metodi. 'Proporzionale' sopravvaluta le sfavorite; "
                       "'potenza' e 'Shin' le correggono. Log loss più basso = probabilità più accurate. L'app passa a un altro metodo solo se migliora in modo chiaro "
                       "(almeno 1 volta l'errore).")
        st.markdown("#### 📏 Affidabilità storica delle percentuali (usata dall'app)")
        taf = tabella_affidabilita(pesi.get("curve", {}))
        if not taf.empty:
            st.dataframe(taf, use_container_width=True, hide_index=True)
            st.caption("Per ogni mercato e fascia di probabilità mostrata dall'app: quante volte è davvero uscito. Se 'Prevista' e 'Uscita davvero' sono vicine, "
                       "la percentuale è affidabile in quella fascia. Gli stessi dati su cui l'app ha stimato i pesi: un controllo fuori campione è nel test su stagioni nuove.")
        st.markdown("#### 🎲 Dove la X esce più spesso delle altre")
        tx = tabella_dove_x(bt)
        if not tx.empty:
            st.dataframe(tx, use_container_width=True, hide_index=True)
        sx_ = pesi.get("soglia_x", {})
        st.caption("La X diventa pronostico dove puntare la X azzecca (in modo statisticamente chiaro) più che puntare la favorita tra 1 e 2: in pratica dove 'X reale' supera sia '1 reale' sia '2 reale'. "
                   f"Soglia trovata dall'app: modello {('X ≥ ' + str(int(sx_['modello'] * 100)) + '%') if sx_.get('modello') else 'nessuna fascia in cui la X sia la più frequente'}; "
                   f"con le quote {('X ≥ ' + str(int(sx_['ancorato'] * 100)) + '%') if sx_.get('ancorato') else 'nessuna fascia in cui la X sia la più frequente'}.")
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

    st.markdown("---")
    st.markdown("### 🔬 Test completo: Goal/No Goal, Over/Under e Risultati Esatti")
    st.caption(f"Test su {N_STAGIONI_TEST} stagioni (la prima è storico iniziale), sempre walk-forward. Varianti: Modello, Senza Dixon-Coles, "
               "Ancorato al mercato (gol attesi ricavati dalle quote 1X2 e Over/Under 2.5 prima della partita), Misto 50/50, "
               "Tiri in porta (gol attesi stimati dai tiri in porta), Gol + Tiri 50/50, Frequenze storiche e "
               "Modello+Frequenze (con il peso migliore per ogni mercato). Valori più bassi = meglio.")
    st.caption(f"Pesi usati nell'app quando non ci sono quote (scelti dal backtest): modello {pesi['gol']*100:.0f}% per Goal/Over-Under, "
               f"{pesi['esatto']*100:.0f}% per i risultati esatti" + (f", su {pesi['n']} partite." if pesi.get("auto") else " (valori di riserva: dati insufficienti)."))
    if st.button("🧪 Esegui test completo"):
        with st.spinner("Calcolo in corso (può richiedere un minuto)..."):
            st.session_state["tc"] = test_completo_cached(div, par["ridge"], par["halflife"], 14, giorno, VERSIONE_TEST)
            st.session_state["tc_div"] = div
    tc = st.session_state.get("tc") if st.session_state.get("tc_div") == div else None
    if tc is not None and ("tiri_ok" not in tc or "q" not in tc or "split" not in tc):
        tc = None  # risultato vecchio rimasto nella sessione: rilancia il test
    if tc is not None:
        tab_a, tab_b, tab_c, tab_d, tab_e, best_ex, tab_cal, tab_roi, tab_tempi = tabelle_test(tc, pesi)
        st.markdown("#### 📏 Qualità per mercato (log loss, più basso è meglio)")
        st.dataframe(tab_a, use_container_width=True, hide_index=True)
        st.caption("Per Risultato esatto il valore è il log score del punteggio reale. 'Mercato (chiusura)' è il riferimento delle quote senza margine, "
                   "disponibile solo per 1X2 e Over/Under 2.5. Le colonne con ± mostrano la differenza media per partita e il suo errore: "
                   "se la differenza è più piccola di circa 2 volte l'errore, non è distinguibile dal caso.")
        st.markdown("#### ⚖️ Calibrazione Goal e Over 2.5: il modello li sottostima o li sovrastima?")
        if not tab_cal.empty:
            st.dataframe(tab_cal, use_container_width=True, hide_index=True)
            st.caption("Differenza = prevista − reale, in punti percentuali. Se per il Goal è NEGATIVA (prevista più bassa della reale), il modello sottostima il Goal "
                       "e sovrastima il No Goal. Le righe 'TUTTE' dicono se c'è uno sbilanciamento medio; le fasce dicono dove. "
                       "La riga 'APP senza quote' è quello che l'app usa quando non ci sono quote; con le quote usa 'Ancorato al mercato'.")
        if not tab_tempi.empty:
            st.markdown("#### 🕐 Primo e secondo tempo: previsto contro reale")
            st.dataframe(tab_tempi, use_container_width=True, hide_index=True)
            st.caption("Verifica del modello dei tempi (totale dei gol della partita + divisione tra i tempi dallo storico). "
                       "Differenza = prevista − reale in punti: vicino a 0 va bene; se è grande, i tempi vanno corretti.")
        if not tab_roi.empty:
            st.markdown("#### 💶 Rendimento per esito e probabilità di mercato (puntando sempre quell'esito)")
            st.dataframe(tab_roi, use_container_width=True, hide_index=True)
            st.caption("Quote di apertura (media dei bookmaker), 1 unità a partita. ROI atteso se il mercato fosse perfetto: circa −5% (il margine). "
                       "Guarda solo le righe con tante partite e un ROI più alto di 2 volte l'errore: con tante caselle, qualcuna esce fortunata per caso.")
        st.markdown("#### ✅ Presi e non presi: quando il modello è sicuro, quante volte ha ragione")
        st.dataframe(tab_b, use_container_width=True, hide_index=True)
        st.caption("Se 'Presi %' è molto sotto 'Prob. dichiarata %', il modello è troppo ottimista a quella soglia.")
        st.markdown("#### 🎯 Risultati esatti: presi e non presi")
        st.dataframe(tab_c, use_container_width=True, hide_index=True)
        st.markdown(f"Ultime 25 partite con la variante migliore sui risultati esatti ({best_ex}):")
        st.dataframe(tab_d, use_container_width=True, hide_index=True)
        if not tab_e.empty:
            st.markdown("#### 🆕 Squadre con poca storia (neopromosse)")
            st.dataframe(tab_e, use_container_width=True, hide_index=True)
            st.caption("Stesse partite, tre fonti: se anche il mercato sbaglia di più su queste squadre, non è un difetto solo del modello.")
        tab_pb = tabella_punteggi_bassi(tc)
        if not tab_pb.empty:
            st.markdown("#### 📉 Punteggi bassi nel tempo (0-0, 1-0, 0-1, 1-1)")
            st.dataframe(tab_pb, use_container_width=True, hide_index=True)
            st.caption("Controllo, non cambia nessuna previsione. Reale contro previsto dal modello nel test a ritroso. Differenza positiva = il modello prevede più punteggi bassi di quanti ne escono. "
                       "'± err.' è l'errore statistico: differenze entro circa 2 volte quel valore possono essere rumore. L'ultima stagione è parziale.")

    st.markdown("---")
    st.markdown("### 🌍 Test completo su tutti i campionati")
    st.caption("Stesso test del pulsante 'Esegui test completo' (8 stagioni, parametri e pesi propri di ogni campionato), lanciato in sequenza sui campionati scelti. "
               "Può richiedere diversi minuti la prima volta di giornata. Alla fine trovi un riepilogo e un file ZIP con le 8 tabelle di ogni campionato.")
    scelti = st.multiselect("Campionati", list(LEAGUES.keys()), default=list(LEAGUES.keys()), key="tutti_scelti")
    if st.button("🌍 Esegui test su tutti i campionati"):
        if not scelti:
            st.info("Scegli almeno un campionato.")
        else:
            risultati_tutti, saltati_tutti = {}, []
            stato_ = st.empty()
            barra_ = st.progress(0.0)
            metodo_prec = METODO_MARGINE["m"]
            try:
                for i_, nome_ in enumerate(scelti):
                    stato_.text(f"{nome_}: calcolo in corso ({i_ + 1} di {len(scelti)})...")
                    try:
                        r_ = esegui_test_campionato(LEAGUES[nome_], giorno)
                    except Exception as e_:
                        saltati_tutti.append(f"{nome_} (errore: {e_})")
                        r_ = None
                    if r_ is None:
                        if not any(sa_.startswith(nome_) for sa_ in saltati_tutti):
                            saltati_tutti.append(f"{nome_} (dati insufficienti)")
                    else:
                        risultati_tutti[nome_] = r_
                    barra_.progress((i_ + 1) / len(scelti))
            finally:
                METODO_MARGINE["m"] = metodo_prec
            stato_.empty()
            barra_.empty()
            st.session_state["tutti"] = (risultati_tutti, saltati_tutti, giorno)
    tut = st.session_state.get("tutti")
    if tut is not None and tut[2] == giorno:
        if tut[1]:
            st.warning("Non completati: " + "; ".join(tut[1]))
        if tut[0]:
            st.markdown("#### Riepilogo dei campionati")
            st.dataframe(riepilogo_campionati(tut[0]), use_container_width=True, hide_index=True)
            st.caption("Log loss: più basso è meglio; 'modello − mercato' positivo = il modello è peggiore delle quote di chiusura. "
                       "Per Goal e Over 2.5, 'prev. − reale' negativo = il programma sottostima. Il dettaglio completo di ogni campionato è nel file ZIP.")
            st.download_button("⬇ Scarica le tabelle di tutti i campionati (ZIP)", data=zip_test_campionati(tut[0]),
                               file_name=f"test_completo_{tut[2]}.zip", mime="application/zip")
            _pb = riepilogo_punteggi_bassi(tut[0])
            if not _pb.empty:
                st.markdown("#### 📉 Punteggi bassi nel tempo, campionati sommati")
                st.dataframe(_pb, use_container_width=True, hide_index=True)
                st.caption("0-0, 1-0, 0-1 e 1-1 insieme: reale contro previsto dal modello, con tutti i campionati scelti sommati. Solo un controllo. "
                           "Differenza positiva = il modello prevede più punteggi bassi del reale. Il dettaglio per campionato è nel file ZIP.")

    st.markdown("---")
    st.markdown("### 🕐 Gol per tempo, stagione per stagione")
    st.caption("Quante partite hanno 0, 1 o 2+ gol nel 1° e nel 2° tempo, per ogni stagione di test (le stesse usate dal Backtest). "
               "'± 2+ gol' è l'errore statistico: se le stagioni differiscono meno di circa 2 volte quell'errore, il campionato non è cambiato in modo misurabile "
               "e la media semplice delle stagioni va bene; se differiscono di più, ha senso dare più peso alle stagioni recenti. L'ultima stagione è parziale.")
    _df_tempi = carica_storico(div, N_STAGIONI_TEST, giorno)
    _tab_ts = tabella_tempi_per_stagione(_df_tempi)
    if _tab_ts.empty:
        st.info("Per questo campionato mancano i dati dei tempi (HTHG/HTAG) o sono troppo pochi.")
    else:
        st.dataframe(_tab_ts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ⚔️ Test: chi farà di più (tiri, tiri in porta, angoli, ammonizioni)")
    st.caption("Per ogni partita le medie attese usano solo i giorni precedenti; si confronta il modello con le sole frequenze di 'casa di più', 'pari' e 'trasferta di più'. "
               "Log score: più basso è meglio; differenza negativa = il modello è migliore.")
    if st.button("⚔️ Esegui test 'chi farà di più'", key="test_chi"):
        _df_chi = carica_storico(div, N_STAGIONI_TEST, giorno)
        _out_chi = test_chi_fa_di_piu(_df_chi) if _df_chi is not None else None
        if _out_chi is None:
            st.warning("Dati insufficienti (mancano le colonne di tiri, angoli o cartellini, o ci sono poche partite).")
        else:
            st.dataframe(_out_chi[0], use_container_width=True, hide_index=True)
            st.dataframe(_out_chi[1], use_container_width=True, hide_index=True)
            st.caption("La seconda tabella confronta, per fasce, la probabilità del pronostico con quanto è uscito davvero.")
    if st.button("🎚️ Taratura 'chi farà di più' (shrinkage e dispersione)", key="tara_chi"):
        _df_tara = carica_storico(div, N_STAGIONI_TEST, giorno)
        _tara = taratura_chi_fa_di_piu(_df_tara) if _df_tara is not None else None
        if _tara is None:
            st.warning("Dati insufficienti per la taratura.")
        else:
            st.dataframe(_tara, use_container_width=True, hide_index=True)
            st.caption("k = partite virtuali verso la media del campionato (attuale 5); fattore sulla dispersione (attuale 1.0). La differenza è dalla configurazione attuale: negativa = meglio. "
                       "Regola fissata in anticipo: si adotta una variante solo se migliora di almeno 2 errori standard in almeno 5 campionati su 6.")

    st.markdown("---")
    st.markdown("### 🧊 Test su stagioni mai viste")
    st.caption("Per ogni stagione (dalla terza in poi) i parametri del modello e i pesi tra modello e frequenze vengono scelti usando SOLO le stagioni precedenti; "
               "poi si misura su quella stagione, che non ha influito sulla scelta. È la verifica più onesta: se i numeri qui restano buoni, "
               "la scelta automatica non ti sta ingannando.")
    if st.button("🧊 Esegui test su stagioni nuove"):
        with st.spinner("Calcolo in corso (può richiedere alcuni minuti la prima volta)..."):
            st.session_state["sn"] = stagioni_nuove_cached(div, par["ridge"], par["halflife"], 14, giorno, VERSIONE_TEST)
            st.session_state["sn_div"] = div
    sn = st.session_state.get("sn") if st.session_state.get("sn_div") == div else None
    if sn is not None and sn[0] is not None:
        st.markdown("#### 📏 1X2: parametri scelti prima contro parametri attuali")
        st.dataframe(sn[0], use_container_width=True, hide_index=True)
        st.caption("Log loss, più basso è meglio. Se 'Scelti prima' è vicino a 'Attuali' (differenza più piccola di circa 2 volte l'errore), "
                   "la scelta automatica dei parametri non sta sovra-adattando. L'ultima stagione è quella in corso, con poche partite.")
        st.markdown("#### 🎯 Goal/No Goal, Over/Under e risultati esatti con pesi scelti sul passato")
        st.dataframe(sn[1], use_container_width=True, hide_index=True)
        st.caption("Se 'Misto − Frequenze' è negativo e più grande di circa 2 volte l'errore, il misto batte davvero le frequenze fuori campione.")
    elif sn is not None:
        st.warning("Dati insufficienti per questo test: servono almeno 3 stagioni.")

    st.markdown("---")
    st.markdown("### 🧭 Test ClubElo (rating Elo delle squadre)")
    st.caption("Aggiunge al modello la differenza di Elo tra le due squadre (dati gratuiti di clubelo.com, presi il giorno prima di ogni partita). "
               "Serve soprattutto per le squadre con pochi dati, come le neopromosse. Si tiene solo se migliora oltre l'errore. "
               "È un test: non cambia le previsioni dell'app.")
    if st.button("🧭 Esegui test ClubElo"):
        with st.spinner("Scarico gli Elo e calcolo (può richiedere alcuni minuti la prima volta)..."):
            st.session_state["elo"] = elo_test_cached(div, par["ridge"], par["halflife"], 14, giorno, VERSIONE_TEST)
            st.session_state["elo_div"] = div
    eo = st.session_state.get("elo") if st.session_state.get("elo_div") == div else None
    if eo is not None and "errore" in eo:
        st.warning(f"Test non eseguito: {eo['errore']}.")
        if eo.get("non_trovate"):
            st.caption("Squadre senza Elo (nomi diversi tra football-data e ClubElo): " + ", ".join(eo["non_trovate"]))
    elif eo is not None:
        e1, e2, e3 = st.columns(3)
        e1.metric("Squadre con Elo", f"{eo['n_squadre']}")
        e2.metric("Partite con Elo (copertura)", f"{eo['copertura']*100:.0f}%")
        e3.metric("Peso medio dell'Elo (β)", f"{eo['info']['beta']:.2f}")
        st.dataframe(eo["tab_a"], use_container_width=True, hide_index=True)
        st.caption("Log loss, più basso è meglio. Se la differenza 'Modello+Elo − Modello' è negativa e più grande di circa 2 volte l'errore, l'Elo migliora davvero.")
        if not eo["tab_b"].empty:
            st.markdown("#### 🆕 Squadre con poca storia contro tutte le altre")
            st.dataframe(eo["tab_b"], use_container_width=True, hide_index=True)
        if eo.get("non_trovate"):
            st.caption("Squadre senza Elo (escluse dal test): " + ", ".join(eo["non_trovate"]))

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


# ======================= COPPE EUROPEE (Champions / Europa League): SOLO PREVISIONE =======================
# Dati gratuiti (openfootball su GitHub): solo risultati, nessuna quota storica. La forza delle squadre viene dall'Elo
# (ClubElo, valore del giorno prima della partita) oppure da un Elo interno calcolato sui soli risultati di coppa.
COPPE = {"Champions League": {"file": "cl.txt", "da": 2018}, "Europa League": {"file": "el.txt", "da": 2020}}
_MESI_COPPE = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_RE_DATA_COPPA = re.compile(r"^\s{1,4}[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_RE_PARTITA_COPPA = re.compile(r"^\s+(?:\d{1,2}:\d{2}\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s*(.*?)\s*$")
_RE_FASE_COPPA = re.compile(r"^[▪•]\s*(.+?)\s*$")
_STOP_COPPE = {"club", "clube", "de", "del", "di", "da", "do", "la", "le", "the", "sk", "fk", "bsc", "bk", "kv", "if", "sc", "nk", "gnk", "pfc", "rb", "cd", "sad"}
_ALIAS_COPPE = {  # nome di coppa (senza parole generiche) -> nome ClubElo
    "internazionale milano": "Inter", "milan": "Milan", "paris saint germain": "Paris SG", "atletico madrid": "Atletico",
    "bayern munchen": "Bayern", "borussia dortmund": "Dortmund", "borussia monchengladbach": "Gladbach",
    "manchester city": "Man City", "manchester united": "Man United", "wolverhampton wanderers": "Wolves",
    "sporting braga": "Braga", "sporting portugal": "Sporting", "sport lisboa benfica": "Benfica", "dinamo kiev": "Dynamo Kyiv",
    "crvena zvezda": "Crvena Zvezda", "paok saloniki": "PAOK", "olympique marseille": "Marseille", "olympique lyonnais": "Lyon",
    "real sociedad futbol": "Sociedad", "athletic": "Athletic", "red bull salzburg": "Salzburg", "feyenoord rotterdam": "Feyenoord",
}


def _parse_coppa_txt(testo, stag):
    """Legge il formato Football.TXT di openfootball. Per le partite ai supplementari usa il punteggio dei 90 minuti."""
    righe, fase, data = [], "", None
    for linea in testo.replace("\r", "").split("\n"):
        mf = _RE_FASE_COPPA.match(linea)
        if mf:
            fase = mf.group(1)
            continue
        md = _RE_DATA_COPPA.match(linea)
        if md:
            mese, giorno_ = _MESI_COPPE.get(md.group(1)), int(md.group(2))
            if not mese:
                data = None
                continue
            anno = int(md.group(3)) if md.group(3) else (stag if mese >= 7 else stag + 1)
            try:
                data = pd.Timestamp(year=anno, month=mese, day=giorno_)
            except Exception:
                data = None
            continue
        mp = _RE_PARTITA_COPPA.match(linea)
        if not mp or data is None:
            continue
        casa, pc, trasf, pt, resto = mp.groups()
        ft = ht = None
        m_et = re.search(r"(\d+)-(\d+)\s+a\.e\.t\.\s*\((\d+)-(\d+),\s*(\d+)-(\d+)\)", resto)
        if m_et:
            ft, ht = (int(m_et.group(3)), int(m_et.group(4))), (int(m_et.group(5)), int(m_et.group(6)))
        elif "a.e.t" not in resto:
            m1 = re.match(r"(\d+)-(\d+)(?:\s*\((\d+)-(\d+)\))?", resto)
            if m1:
                ft = (int(m1.group(1)), int(m1.group(2)))
                ht = (int(m1.group(3)), int(m1.group(4))) if m1.group(3) is not None else None
        righe.append({"Date": data, "HomeTeam": casa.strip(), "AwayTeam": trasf.strip(), "PaeseH": pc, "PaeseA": pt,
                      "FTHG": ft[0] if ft else np.nan, "FTAG": ft[1] if ft else np.nan,
                      "HTHG": ht[0] if ht else np.nan, "HTAG": ht[1] if ht else np.nan, "Stag": stag, "Fase": fase})
    return pd.DataFrame(righe)


def _fase_neutra(fase):
    f = str(fase).strip().lower()
    return float(f.endswith("final") and "semi" not in f and "quarter" not in f)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def carica_storico_coppa(nome_coppa, giorno):
    """Ritorna (DataFrame, stagioni non disponibili). Le partite non ancora giocate hanno FTHG/FTAG vuoti."""
    info = COPPE[nome_coppa]
    oggi = date.today()
    anno_corr = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames, mancanti = [], []
    for y in range(info["da"], anno_corr + 1):
        slug = f"{y}-{str(y + 1)[2:]}"
        url = f"https://raw.githubusercontent.com/openfootball/champions-league/master/{slug}/{info['file']}"
        try:
            res = requests.get(url, timeout=20, headers=UA)
        except Exception as e:
            mancanti.append(f"{slug} ({type(e).__name__})")
            continue
        if res.status_code != 200 or len(res.text) < 200:
            mancanti.append(f"{slug} (HTTP {res.status_code})")
            continue
        d = _parse_coppa_txt(res.text, y)
        if not d.empty:
            frames.append(d)
    if not frames:
        return None, mancanti
    df = pd.concat(frames, ignore_index=True)
    # stessa squadra con nomi diversi tra stagioni (es. "Bayern München" / "FC Bayern München"): si usa il nome più recente
    nomi = pd.concat([df[["HomeTeam", "PaeseH", "Date"]].set_axis(["N", "P", "D"], axis=1), df[["AwayTeam", "PaeseA", "Date"]].set_axis(["N", "P", "D"], axis=1)])
    nomi["K"] = nomi["P"] + "|" + nomi["N"].map(lambda n: " ".join(_token_coppa(n)))
    recente = nomi.sort_values("D").drop_duplicates("K", keep="last").set_index("K")["N"]
    mappa = {(p, n): recente[k] for p, n, k in zip(nomi["P"], nomi["N"], nomi["K"])}
    df["HomeTeam"] = [mappa[(p, n)] for p, n in zip(df["PaeseH"], df["HomeTeam"])]
    df["AwayTeam"] = [mappa[(p, n)] for p, n in zip(df["PaeseA"], df["AwayTeam"])]
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").sort_values("Date").reset_index(drop=True)
    df["Neutro"] = df["Fase"].map(_fase_neutra)
    return df, mancanti


def elo_interno_coppe(df, k=20.0, vantaggio=60.0, base=1500.0):
    """Elo calcolato solo sui risultati di coppa (si parte tutti da 1500): riferimento senza ClubElo. Ritorna (df con EloH/EloA/EloDiff, rating finali)."""
    rating, eh, ea = {}, [], []
    for r in df.itertuples():
        a, b = rating.get(r.HomeTeam, base), rating.get(r.AwayTeam, base)
        eh.append(a)
        ea.append(b)
        if pd.isna(r.FTHG):
            continue
        e_casa = 1 / (1 + 10 ** (-(a + (0.0 if r.Neutro else vantaggio) - b) / 400))
        s = 1.0 if r.FTHG > r.FTAG else (0.5 if r.FTHG == r.FTAG else 0.0)
        diff = abs(r.FTHG - r.FTAG)
        mult = 1.0 if diff <= 1 else (1.5 if diff == 2 else (11 + diff) / 8)
        delta = k * mult * (s - e_casa)
        rating[r.HomeTeam], rating[r.AwayTeam] = a + delta, b - delta
    d = df.copy()
    d["EloH"], d["EloA"] = eh, ea
    d["EloDiff"] = (d["EloH"] - d["EloA"]) / 400.0
    return d, rating


def _token_coppa(n):
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return [t for t in n.split() if t not in _STOP_NOMI and t not in _STOP_COPPE and not t.isdigit()]


def _sottoinsieme_prefissi(corti, lunghi):
    return bool(corti) and all(any(t2.startswith(t1) and len(t1) >= 3 or t1 == t2 for t2 in lunghi) for t1 in corti)


def _abbina_uno_coppa(nome, candidati):
    """Trova il club ClubElo più simile (tra i candidati). Ritorna (nome ClubElo, punteggio 0-1)."""
    tok = _token_coppa(nome)
    chiave = " ".join(tok)
    if chiave in _ALIAS_COPPE:
        voluto = _compatta_elo(_ALIAS_COPPE[chiave])
        for c in candidati:
            if _compatta_elo(c) == voluto:
                return c, 1.0
    cn = _compatta_elo(nome)
    best, bs = None, 0.0
    for c in candidati:
        tc = _token_coppa(c)
        sc = _sim_elo(cn, _compatta_elo(c))
        if _sottoinsieme_prefissi(tc, tok) or _sottoinsieme_prefissi(tok, tc):
            sc = max(sc, 0.95)
        if sc > bs:
            best, bs = c, sc
    return best, bs


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def carica_elo_coppe(squadre_paesi, giorno):
    """ClubElo (gratuito) per tutte le squadre delle coppe. Ritorna (serie per squadra, tabella degli abbinamenti, errore)."""
    base_ok = {"b": None}

    def leggi(percorso):
        err_ = None
        for base in ([base_ok["b"]] if base_ok["b"] else ELO_BASI):
            d_, err_ = _leggi_csv(f"{base}/{percorso}")
            if d_ is not None:
                base_ok["b"] = base
                return d_, None
        return None, err_

    snaps, err = [], None
    for g in (giorno, "2021-01-01"):
        s, e = leggi(g)
        if s is not None and {"Club", "Country"}.issubset(s.columns):
            snaps.append(s[["Club", "Country"]])
        else:
            err = e
    if not snaps:
        return {}, pd.DataFrame(), err or "formato inatteso"
    club = pd.concat(snaps).drop_duplicates()
    abb, righe = {}, []
    for sq, paese in squadre_paesi:
        cand = club.loc[club["Country"] == paese, "Club"].astype(str).unique().tolist() or club["Club"].astype(str).unique().tolist()
        best, sc = _abbina_uno_coppa(sq, cand)
        ok = best is not None and sc >= 0.88
        righe.append({"Squadra": sq, "Paese": paese, "Club ClubElo": best if ok else None, "Somiglianza": round(float(sc), 2)})
        if ok:
            abb[sq] = best

    def scarica(item):
        sq, nome = item
        d, _ = leggi(nome)
        if d is None or not {"Elo", "From", "To"}.issubset(d.columns):
            return sq, None
        d["From"] = pd.to_datetime(d["From"], errors="coerce")
        d["To"] = pd.to_datetime(d["To"], errors="coerce")
        d = d.dropna(subset=["From", "To", "Elo"]).sort_values("From")
        return sq, (d[["From", "To", "Elo"]].reset_index(drop=True) if not d.empty else None)

    serie = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sq, srs in ex.map(scarica, list(abb.items())):
            if srs is not None:
                serie[sq] = srs
    return serie, pd.DataFrame(righe), None


def _tau_vett(x, y, lh, la, rho):
    t = np.ones_like(lh)
    m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
    t[m00] = 1 - lh[m00] * la[m00] * rho
    t[m01] = 1 + lh[m01] * rho
    t[m10] = 1 + la[m10] * rho
    t[m11] = 1 - rho
    return np.maximum(t, 1e-6)


def adatta_coppe(df, senza_elo=False):
    """Modello a 4 parametri: log λ_casa = μ + vantaggio·(non neutro) + β·d, log λ_trasf = μ − β·d, con d = differenza Elo / 400,
    più il rho di Dixon-Coles. Stima per massima verosimiglianza. senza_elo=True blocca β a 0 (solo medie di coppa)."""
    x, y = df["FTHG"].to_numpy(float), df["FTAG"].to_numpy(float)
    d = df["EloDiff"].to_numpy(float)
    casa = 1.0 - df["Neutro"].to_numpy(float) if "Neutro" in df.columns else np.ones(len(df))
    gx, gy = gammaln(x + 1), gammaln(y + 1)

    def nll(p):
        mu, ha, beta, rho = p
        lh, la = np.exp(mu + ha * casa + beta * d), np.exp(mu - beta * d)
        ll = x * np.log(lh) - lh - gx + y * np.log(la) - la - gy + np.log(_tau_vett(x, y, lh, la, rho))
        return -float(ll.sum())
    limiti = [(-2, 2), (-1, 1), (0, 0) if senza_elo else (-1, 3), (-0.25, 0.25)]
    r = minimize(nll, x0=[0.2, 0.2, 0.0 if senza_elo else 0.4, 0.0], method="L-BFGS-B", bounds=limiti)
    mu, ha, beta, rho = (float(v) for v in r.x)
    return {"mu": mu, "ha": ha, "beta": beta, "rho": rho, "n": int(len(df))}


def lambda_coppe(m, elo_diff, neutro=0.0):
    return math.exp(m["mu"] + m["ha"] * (1 - neutro) + m["beta"] * elo_diff), math.exp(m["mu"] - m["beta"] * elo_diff)


def test_coppe(df, passo=30):
    """Walk-forward: ogni blocco di `passo` giorni è previsto con un modello stimato solo sulle partite precedenti (partendo da un anno di storico).
    Confronta: Frequenze, Solo medie (senza Elo) e Modello con Elo. Ritorna (tabella log score, tabella calibrazione, parametri finali, n partite) o None."""
    d = df.dropna(subset=["FTHG", "FTAG", "EloDiff"]).sort_values("Date").reset_index(drop=True)
    if len(d) < 300:
        return None
    t, fine = d["Date"].min() + pd.Timedelta(days=365), d["Date"].max()
    rec = {k: [] for k in ("x", "y", "pe", "pm", "ps", "fe")}
    P = {"Modello": [], "Solo medie": [], "Frequenze": []}
    mk_nomi = ("1X2", "Over 2.5", "Goal", "Risultato esatto")
    L = {v: {m: [] for m in mk_nomi} for v in P}
    cal = {m: [] for m in ("1", "X", "2", "Over 2.5", "Goal")}
    while t <= fine:
        blocco = d[(d["Date"] >= t) & (d["Date"] < t + pd.Timedelta(days=passo))]
        train = d[d["Date"] < t]
        if len(blocco) and len(train) >= 150:
            mm, ms = adatta_coppe(train), adatta_coppe(train, senza_elo=True)
            esiti_tr = np.where(train["FTHG"] > train["FTAG"], 0, np.where(train["FTHG"] == train["FTAG"], 1, 2))
            f1x2 = np.bincount(esiti_tr, minlength=3) / len(esiti_tr)
            fo25 = float(((train["FTHG"] + train["FTAG"]) >= 3).mean())
            fg = float(((train["FTHG"] > 0) & (train["FTAG"] > 0)).mean())
            cm = np.full((11, 11), 0.5)
            np.add.at(cm, (np.minimum(train["FTHG"].astype(int), 10), np.minimum(train["FTAG"].astype(int), 10)), 1)
            cm /= cm.sum()
            for r in blocco.itertuples():
                xg, yg = int(min(r.FTHG, 10)), int(min(r.FTAG, 10))
                esito = 0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2)
                o25, bt = int(r.FTHG + r.FTAG >= 3), int(r.FTHG > 0 and r.FTAG > 0)
                for nome, mod in (("Modello", mm), ("Solo medie", ms)):
                    lh, la = lambda_coppe(mod, r.EloDiff, r.Neutro)
                    M = matrice_punteggi(lh, la, mod["rho"])
                    mk = mercati(M)
                    p3 = [mk["p1"], mk["px"], mk["p2"]]
                    L[nome]["1X2"].append(-math.log(max(p3[esito], 1e-9)))
                    L[nome]["Over 2.5"].append(-math.log(max(mk["over2.5"] if o25 else 1 - mk["over2.5"], 1e-9)))
                    L[nome]["Goal"].append(-math.log(max(mk["btts"] if bt else 1 - mk["btts"], 1e-9)))
                    L[nome]["Risultato esatto"].append(-math.log(max(M[xg, yg], 1e-9)))
                    if nome == "Modello":
                        for k_, v_, a_ in (("1", p3[0], esito == 0), ("X", p3[1], esito == 1), ("2", p3[2], esito == 2),
                                           ("Over 2.5", mk["over2.5"], o25), ("Goal", mk["btts"], bt)):
                            cal[k_].append((v_, float(a_)))
                L["Frequenze"]["1X2"].append(-math.log(max(f1x2[esito], 1e-9)))
                L["Frequenze"]["Over 2.5"].append(-math.log(max(fo25 if o25 else 1 - fo25, 1e-9)))
                L["Frequenze"]["Goal"].append(-math.log(max(fg if bt else 1 - fg, 1e-9)))
                L["Frequenze"]["Risultato esatto"].append(-math.log(max(cm[xg, yg], 1e-9)))
        t += pd.Timedelta(days=passo)
    n = len(L["Modello"]["1X2"])
    if n < 100:
        return None
    righe = []
    for m in mk_nomi:
        a, b, c = np.array(L["Modello"][m]), np.array(L["Solo medie"][m]), np.array(L["Frequenze"][m])
        righe.append({"Mercato": m, "Partite": n, "Frequenze": round(float(c.mean()), 4), "Solo medie (senza Elo)": round(float(b.mean()), 4),
                      "Modello con Elo": round(float(a.mean()), 4), "Modello − Frequenze (± err.)": _diff_se(a, c),
                      "Modello − Solo medie (± err.)": _diff_se(a, b)})
    cal_righe = []
    for k_, v_ in cal.items():
        if v_:
            pr, ac = np.mean([a for a, _ in v_]) * 100, np.mean([b for _, b in v_]) * 100
            cal_righe.append({"Mercato": k_, "Previsto %": round(float(pr), 1), "Reale %": round(float(ac), 1), "Differenza (punti)": round(float(pr - ac), 1)})
    return pd.DataFrame(righe), pd.DataFrame(cal_righe), adatta_coppe(d), n


API_FOOTBALL_LEGA = {"Champions League": 2, "Europa League": 3}  # codici di API-Football; il nome restituito dall'API conferma che sono giusti


def api_football_get(chiave, percorso, params=None):
    """Una chiamata a API-Football. Ritorna (json, intestazioni sui limiti, errore). La chiave non viene salvata né mostrata."""
    try:
        r = requests.get("https://v3.football.api-sports.io/" + percorso.lstrip("/"), headers={"x-apisports-key": chiave}, params=params or {}, timeout=20)
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"
    lim = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
    try:
        js = r.json()
    except Exception:
        return None, lim, f"HTTP {r.status_code}: risposta non valida"
    return js, lim, (None if r.status_code == 200 else f"HTTP {r.status_code}")


def diagnosi_api_football(chiave, id_lega, anno):
    """Verifica a basso costo (5 richieste): stato dell'account, copertura della competizione, prossime partite, quote di una partita e tipi di scommessa."""
    out = {"errori": [], "stato": {}, "lega": {}, "copertura": {}, "partite": [], "bookmakers": [], "scommesse_partita": [], "tipi_scommessa": [], "log": []}

    def passo(nome, percorso, params=None):
        js, lim, err = api_football_get(chiave, percorso, params)
        errs = js.get("errors") if isinstance(js, dict) else None
        if errs:
            out["errori"] += [f"{nome}: {k}: {v}" for k, v in (errs.items() if isinstance(errs, dict) else enumerate(errs))]
        if err:
            out["errori"].append(f"{nome}: {err}")
        out["log"].append({"passo": nome, "percorso": percorso, "parametri": params or {}, "risultati": js.get("results") if isinstance(js, dict) else None, "limiti": lim})
        return js if isinstance(js, dict) else {}

    js = passo("stato", "status")
    resp = js.get("response")
    if isinstance(resp, dict):
        out["stato"] = {"piano": (resp.get("subscription") or {}).get("plan"), "attivo": (resp.get("subscription") or {}).get("active"),
                        "richieste_oggi": (resp.get("requests") or {}).get("current"), "limite_giornaliero": (resp.get("requests") or {}).get("limit_day")}
    js = passo("copertura", "leagues", {"id": id_lega, "season": anno})
    for it in (js.get("response") or [])[:1]:
        out["lega"] = {"nome": (it.get("league") or {}).get("name"), "paese": (it.get("country") or {}).get("name")}
        for sz in it.get("seasons") or []:
            if sz.get("year") == anno:
                out["copertura"] = {k: v for k, v in (sz.get("coverage") or {}).items() if not isinstance(v, dict)}
                out["copertura"].update({f"fixtures.{k}": v for k, v in ((sz.get("coverage") or {}).get("fixtures") or {}).items()})
    js = passo("prossime partite", "fixtures", {"league": id_lega, "season": anno, "next": 10})
    ids = []
    for it in js.get("response") or []:
        fx_ = it.get("fixture") or {}
        tm = it.get("teams") or {}
        out["partite"].append({"id": fx_.get("id"), "data": fx_.get("date"), "casa": (tm.get("home") or {}).get("name"), "trasferta": (tm.get("away") or {}).get("name")})
        ids.append(fx_.get("id"))
    if ids:
        js = passo("quote di una partita", "odds", {"fixture": ids[0]})
        nomi = {}
        for it in (js.get("response") or [])[:1]:
            for bk in it.get("bookmakers") or []:
                out["bookmakers"].append({"bookmaker": bk.get("name"), "tipi di scommessa": len(bk.get("bets") or [])})
                for b in bk.get("bets") or []:
                    nomi[b.get("name")] = nomi.get(b.get("name"), 0) + 1
        out["scommesse_partita"] = [{"scommessa": k, "bookmaker che la offrono": v} for k, v in sorted(nomi.items(), key=lambda kv: -kv[1])]
    js = passo("tipi di scommessa", "odds/bets")
    out["tipi_scommessa"] = [b.get("name") for b in (js.get("response") or []) if b.get("name")]
    return out


# ======================= TAB COPPE EUROPEE =======================
with tab_coppe:
    st.markdown("### 🌍 Coppe europee — solo previsione")
    st.caption("Dati gratuiti di openfootball (solo risultati: non ci sono quote storiche). La forza delle squadre viene dall'Elo; il modello stima da sole le partite di coppa "
               "quanti gol attendersi in base alla differenza di Elo. Non è confrontato col mercato: va letto come una stima, non come un valore.")
    coppa = st.selectbox("Competizione", list(COPPE.keys()), key="coppa_sel")
    with st.expander("🔌 API-Football: verifica chiave, quote e calendario"):
        if not chiave_apif:
            st.info("Inserisci la chiave API-Football nella barra laterale (gratis su api-football.com, in Account → My Access).")
        else:
            st.caption("Fa 5 richieste (nel piano gratuito ne hai 100 al giorno): stato dell'account, copertura della competizione scelta, prossime partite, quote di una partita e tipi di scommessa. "
                       "La chiave non viene salvata né mostrata, e non finisce nel file scaricabile.")
            if st.button("🔌 Verifica chiave e copertura", key="apif_verifica"):
                _oggi = date.today()
                with st.spinner("Controllo in corso..."):
                    dg = diagnosi_api_football(chiave_apif, API_FOOTBALL_LEGA[coppa], _oggi.year if _oggi.month >= 7 else _oggi.year - 1)
                if dg["errori"]:
                    st.warning("L'API ha risposto con questi avvisi (di solito dicono cosa non è incluso nel piano):\n\n" + "\n\n".join(f"- {e}" for e in dg["errori"]))
                if dg["stato"]:
                    st.write(f"**Account:** piano {dg['stato'].get('piano')}, richieste oggi {dg['stato'].get('richieste_oggi')} su {dg['stato'].get('limite_giornaliero')}.")
                if dg["lega"]:
                    st.write(f"**Competizione trovata:** {dg['lega'].get('nome')} ({dg['lega'].get('paese')}) — controlla che sia quella giusta.")
                if dg["copertura"]:
                    st.write("**Copertura della stagione in corso:**")
                    st.dataframe(pd.DataFrame([{"Dato": k, "Disponibile": "sì" if v else "no"} for k, v in dg["copertura"].items()]), use_container_width=True, hide_index=True)
                if dg["partite"]:
                    st.write("**Prossime partite:**")
                    st.dataframe(pd.DataFrame(dg["partite"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("Nessuna partita futura restituita per questa stagione.")
                if dg["bookmakers"]:
                    st.write("**Quote della prima partita, per bookmaker:**")
                    st.dataframe(pd.DataFrame(dg["bookmakers"]), use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame(dg["scommesse_partita"]), use_container_width=True, hide_index=True)
                if dg["tipi_scommessa"]:
                    _kw = ("both", "goal", "over", "under", "card", "scorer", "exact", "half", "double", "corner")
                    with st.expander(f"Tutti i tipi di scommessa dell'API ({len(dg['tipi_scommessa'])})"):
                        st.write(", ".join(dg["tipi_scommessa"]))
                    st.caption("Tipi che ci interessano: " + ", ".join(t for t in dg["tipi_scommessa"] if any(k in t.lower() for k in _kw)))
                st.download_button("⬇ Scarica la diagnosi (JSON, senza la chiave)", data=json.dumps(dg, indent=2, ensure_ascii=False, default=str),
                                   file_name=f"diagnosi_api_football_{giorno}.json", mime="application/json", key="apif_dl")
    df_c, mancanti_c = carica_storico_coppa(coppa, giorno)
    if df_c is None:
        st.error("Non riesco a scaricare i dati delle coppe (openfootball). Riprova tra poco.")
    else:
        giocate_c = df_c[df_c["FTHG"].notna()]
        st.info(f"{len(giocate_c)} partite giocate, dal {giocate_c['Date'].min():%d/%m/%Y} al {giocate_c['Date'].max():%d/%m/%Y}.")
        if mancanti_c:
            st.caption("Stagioni non disponibili nella fonte: " + ", ".join(mancanti_c))
        fonte_elo_c = st.radio("Forza delle squadre", ["Elo interno (solo risultati di coppa)", "ClubElo (scarica i dati: qualche minuto la prima volta)"], horizontal=True, key="coppa_fonte",
                                 help="Il test a ritroso dice quale delle due funziona meglio. ClubElo conosce anche la forza nei campionati nazionali, ma richiede il download di un file per ogni club.")
        elo_att_c, abb_c, d_c = {}, pd.DataFrame(), None
        if fonte_elo_c.startswith("ClubElo"):
            sp_c = tuple(sorted(set(zip(df_c["HomeTeam"], df_c["PaeseH"])) | set(zip(df_c["AwayTeam"], df_c["PaeseA"]))))
            with st.spinner("Scarico l'Elo di ogni club (solo la prima volta di giornata, può richiedere un paio di minuti)..."):
                serie_c, abb_c, err_c = carica_elo_coppe(sp_c, giorno)
            if err_c or not serie_c:
                st.warning(f"ClubElo non disponibile ({err_c or 'nessun club abbinato'}): uso l'Elo interno.")
                fonte_elo_c = "Elo interno (solo risultati di coppa)"
            else:
                d_c = aggiungi_elo(df_c, serie_c)
                elo_att_c = {sq_: float(s_["Elo"].iloc[-1]) for sq_, s_ in serie_c.items()}
                cop_c = float(d_c.loc[d_c["FTHG"].notna(), "EloDiff"].notna().mean()) * 100
                st.caption(f"Elo disponibile per il {cop_c:.0f}% delle partite giocate ({len(serie_c)} club su {len(sp_c)}).")
                if cop_c < 70:
                    st.warning("Meno del 70% delle partite ha l'Elo di entrambe le squadre: i risultati del test sono meno affidabili.")
                if not abb_c.empty:
                    with st.expander("Abbinamenti squadra di coppa → club ClubElo (controlla i dubbi)"):
                        st.dataframe(abb_c.sort_values("Somiglianza"), use_container_width=True, hide_index=True)
        if fonte_elo_c.startswith("Elo interno"):
            d_c, rating_c = elo_interno_coppe(df_c)
            elo_att_c = dict(rating_c)
        base_c = d_c[d_c["FTHG"].notna()].dropna(subset=["EloDiff"])

        st.markdown("#### 🧪 Test a ritroso")
        st.caption("Ogni blocco di 30 giorni è previsto con un modello stimato solo sulle partite precedenti (si parte da un anno di storico). "
                   "Si confronta con le sole frequenze e con un modello senza Elo. Log score: più basso è meglio; differenza negativa = il modello è migliore.")
        if st.button("🧪 Esegui test coppe", key="coppa_test"):
            out_c = test_coppe(d_c)
            if out_c is None:
                st.warning("Troppe poche partite con l'Elo per fare il test.")
            else:
                tab_c, cal_c, m_c, n_c = out_c
                st.dataframe(tab_c, use_container_width=True, hide_index=True)
                st.dataframe(cal_c, use_container_width=True, hide_index=True)
                st.caption(f"{n_c} partite previste. Parametri sull'intero storico: gol di base {math.exp(m_c['mu']):.2f}, vantaggio casa {m_c['ha']:+.2f}, "
                           f"peso dell'Elo {m_c['beta']:.2f} (per 400 punti), rho {m_c['rho']:+.3f}. "
                           "Nella calibrazione, 'Differenza' è previsto meno reale.")
                st.download_button("⬇ Scarica le tabelle (CSV)", data=pd.concat([tab_c.assign(Tabella="log score"), cal_c.assign(Tabella="calibrazione")]).to_csv(index=False),
                                   file_name=f"test_coppe_{coppa.split()[0].lower()}_{giorno}.csv", mime="text/csv", key="coppa_dl")

        st.markdown("#### 🔮 Previsione di una partita")
        if len(base_c) < 150 or not elo_att_c:
            st.info("Dati insufficienti per stimare il modello.")
        else:
            recenti_c = set(df_c.loc[df_c["Date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=730), ["HomeTeam", "AwayTeam"]].to_numpy().ravel())
            squadre_c = sorted(t for t in elo_att_c if t in recenti_c)
            if len(squadre_c) < 2:
                st.info("Nessuna squadra con Elo tra quelle recenti.")
            else:
                cc1, cc2 = st.columns(2)
                casa_c = cc1.selectbox("Casa", squadre_c, key="coppa_casa")
                trasf_c = cc2.selectbox("Trasferta", [t for t in squadre_c if t != casa_c], key="coppa_trasf")
                neutro_c = st.checkbox("Campo neutro (finale)", value=False, key="coppa_neutro")
                m_fin = adatta_coppe(base_c)
                diff_c = (elo_att_c[casa_c] - elo_att_c[trasf_c]) / 400.0
                lh_c, la_c = lambda_coppe(m_fin, diff_c, float(neutro_c))
                M_c = matrice_punteggi(lh_c, la_c, m_fin["rho"])
                mk_c = mercati(M_c)
                st.markdown(f"**{casa_c} vs {trasf_c}** — Elo {elo_att_c[casa_c]:.0f} contro {elo_att_c[trasf_c]:.0f}, gol attesi {lh_c:.2f} - {la_c:.2f}")
                q1, q2, q3 = st.columns(3)
                q1.metric("1 (Casa)", f"{mk_c['p1'] * 100:.1f}%", f"quota equa {1 / mk_c['p1']:.2f}", delta_color="off")
                q2.metric("X", f"{mk_c['px'] * 100:.1f}%", f"quota equa {1 / mk_c['px']:.2f}", delta_color="off")
                q3.metric("2 (Trasferta)", f"{mk_c['p2'] * 100:.1f}%", f"quota equa {1 / mk_c['p2']:.2f}", delta_color="off")
                righe_c = [{"Mercato": "Goal", "Probabilità %": round(mk_c["btts"] * 100, 1)}, {"Mercato": "No Goal", "Probabilità %": round((1 - mk_c["btts"]) * 100, 1)}]
                for ln in (1.5, 2.5, 3.5):
                    righe_c.append({"Mercato": f"Over {ln}", "Probabilità %": round(mk_c[f"over{ln}"] * 100, 1)})
                    righe_c.append({"Mercato": f"Under {ln}", "Probabilità %": round((1 - mk_c[f"over{ln}"]) * 100, 1)})
                st.dataframe(pd.DataFrame(righe_c), use_container_width=True, hide_index=True)
                top_c = sorted(((i, j, M_c[i, j]) for i in range(6) for j in range(6)), key=lambda z: -z[2])[:5]
                st.dataframe(pd.DataFrame([{"Risultato": f"{i}-{j}", "Probabilità %": round(p * 100, 1)} for i, j, p in top_c]), use_container_width=True, hide_index=True)
                st.caption("Solo previsione: non ci sono quote storiche, quindi non si può dire se una percentuale conviene. Nel test, di solito la parte più utile è l'1X2 e il risultato esatto; "
                           "Goal e Over/Under spesso non migliorano rispetto alle frequenze.")
        fut_c = df_c[df_c["FTHG"].isna() & (df_c["Date"] >= pd.Timestamp(date.today()))]
        if not fut_c.empty:
            with st.expander("Partite ancora da giocare nella fonte"):
                st.dataframe(fut_c[["Date", "HomeTeam", "AwayTeam", "Fase"]], use_container_width=True, hide_index=True)
        else:
            st.caption("La fonte non ha ancora il calendario della stagione in corso: scegli le due squadre a mano.")

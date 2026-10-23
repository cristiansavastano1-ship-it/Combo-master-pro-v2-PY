import difflib
import io
import time
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
from scipy.special import gammaln

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
VERSIONE_TEST = 6  # cambia quando cambia la struttura dei risultati del test (invalida la cache)
HALFLIFE_SPLIT = 365  # emivita fissa (giorni) per la ripartizione dei gol tra i tempi, indipendente da quella del modello (esperimento)
N_STAGIONI_TEST = 5  # il test usa più stagioni del modello per avere più partite su cui misurare
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FINESTRA_GIORNI = 1000  # finestra di addestramento (circa 3 stagioni), come nell'app live
COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
COL_OU_PRESA = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")]
COL_OU_CHIUSURA = [("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5")]
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
            S_blocco = tabella_split(train, data_rif=t, halflife=HALFLIFE_SPLIT)
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


def tabella_split(df, s=10.0, data_rif=None, halflife=None):
    """Come si dividono davvero i gol tra 1° e 2° tempo, dato il totale della partita (frequenze storiche, lisciate verso la divisione binomiale).
    S[t][k] = probabilità che in una partita con t gol ne siano stati segnati k nel 1° tempo.
    Se data_rif e halflife sono dati, le partite pesano 0.5^(età/halflife) (come nel modello principale); altrimenti pesano tutte 1."""
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
        if data_rif is not None and halflife and halflife > 0:
            eta_gg = (data_rif - d["Date"]).dt.days.clip(lower=0).to_numpy(float)
            w = np.exp(-np.log(2) * eta_gg / halflife)
        else:
            w = np.ones(len(d))
        pi = float(min(max((w * h1).sum() / max((w * ft).sum(), 1), 0.3), 0.6))
    for t in range(11):
        prior = np.array([math.comb(t, k) * pi ** k * (1 - pi) ** (t - k) for k in range(t + 1)])
        if d is None:
            S[t, :t + 1] = prior
            continue
        m = ft == t
        counts = np.bincount(np.clip(h1[m], 0, t), weights=w[m], minlength=t + 1)[:t + 1].astype(float)
        S[t, :t + 1] = (counts + s * prior) / (w[m].sum() + s)
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


def tabelle_tempi(M_gol, S, freq=None):
    p1, p2, pm = probs_tempi(M_gol, S)
    out = {}

    def righe(p, nome):
        rows = [_riga_prob(f"{nome} {a}-{b} gol", p[a:b + 1].sum(), freq) for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]]
        rows.append(_riga_prob(f"{nome} almeno 1 gol", 1 - p[0], freq))
        rows.append(_riga_prob(f"{nome} almeno 2 gol", 1 - p[0] - p[1], freq))
        return pd.DataFrame(rows)
    out["t1"], out["t2"] = righe(p1, "1° tempo"), righe(p2, "2° tempo")
    out["piu_gol"] = pd.DataFrame([_riga_prob("Primo tempo più gol", np.tril(pm, -1).sum(), freq),
                                   _riga_prob("Gol uguali nei due tempi", np.trace(pm), freq),
                                   _riga_prob("Secondo tempo più gol", np.triu(pm, 1).sum(), freq)])
    for chiave, p in (("t1", p1), ("t2", p2)):
        w2 = {f"{a}-{b}": float(p[a:b + 1].sum()) for a, b in [(0, 1), (1, 2), (2, 3)]}
        out[chiave + "_best"] = max(w2, key=w2.get)
        out[chiave + "_best_p"] = w2[out[chiave + "_best"]]
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
        out[chiave + "_best"] = max(w2, key=w2.get)
        out[chiave + "_best_p"] = w2[out[chiave + "_best"]]
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
split_live = tabella_split(df_storico, data_rif=oggi_ts, halflife=HALFLIFE_SPLIT)
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

tab_calendario, tab_pronostici, tab_classifica, tab_value_finder, tab_ai_schedine, tab_monte_carlo, tab_backtest, tab_kelly = st.tabs([
    "📅 Calendario", "📋 Pronostici", "🏆 Classifica", "⚡ Value Finder", "🤖 Schedine AI", "🎲 Monte Carlo", "🧪 Backtest", "🧮 Value Bet",
])


def mostra_analisi(casa, trasf, data, ora):
    d = distribuzione(casa, trasf)
    lh, la = gol_attesi(d["M_gol"])
    M = d["M_es"]
    mk = mercati(d["M_1x2"])
    mg = mercati(d["M_gol"])
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
    st.caption(f"Fonte Goal / Over-Under / Risultati esatti: {d['fonte']}.")
    if not d["ancorata"]:
        st.warning("⚠ Senza le quote del bookmaker, Goal/No Goal e Over/Under hanno poco potere predittivo: nel backtest su circa 1200 partite "
                   "non hanno battuto in modo netto le semplici frequenze storiche. Considerali indicativi, vicini alla media del campionato.")
    soglia_x = pesi.get("soglia_x", {}).get("ancorato" if d["ancorata"] else "modello")
    pr = tabelle_pronostico(d["M_1x2"], d["M_gol"], freq_combo, soglia_x)
    pt = tabelle_tempi(d["M_gol"], split_live, freq_combo)
    ps = tabelle_squadre(d["M_gol"], freq_combo)
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
    u2_.metric("Multigol casa più probabile (2 gol)", ps["casa_best"] + " gol", f"{ps['casa_best_p']*100:.1f}%", delta_color="off")
    u3_.metric("Multigol trasferta più probabile (2 gol)", ps["trasf_best"] + " gol", f"{ps['trasf_best_p']*100:.1f}%", delta_color="off")
    v1_, v2_, v3_ = st.columns(3)
    v1_.metric("Multigol 1° tempo più probabile (2 gol)", pt["t1_best"] + " gol", f"{pt['t1_best_p']*100:.1f}%", delta_color="off")
    v2_.metric("Multigol 2° tempo più probabile (2 gol)", pt["t2_best"] + " gol", f"{pt['t2_best_p']*100:.1f}%", delta_color="off")
    v3_.metric("Tempo con più gol", pt["piu_best"], f"{pt['piu_best_p']*100:.1f}%", delta_color="off")
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

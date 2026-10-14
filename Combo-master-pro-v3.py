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
VERSIONE_TEST = 3  # cambia quando cambia la struttura dei risultati del test (invalida la cache)
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


def senza_margine(quote):
    p = 1.0 / np.asarray(quote, dtype=float)
    return p / p.sum()


def _leggi_csv(url):
    """Ritorna (DataFrame, None) oppure (None, motivo). Gestisce il BOM UTF-8 che football-data.co.uk mette in fixtures.csv."""
    try:
        res = requests.get(url, timeout=20, headers=UA)
    except Exception as e:
        return None, f"connessione fallita ({type(e).__name__})"
    if res.status_code != 200:
        return None, f"risposta HTTP {res.status_code}"
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


def esegui_test_completo(df, ridge, halflife, passo, solo_base=False, inizio=None):
    """Walk-forward come il backtest, ma conserva l'intera distribuzione dei punteggi di ogni variante.
    Con solo_base=True calcola solo Modello e Frequenze storiche (più veloce: serve a scegliere i pesi)."""
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2:
        return None
    t = inizio if inizio is not None else df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    nan_m = np.full((11, 11), np.nan)
    rec = {"Partita": [], "xs": [], "ys": [], "ok": [], "ou_ok": [], "m1x2": [], "mou": [], "min_storia": [], "tiri_ok": [], "date": []}
    Ms = {v: [] for v in VARIANTI}
    ha_tiri = {"HST", "AST"}.issubset(df.columns)
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=FINESTRA_GIORNI))]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, True)
            cnt = pd.concat([train["HomeTeam"], train["AwayTeam"]]).value_counts()
            m_freq = matrice_frequenze(train)
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
                    Mn = matrice_punteggi(lh, la, 0.0)
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


def pesi_ottimi(res):
    """Peso del modello (resto: frequenze storiche) che minimizza l'errore: uno per i mercati gol, uno per i risultati esatti."""
    xs, ys = res["xs"], res["ys"]
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {"over1.5": (xs + ys > 1.5).astype(float), "over2.5": (xs + ys > 2.5).astype(float),
            "over3.5": (xs + ys > 3.5).astype(float), "btts": ((xs > 0) & (ys > 0)).astype(float)}
    Mb, Mf = res["Ms"]["Modello"], res["Ms"]["Frequenze storiche"]
    gol, esatto = {}, {}
    for w in W_GRID:
        Mw = w * Mb + (1 - w) * Mf
        gol[w] = float(np.mean([_vettore_perdita(Mw, k_, xs, ys, outc, esito).mean() for k_ in ("over1.5", "over2.5", "over3.5", "btts")]))
        esatto[w] = float(_vettore_perdita(Mw, "exact", xs, ys, outc, esito).mean())
    return {"gol": min(gol, key=gol.get), "esatto": min(esatto, key=esatto.get), "n": len(xs)}


def tabelle_test(res):
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
    righe, migliore = [], {}
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
    return tab_a, tab_b, tab_c, tab_d, pd.DataFrame(righe_e), best_ex


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def test_completo_cached(div, ridge, halflife, passo, giorno, versione):
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    return None if df is None else esegui_test_completo(df, ridge, halflife, passo)


@st.cache_data(ttl=6 * 3600, show_spinner="Calcolo i pesi tra modello e frequenze storiche (solo la prima volta di giornata)...")
def pesi_blend_auto(div, ridge, halflife, giorno):
    """Pesi scelti dal backtest: quanto fidarsi del modello (e quanto delle frequenze storiche) quando non ci sono quote."""
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    res = esegui_test_completo(df, ridge, halflife, 28, solo_base=True) if df is not None else None
    if res is None or len(res["xs"]) < 100:
        return {"gol": 0.5, "esatto": 0.75, "n": 0, "auto": False}
    w = pesi_ottimi(res)
    w["auto"] = True
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
fx, fx_err = carica_fixtures(div)
oggi_ts = pd.Timestamp.now().normalize()
calendario, fonte_cal = costruisci_calendario(div, df_storico, fx, chiave_api, modello["squadre"])

# Quote del prossimo turno (1X2 + Over/Under 2.5 senza margine) e distribuzione dei punteggi "migliore disponibile"
pesi = pesi_blend_auto(div, par["ridge"], par["halflife"], giorno)
M_freq_live = matrice_frequenze(df_storico)
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
    Mg = pesi["gol"] * Mb + (1 - pesi["gol"]) * M_freq_live
    Me = pesi["esatto"] * Mb + (1 - pesi["esatto"]) * M_freq_live
    return {"M_1x2": Mb, "M_gol": Mg, "M_es": Me, "ancorata": False,
            "fonte": f"modello + frequenze storiche (peso del modello: {pesi['gol']*100:.0f}% per Goal/Over, {pesi['esatto']*100:.0f}% per i risultati esatti) – nessuna quota disponibile"}


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
    st.caption(f"Fonte Goal / Over-Under / Risultati esatti: {d['fonte']}.")
    if not d["ancorata"]:
        st.warning("⚠ Senza le quote del bookmaker, Goal/No Goal e Over/Under hanno poco potere predittivo: nel backtest su circa 1200 partite "
                   "non hanno battuto in modo netto le semplici frequenze storiche. Considerali indicativi, vicini alla media del campionato.")
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
    if tc is not None and "tiri_ok" not in tc:
        tc = None  # risultato vecchio rimasto nella sessione: rilancia il test
    if tc is not None:
        tab_a, tab_b, tab_c, tab_d, tab_e, best_ex = tabelle_test(tc)
        st.markdown("#### 📏 Qualità per mercato (log loss, più basso è meglio)")
        st.dataframe(tab_a, use_container_width=True, hide_index=True)
        st.caption("Per Risultato esatto il valore è il log score del punteggio reale. 'Mercato (chiusura)' è il riferimento delle quote senza margine, "
                   "disponibile solo per 1X2 e Over/Under 2.5. Le colonne con ± mostrano la differenza media per partita e il suo errore: "
                   "se la differenza è più piccola di circa 2 volte l'errore, non è distinguibile dal caso.")
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

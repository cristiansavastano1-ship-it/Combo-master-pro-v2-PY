
import io
import math
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
# Dominio SENZA "www": con "www" football-data.co.uk blocca il traffico non britannico.
BASE_URL = "https://football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0"}
# Quote 1X2 "prendibili" (ordine di preferenza) e quote di chiusura (per il mercato di riferimento)
COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]


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
    """Risultati e quote delle ultime n stagioni. `giorno` serve solo a far scadere la cache ogni giorno."""
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


# ======================= MODELLO =======================
def adatta_modello(df, data_rif, ridge, halflife, usa_dc):
    """Poisson con forza d'attacco e di difesa per squadra, vantaggio casa, shrinkage (ridge)
    verso la media e pesi che decadono col tempo. Opzionale: correzione Dixon-Coles sui punteggi bassi."""
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
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    th = res.x
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


@st.cache_data(ttl=6 * 3600)
def modello_corrente(div, n_stagioni, ridge, halflife, usa_dc, giorno):
    df = carica_storico(div, n_stagioni, giorno)
    if df is None or len(df) < 100:
        return None
    return adatta_modello(df, pd.Timestamp.now().normalize(), ridge, halflife, usa_dc)


# ======================= BACKTEST =======================
def esegui_backtest_df(df, ridge, halflife, usa_dc, passo):
    """Walk-forward: ogni blocco di `passo` giorni viene previsto con un modello addestrato
    solo sulle partite precedenti. La prima stagione serve solo da storico iniziale."""
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
                    "p1": mk["p1"], "px": mk["px"], "p2": mk["p2"],
                    "f1": freq[0], "fx": freq[1], "f2": freq[2],
                    "q1": qp[0] if qp else np.nan, "qx": qp[1] if qp else np.nan, "q2": qp[2] if qp else np.nan,
                    "m1": pm[0], "mx": pm[1], "m2": pm[2],
                })
        t += pd.Timedelta(days=passo)
    return pd.DataFrame(righe)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_cached(div, n_stagioni, ridge, halflife, usa_dc, passo, giorno):
    df = carica_storico(div, n_stagioni, giorno)
    if df is None:
        return None
    return esegui_backtest_df(df, ridge, halflife, usa_dc, passo)


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
    P = b[["p1", "px", "p2"]].to_numpy()
    Q = b[["q1", "qx", "q2"]].to_numpy()
    M = b[["m1", "mx", "m2"]].to_numpy()
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
        pr = profitto[sel]
        c = clv[sel]
        righe.append({
            "Edge min %": s, "Scommesse": n,
            "Vinte %": round((y[sel] == k[sel]).mean() * 100, 1),
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


# ======================= INTERFACCIA =======================
st.markdown("### 🏆 Seleziona Campionato (Mobile & Desktop)")
campionato = st.selectbox("Scegli il torneo da analizzare", list(LEAGUES.keys()), label_visibility="collapsed")
div = LEAGUES[campionato]

st.sidebar.markdown("### ⚙ Parametri del modello")
n_stagioni = st.sidebar.slider("Stagioni di storico", 2, 4, 3)
ridge = st.sidebar.slider("Shrinkage (λ)", 0.1, 20.0, 2.0, 0.1)
halflife = st.sidebar.slider("Emivita dei pesi (giorni, 0 = nessun decadimento)", 0, 1000, 365, 30)
usa_dc = st.sidebar.checkbox("Correzione Dixon-Coles", True)
st.sidebar.caption("Valori di partenza non validati: verificali nel tab Backtest prima di fidarti delle probabilità.")
st.sidebar.markdown("---")
st.sidebar.info("💡 Dati: football-data.co.uk (risultati, quote, prossimo turno). Nessuna API key necessaria.")

giorno = date.today().isoformat()
df_storico = carica_storico(div, n_stagioni, giorno)
modello = modello_corrente(div, n_stagioni, ridge, halflife, usa_dc, giorno) if df_storico is not None else None

tab_value_finder, tab_match, tab_forze, tab_backtest, tab_kelly = st.tabs([
    "⚡ Value Finder", "📅 Analisi Partita", "📊 Forza Squadre", "🧪 Backtest", "🧮 Value Bet",
])

if modello is None:
    st.warning("Impossibile scaricare lo storico da football-data.co.uk (o dati insufficienti). Riprova tra poco.")
    st.stop()

# --- VALUE FINDER ---
with tab_value_finder:
    st.subheader("⚡ Scanner Value Finder")
    fx = carica_fixtures(div)
    if fx is None:
        st.warning("Impossibile scaricare il prossimo turno e le quote (fixtures.csv).")
    else:
        righe = []
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
            nota = "⚠ senza storico" if (casa not in modello["idx"] or trasf not in modello["idx"]) else ""
            righe.append({
                "Data": r["Date"].strftime("%d/%m") if "Date" in r and pd.notna(r["Date"]) else "",
                "Partita": f"{casa} vs {trasf}", "Selezione": nomi[k],
                "P1 %": round(P[0] * 100, 1), "PX %": round(P[1] * 100, 1), "P2 %": round(P[2] * 100, 1),
                "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1),
                "Quota_Book": round(q[k], 2), "Edge": round((P[k] * q[k] - 1) * 100, 1), "Nota": nota,
            })
        if not righe:
            st.info("Nessuna partita con quote nel prossimo turno (il file copre solo i prossimi giorni).")
        else:
            dfv = pd.DataFrame(righe).sort_values("Edge", ascending=False)
            st.caption("Edge = prob. modello × quota − 1, sulla media dei bookmaker. Un edge alto di solito è un errore del modello: "
                       "controlla nel tab Backtest da quale soglia l'edge ha avuto CLV positivo.")
            min_edge = st.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
            st.dataframe(dfv[dfv["Edge"] >= min_edge], use_container_width=True, hide_index=True)
            if (dfv["Nota"] != "").any():
                st.caption("⚠ Squadre senza storico (es. neopromosse) sono trattate come squadra media: probabilmente sopravvalutate.")

# --- ANALISI PARTITA ---
with tab_match:
    st.subheader("📅 Analisi Partita")
    nomi_sq = modello["squadre"]
    c1, c2 = st.columns(2)
    sq_c = c1.selectbox("Casa", nomi_sq, index=0, key="m_casa")
    sq_t = c2.selectbox("Trasferta", nomi_sq, index=min(1, len(nomi_sq) - 1), key="m_trasf")
    if sq_c == sq_t:
        st.info("Scegli due squadre diverse.")
    else:
        lh, la = lambda_partita(modello, sq_c, sq_t)
        M = matrice_punteggi(lh, la, modello["rho"])
        mk = mercati(M)
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 20px; margin: 10px 0; text-align:center;">
                <span style="color:#38bdf8; font-weight:800; font-size:1.3rem;">{sq_c}</span>
                <span style="color:#a855f7; margin:0 14px; font-weight:700;">VS</span>
                <span style="color:#f43f5e; font-weight:800; font-size:1.3rem;">{sq_t}</span><br>
                <span style="color:#94a3b8; font-size:0.9rem;">Gol attesi: {lh:.2f} - {la:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("Segno 1 (Casa)", f"{mk['p1']*100:.1f}%", f"Quota equa: {1/mk['p1']:.2f}")
        a2.metric("Segno X (Pareggio)", f"{mk['px']*100:.1f}%", f"Quota equa: {1/mk['px']:.2f}")
        a3.metric("Segno 2 (Trasferta)", f"{mk['p2']*100:.1f}%", f"Quota equa: {1/mk['p2']:.2f}")
        st.markdown("#### ⚽ Under / Over & Goal / No Goal")
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric("Over 1.5", f"{mk['over1.5']*100:.1f}%", f"Under: {(1-mk['over1.5'])*100:.1f}%")
        u2.metric("Over 2.5", f"{mk['over2.5']*100:.1f}%", f"Under: {(1-mk['over2.5'])*100:.1f}%")
        u3.metric("Over 3.5", f"{mk['over3.5']*100:.1f}%", f"Under: {(1-mk['over3.5'])*100:.1f}%")
        u4.metric("Goal (BTTS)", f"{mk['btts']*100:.1f}%", f"Quota: {1/mk['btts']:.2f}")
        u5.metric("No Goal", f"{(1-mk['btts'])*100:.1f}%", f"Quota: {1/(1-mk['btts']):.2f}")
        st.markdown("#### 🎯 Top 3 Risultati Esatti più Probabili")
        flat = sorted(((i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])), key=lambda t: t[2], reverse=True)[:3]
        cols = st.columns(3)
        for col, medaglia, (i, j, p) in zip(cols, ["🥇", "🥈", "🥉"], flat):
            col.info(f"{medaglia} **{i} - {j}** ({p*100:.1f}%)")

# --- FORZA SQUADRE ---
with tab_forze:
    st.subheader("📊 Forza Squadre (attacco e difesa)")
    st.caption("Valori in scala logaritmica rispetto alla squadra media (0). Attacco alto = segna di più; difesa alta = subisce meno.")
    df_f = pd.DataFrame({
        "Squadra": modello["squadre"],
        "Attacco": np.round(modello["att"], 3),
        "Difesa": np.round(-modello["dfn"], 3),
    })
    df_f["Totale"] = (df_f["Attacco"] + df_f["Difesa"]).round(3)
    df_f = df_f.sort_values("Totale", ascending=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Vantaggio casa (gol ×)", f"{math.exp(modello['hm']):.2f}")
    k2.metric("Gol base (squadra media, in trasferta)", f"{math.exp(modello['mu']):.2f}")
    k3.metric("Rho Dixon-Coles", f"{modello['rho']:+.3f}")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    fig = px.bar(df_f, x="Squadra", y="Totale", color="Attacco", template="plotly_dark", title="Forza complessiva (attacco + difesa)")
    st.plotly_chart(fig, use_container_width=True)

# --- BACKTEST ---
with tab_backtest:
    st.subheader("🧪 Backtest walk-forward")
    st.caption("Ogni blocco di partite è previsto da un modello addestrato solo sul passato. La prima stagione serve da storico iniziale. "
               "Si confronta con le frequenze storiche (modello banale) e con le quote di chiusura senza margine (il riferimento da battere).")
    passo = st.select_slider("Aggiornamento del modello ogni (giorni)", options=[7, 14, 21, 28], value=14)
    if st.button("🚀 Esegui Backtest"):
        with st.spinner("Calcolo in corso..."):
            st.session_state["bt"] = backtest_cached(div, n_stagioni, ridge, halflife, usa_dc, passo, giorno)
            st.session_state["bt_key"] = (div, n_stagioni, ridge, halflife, usa_dc, passo)
    bt = st.session_state.get("bt")
    if bt is not None and st.session_state.get("bt_key") != (div, n_stagioni, ridge, halflife, usa_dc, passo):
        st.info("I parametri sono cambiati dall'ultima esecuzione: rilancia il backtest.")
    if bt is not None and not bt.empty:
        mt = metriche_backtest(bt)
        st.markdown(f"#### 📏 Qualità delle probabilità 1X2 ({mt['n']} partite)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Log loss modello", f"{mt['modello'][0]:.4f}", f"Brier {mt['modello'][1]:.4f}", delta_color="off")
        r2.metric("Log loss frequenze storiche", f"{mt['frequenze'][0]:.4f}", f"Brier {mt['frequenze'][1]:.4f}", delta_color="off")
        if "mercato" in mt:
            r3.metric("Log loss mercato (chiusura)", f"{mt['mercato'][0]:.4f}", f"Brier {mt['mercato'][1]:.4f}", delta_color="off")
            gap = mt["modello_su_mercato"][0] - mt["mercato"][0]
            st.caption(f"Sulle {mt['n_mercato']} partite con quote di chiusura il modello fa {mt['modello_su_mercato'][0]:.4f} contro {mt['mercato'][0]:.4f} del mercato "
                       f"(scarto {gap:+.4f}; più basso è meglio). Se lo scarto è positivo, il mercato prevede meglio di te.")
        else:
            r3.metric("Log loss mercato", "N.D.")
            st.caption("Quote di chiusura non disponibili in questo storico.")

        st.markdown("#### 💰 Scommesse simulate (1 unità sull'esito con edge più alto)")
        tb = tabella_scommesse(bt)
        if tb.empty:
            st.info("Quote non disponibili per simulare le scommesse.")
        else:
            st.dataframe(tb, use_container_width=True, hide_index=True)
            st.caption("Il ROI su poche centinaia di scommesse è molto rumoroso: guarda l'errore (± ROI) e il CLV. "
                       "Un CLV medio positivo in modo stabile è un segnale più affidabile del ROI.")

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

    st.markdown("#### 🔎 Confronto parametri (griglia)")
    st.caption("Prova diverse combinazioni di shrinkage ed emivita. Scegli una zona stabile, non il valore minimo assoluto: "
               "ottimizzare sugli stessi dati del test porta a sovra-adattarsi.")
    if st.button("Cerca parametri (richiede qualche decina di secondi)"):
        righe = []
        barra = st.progress(0.0)
        combo = [(r_, h_) for r_ in (0.5, 1.0, 2.0, 4.0, 8.0) for h_ in (120, 240, 480, 0)]
        for i, (r_, h_) in enumerate(combo):
            b_ = backtest_cached(div, n_stagioni, r_, h_, usa_dc, 28, giorno)
            if b_ is not None and not b_.empty:
                m_ = metriche_backtest(b_)
                righe.append({"Shrinkage λ": r_, "Emivita (gg)": h_ if h_ else "nessuna", "Log loss": round(m_["modello"][0], 4),
                              "Brier": round(m_["modello"][1], 4)})
            barra.progress((i + 1) / len(combo))
        if righe:
            st.dataframe(pd.DataFrame(righe).sort_values("Log loss"), use_container_width=True, hide_index=True)

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

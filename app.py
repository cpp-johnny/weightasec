import html
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════
MAX_FILE_BYTES      = 5 * 1024 * 1024
MAX_ROWS            = 5_000
BB_WINDOW           = 20
BOOTSTRAP_N         = 1_000
ROLLING_WIN         = 30
SLOPE_STABLE_THRESH = 0.01 / 7   # ~0.01 kg/week — below this = "stable"
MAE_WINDOW          = 14          # last N days used for walk-forward MAE

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Weight a Sec...",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background-color:#0d0d0f; color:#e8e6e1; }
.main .block-container { padding:2rem 2.5rem 3rem 2.5rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#131316 !important; border-right:1px solid #222228; }
[data-testid="stSidebar"] .stMarkdown p { font-size:0.78rem; color:#666; letter-spacing:0.08em; text-transform:uppercase; }
h1 { font-family:'DM Serif Display',serif; font-size:2.6rem !important; letter-spacing:-0.02em; color:#f0ede8 !important; line-height:1.1 !important; }
h2 { font-family:'DM Serif Display',serif; font-size:1.5rem !important; color:#d4c9b8 !important; }
h3 { font-family:'DM Sans',sans-serif; font-size:0.72rem !important; font-weight:500 !important; letter-spacing:0.12em; text-transform:uppercase; color:#666 !important; margin-bottom:0.2rem !important; }
.metric-card { background:#16161a; border:1px solid #222228; border-radius:12px; padding:1.2rem 1.4rem; position:relative; overflow:hidden; transition:border-color 0.2s; }
.metric-card:hover { border-color:#3a3a44; }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--accent,#c8a97e); opacity:0.7; }
.metric-label { font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; color:#666; margin-bottom:0.4rem; }
.metric-value { font-family:'DM Mono',monospace; font-size:2rem; font-weight:400; color:#f0ede8; line-height:1; }
.metric-sub   { font-family:'DM Mono',monospace; font-size:0.78rem; margin-top:0.3rem; }
.metric-up    { color:#e07070; }
.metric-down  { color:#6bbf8e; }
.metric-flat  { color:#888; }
.section-title { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#a89880; border-bottom:1px solid #222228; padding-bottom:0.4rem; margin:1.8rem 0 1rem 0; }
.insight-box { background:#16161a; border:1px solid #222228; border-left:3px solid #c8a97e; border-radius:8px; padding:1rem 1.2rem; font-size:0.88rem; line-height:1.6; color:#c0bbb4; }
.warn-box    { background:#16161a; border:1px solid #222228; border-left:3px solid #e07070; border-radius:8px; padding:1rem 1.2rem; font-size:0.88rem; line-height:1.6; color:#c0bbb4; }
.integrity-box { background:#16161a; border:1px solid #222228; border-radius:12px; padding:1.2rem 1.4rem; }
.dq-green { color:#6bbf8e; font-weight:600; }
.dq-amber { color:#c8a97e; font-weight:600; }
.dq-red   { color:#e07070; font-weight:600; }
[data-testid="stHorizontalBlock"] { gap:1rem; }
.stTabs [data-baseweb="tab-list"] { background:#131316; border-bottom:1px solid #222228; gap:0; }
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-size:0.8rem; letter-spacing:0.06em; text-transform:uppercase; color:#666; padding:0.7rem 1.2rem; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#c8a97e !important; border-bottom-color:#c8a97e !important; background:transparent !important; }
.stSlider > div { color:#888; }
hr { border-color:#222228 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  EMBEDDED DEFAULT DATA
# ═══════════════════════════════════════════════
DEFAULT_CSV = """Date,Weight,Delta
1/1/2026,62.95,-0.7
2/1/2026,62.25,0.45
3/1/2026,62.7,-0.3
4/1/2026,62.4,-0.1
5/1/2026,62.3,0.35
6/1/2026,62.65,-0.25
7/1/2026,62.4,1.15
8/1/2026,63.55,-0.85
9/1/2026,62.7,-0.3
10/1/2026,62.4,0
11/1/2026,62.4,0.6
12/1/2026,63,-0.1
13/1/2026,62.9,0
14/1/2026,62.9,1.05
15/1/2026,63.95,0
16/1/2026,63.95,-0.1
17/1/2026,63.85,-0.35
18/1/2026,63.5,0.6
19/1/2026,64.1,-0.05
20/1/2026,64.05,0
21/1/2026,64.05,0.25
22/1/2026,64.3,0.35
23/1/2026,64.65,-0.35
24/1/2026,64.3,0.35
25/1/2026,64.65,-0.55
26/1/2026,64.1,0.7
27/1/2026,64.8,0.1
28/1/2026,64.9,-0.35
29/1/2026,64.55,0.55
30/1/2026,65.1,-0.1
31/1/2026,65,-0.35
1/2/2026,64.65,-1.55
2/2/2026,63.1,0.4
3/2/2026,63.5,0.5
4/2/2026,64,0.65
5/2/2026,64.65,-0.55
6/2/2026,64.1,1.2
7/2/2026,65.3,-0.8
8/2/2026,64.5,0
9/2/2026,64.5,-0.1
10/2/2026,64.4,0.1
11/2/2026,64.5,0.3
12/2/2026,64.8,-0.55
13/2/2026,64.25,0.7
14/2/2026,64.95,0.3
15/2/2026,65.25,-0.8
16/2/2026,64.45,1.05
17/2/2026,65.5,-0.35
18/2/2026,65.15,-0.2
19/2/2026,64.95,-0.4
20/2/2026,64.55,1.55
21/2/2026,66.1,-0.75
22/2/2026,65.35,-1.15
23/2/2026,64.2,-0.7
24/2/2026,63.5,0.85
25/2/2026,64.35,0.3
26/2/2026,64.65,-0.2
27/2/2026,64.45,0.1
28/2/2026,64.55,0.45
1/3/2026,65,-0.25
2/3/2026,64.75,0.25
3/3/2026,65,0.2
4/3/2026,65.2,0.4
5/3/2026,65.6,-0.2
6/3/2026,65.4,-0.4
7/3/2026,65,0.4
8/3/2026,65.4,0.15
9/3/2026,65.55,-0.55
10/3/2026,65,1
11/3/2026,66,-0.25
12/3/2026,65.75,0.6
13/3/2026,66.35,0.5
14/3/2026,66.85,-0.25
15/3/2026,66.6,-1
16/3/2026,65.6,0.4
17/3/2026,66,0.35
18/3/2026,66.35,0.7
19/3/2026,67.05,0.2
20/3/2026,67.25,-0.75
21/3/2026,66.5,0.7
22/3/2026,67.2,-0.45
23/3/2026,66.75,0.55
24/3/2026,67.3,-0.05
25/3/2026,67.25,0
26/3/2026,67.25,0.25
27/3/2026,67.5,0.55
28/3/2026,68.05,-0.05
29/3/2026,68,-0.95
30/3/2026,67.05,0.45
31/3/2026,67.5,1.75
1/4/2026,69.25,-0.45
2/4/2026,68.8,0.35
3/4/2026,69.15,-0.4
4/4/2026,68.75,0
5/4/2026,68.75,-0.75
6/4/2026,68,0.75
7/4/2026,68.75,-0.4
8/4/2026,68.35,-0.1
9/4/2026,68.25,0.1
10/4/2026,68.35,0.2
11/4/2026,68.55,0.45
12/4/2026,69,-0.75
13/4/2026,68.25,0.75
14/4/2026,69,0.4
15/4/2026,69.4,-0.8
16/4/2026,68.6,0.45
17/4/2026,69.05,-0.05
18/4/2026,69,-0.15
19/4/2026,68.85,0.15
20/4/2026,69,-0.4
21/4/2026,68.6,0.65
22/4/2026,69.25,0.45
23/4/2026,69.7,-0.6
24/4/2026,69.1,0
25/4/2026,69.1,0.8
26/4/2026,69.9,-0.7
27/4/2026,69.2,-0.4
28/4/2026,68.8,1.4
29/4/2026,70.2,-0.25
30/4/2026,69.95,-0.65
1/5/2026,69.3,0"""

# ═══════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════
CARD_BG = "#16161a"
GRID    = "#1e1e24"
TEXT    = "#e8e6e1"
MUTED   = "#555560"
GOLD    = "#c8a97e"
BLUE    = "#7eafc8"
GREEN   = "#7ec89e"
PURPLE  = "#9e7ec8"
RED     = "#e07070"
TEAL    = "#7ec8c8"

# ═══════════════════════════════════════════════
#  PURE HELPERS
# ═══════════════════════════════════════════════

def safe(value) -> str:
    return html.escape(str(value))


def sanitize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: ("'" + x)
            if isinstance(x, str) and x[:1] in ("=", "+", "-", "@") else x)
    return df


def _chip(val: float) -> str:
    if abs(val) < 0.001:
        return '<span class="metric-flat">→ 0.00 kg</span>'
    css = "metric-up" if val > 0 else "metric-down"
    sym = "↑" if val > 0 else "↓"
    return f'<span class="{css}">{sym} {abs(val):.2f} kg</span>'


def base_layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="DM Serif Display", size=16,
                   color="#a89880"), x=0.01),
        height=height,
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color=TEXT, size=11),
        xaxis=dict(gridcolor=GRID, showline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, showline=False, tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=10)),
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
    )


# ═══════════════════════════════════════════════
#  LAYER 2 — WALK-FORWARD MAE
# ═══════════════════════════════════════════════

def _wf_mae_naive(series: np.ndarray, window: int) -> float:
    """
    Walk-forward 1-step MAE for naive model: predict y[t] = y[t-1].
    Averaged over last `window` steps. This is the irreducible noise floor —
    if a model can't beat this, it adds no signal beyond persistence.
    """
    errors = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    return float(np.mean(errors[-window:]))


def _wf_mae_ols(series: np.ndarray, window: int) -> float:
    """
    Walk-forward 1-step MAE for global OLS.
    At each step t, fits OLS on [0..t-1], predicts t.
    Evaluated over last `window` steps only.
    """
    n     = len(series)
    x_all = np.arange(n, dtype=float)
    errors = []
    start  = max(3, n - window - 1)
    for t in range(start, n):
        xi  = x_all[:t]; yi = series[:t]
        xm  = xi.mean(); ym = yi.mean(); dxi = xi - xm
        den = (dxi * dxi).sum()
        pred = yi[-1] if den == 0 else (ym - ((dxi*(yi-ym)).sum()/den)*xm) + \
               ((dxi*(yi-ym)).sum()/den)*x_all[t]
        errors.append(abs(series[t] - pred))
    return float(np.mean(errors[-window:])) if errors else np.nan


def _wf_mae_holt(series: np.ndarray, window: int,
                 alpha: float, beta: float) -> float:
    """
    Walk-forward 1-step MAE for Holt smoothing using globally-fitted α/β.
    At each step the prediction uses state accumulated only up to t-1.
    """
    n   = len(series)
    lvl = series[0]
    trd = series[1] - series[0] if n > 1 else 0.0
    errors = []
    for i in range(1, n):
        pred = lvl + trd
        if i >= n - window:
            errors.append(abs(series[i] - pred))
        l_t = alpha * series[i] + (1 - alpha) * pred
        b_t = beta  * (l_t - lvl) + (1 - beta) * trd
        lvl, trd = l_t, b_t
    return float(np.mean(errors)) if errors else np.nan


# ═══════════════════════════════════════════════
#  LAYER 1 — DATA QUALITY
# ═══════════════════════════════════════════════

def compute_data_quality(df: pd.DataFrame) -> dict:
    """
    Signals that determine whether the statistical stack can be trusted:
      gaps     — inter-measurement gaps > 2 days (corrupt rolling calcs)
      n        — total observations (short series inflates bootstrap variance)
      vol_cv   — coefficient of variation of 7-day rolling σ
                 (high CV = noise regime has shifted = smoothing assumptions broken)
    """
    gaps    = int((df["Date"].diff().dt.days > 2).sum())
    max_gap = int(df["Date"].diff().dt.days.max())
    n       = len(df)

    vol_s  = df["Volatility"].dropna()
    vol_cv = float(vol_s.std() / vol_s.mean()) if len(vol_s) > 0 and vol_s.mean() > 0 else 0.0

    return dict(
        gaps=gaps,
        gap_status  = "green" if gaps == 0 else ("amber" if gaps <= 3 else "red"),
        max_gap=max_gap,
        n=n,
        n_status    = "green" if n >= 30 else ("amber" if n >= 14 else "red"),
        vol_cv=vol_cv,
        vol_status  = "green" if vol_cv < 0.3 else ("amber" if vol_cv < 0.6 else "red"),
    )


# ═══════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════

def _detect_columns(df_raw: pd.DataFrame):
    cols_lower = {c.strip().lower(): c for c in df_raw.columns}

    date_keys  = ["date","day","timestamp","time","dt","recorded"]
    date_col   = next((cols_lower[k] for k in date_keys if k in cols_lower), None)
    if date_col is None:
        for c in df_raw.columns:
            try: pd.to_datetime(df_raw[c].dropna().iloc[:3], dayfirst=True); date_col=c; break
            except Exception: pass
    if date_col is None:
        raise ValueError("No Date column found. Expected 'date', 'day', or 'timestamp'.")

    weight_keys = ["weight","kg","mass","bw","bodyweight","wt","lbs","lb"]
    weight_col  = next((cols_lower[k] for k in weight_keys if k in cols_lower), None)
    if weight_col is None:
        for c in df_raw.columns:
            if c == date_col: continue
            if pd.api.types.is_numeric_dtype(df_raw[c]): weight_col=c; break
    if weight_col is None:
        raise ValueError("No Weight column found. Expected a numeric column named 'weight', 'kg', etc.")

    delta_keys = ["delta","change","diff","chg","delta_kg"]
    delta_col  = next((cols_lower[k] for k in delta_keys if k in cols_lower), None)
    if delta_col is None:
        for orig in df_raw.columns:
            if orig.strip() in ("∆","Δ"): delta_col=orig; break

    out = pd.DataFrame()
    out["Date"]   = df_raw[date_col]
    out["Weight"] = pd.to_numeric(df_raw[weight_col], errors="coerce")

    had_delta = False
    if delta_col:
        parsed = pd.to_numeric(df_raw[delta_col], errors="coerce")
        if parsed.notna().mean() >= 0.5:
            out["Delta"] = parsed; had_delta = True
    if not had_delta:
        out["Delta"] = np.nan

    return out, had_delta


def _fit_holt(series: np.ndarray):
    def _run(params, y):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1): return 1e12
        lvl=[y[0]]; trd=[y[1]-y[0] if len(y)>1 else 0.0]; sse=0.0
        for i in range(1,len(y)):
            pred=lvl[-1]+trd[-1]; sse+=(y[i]-pred)**2
            l_t=a*y[i]+(1-a)*pred; b_t=b*(l_t-lvl[-1])+(1-b)*trd[-1]
            lvl.append(l_t); trd.append(b_t)
        return sse

    res   = minimize(_run, x0=[0.3,0.1], args=(series,), method="Nelder-Mead",
                     options={"xatol":1e-5,"fatol":1e-5,"maxiter":1000})
    alpha = float(np.clip(res.x[0] if res.success else 0.3, 1e-4, 1-1e-4))
    beta  = float(np.clip(res.x[1] if res.success else 0.1, 1e-4, 1-1e-4))

    lvl=[series[0]]; trd=[series[1]-series[0] if len(series)>1 else 0.0]
    fitted=[lvl[0]+trd[0]]
    for i in range(1,len(series)):
        l_t=alpha*series[i]+(1-alpha)*(lvl[-1]+trd[-1])
        b_t=beta*(l_t-lvl[-1])+(1-beta)*trd[-1]
        lvl.append(l_t); trd.append(b_t); fitted.append(l_t+b_t)

    return alpha, beta, lvl, trd, fitted


def _rolling_ols(x: np.ndarray, y: np.ndarray,
                 win: int, fb_slope: float, fb_intercept: float):
    n=len(x); rs=np.empty(n); rt=np.empty(n)
    for i in range(n):
        lo=max(0,i-win+1); xi=x[lo:i+1]; yi=y[lo:i+1]
        if len(xi)<3: rs[i]=fb_slope; rt[i]=fb_intercept+fb_slope*x[i]; continue
        xm=xi.mean(); ym=yi.mean(); dxi=xi-xm; den=(dxi*dxi).sum()
        s=((dxi*(yi-ym)).sum()/den) if den!=0 else fb_slope
        rs[i]=s; rt[i]=(ym-s*xm)+s*x[i]
    return rs, rt


@st.cache_data(show_spinner=False)
def load_data(csv_text: str):
    df_raw = pd.read_csv(StringIO(csv_text), engine="python", on_bad_lines="skip")
    if len(df_raw) > MAX_ROWS:
        raise ValueError(f"Too many rows ({len(df_raw):,}). Maximum is {MAX_ROWS:,}.")

    df, had_delta = _detect_columns(df_raw)
    df["Date"]   = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Date","Weight"])
    if df.empty: raise ValueError("No valid (Date, Weight) rows found after parsing.")
    df = df.sort_values("Date").reset_index(drop=True)

    # Delta — double fillna(0) prevents first-row NaN ever propagating
    if not had_delta:
        df["Delta"] = df["Weight"].diff().fillna(0)
    else:
        df["Delta"] = df["Delta"].fillna(df["Weight"].diff()).fillna(0)

    df["DayNum"] = (df["Date"] - df["Date"].iloc[0]).dt.days

    for w in [7,14,30]:
        df[f"MA{w}"] = df["Weight"].rolling(w, min_periods=max(1,w//2)).mean()
    df["EMA7"] = df["Weight"].ewm(span=7, adjust=False).mean()

    slope_g, intercept_g, r_g, _, _ = stats.linregress(df["DayNum"], df["Weight"])
    df["Trend_global"] = intercept_g + slope_g * df["DayNum"]
    df["slope_global"] = slope_g
    df["r2_global"]    = r_g ** 2

    rs, rt = _rolling_ols(df["DayNum"].values, df["Weight"].values,
                          ROLLING_WIN, slope_g, intercept_g)
    df["slope_rolling"] = rs
    df["Trend_rolling"] = rt

    alpha, beta, levels, trends, holt_fitted = _fit_holt(df["Weight"].values)
    df["Holt"]        = holt_fitted
    df["_holt_level"] = levels
    df["_holt_trend"] = trends
    df["_holt_alpha"] = alpha
    df["_holt_beta"]  = beta

    df["Volatility"] = df["Weight"].rolling(7, min_periods=7).std()
    df["BB_mid"]     = df["Weight"].rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    bb_std           = df["Weight"].rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    df["BB_upper"]   = df["BB_mid"] + 2 * bb_std
    df["BB_lower"]   = df["BB_mid"] - 2 * bb_std

    df["DOW"]       = df["Date"].dt.day_name()
    df["Month"]     = df["Date"].dt.strftime("%b %Y")
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["Weekday"]   = df["Date"].dt.weekday
    df["CumChange"] = df["Weight"] - df["Weight"].iloc[0]
    df["RoC7"]      = df["Weight"].diff(7)

    return df, slope_g, r_g ** 2


@st.cache_data(show_spinner=False)
def bootstrap_forecast(weights_tuple: tuple, fitted_tuple: tuple,
                        n_ahead: int, last_fitted_val: float,
                        proj_slope: float, n_boot: int = BOOTSTRAP_N):
    """
    Parametric bootstrap PI aligned to the active model.
    Residual window capped at min(30, n//2) — robust for short series.
    No damping — intervals widen naturally with horizon.
    """
    weights = np.array(weights_tuple); fitted = np.array(fitted_tuple)
    resids  = weights - fitted
    n_resid = min(30, max(5, len(weights) // 2))
    resids  = resids[-n_resid:]

    rng       = np.random.default_rng(42)
    noise     = rng.choice(resids, size=(n_boot, n_ahead), replace=True)
    trend_ext = proj_slope * np.arange(1, n_ahead + 1)
    paths     = last_fitted_val + trend_ext + np.cumsum(noise, axis=1)

    return np.percentile(paths, 2.5, axis=0), np.percentile(paths, 97.5, axis=0)


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💪 Weight a Sec...")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                help="Columns: Date, Weight, (optional) Delta")
    st.markdown("---")
    st.markdown("**Chart layers**")
    show_raw   = st.checkbox("Raw data points",        value=True)
    show_ma7   = st.checkbox("MA 7",                   value=True)
    show_ma14  = st.checkbox("MA 14",                  value=False)
    show_ma30  = st.checkbox("MA 30",                  value=True)
    show_ema   = st.checkbox("EMA 7",                  value=False)
    show_holt  = st.checkbox("Holt smoothing",         value=False)
    show_trend = st.checkbox("Global trend (OLS)",     value=True)
    show_roll  = st.checkbox("Rolling trend (30-day)", value=False)
    show_band  = st.checkbox("Bollinger bands (20,2)", value=True)

    st.markdown("---")
    st.markdown("**Goal weight (kg)**")
    goal = st.number_input("", value=75.0, step=0.5, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Projection**")
    proj_days  = st.slider("Days ahead", 7, 60, 30)
    proj_model = st.radio("Model", ["Holt (trend-aware)", "Global OLS"], index=0)

# ═══════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════
if uploaded is not None:
    if not uploaded.name.lower().endswith(".csv"):
        st.error("Only .csv files are accepted."); st.stop()
    if uploaded.size > MAX_FILE_BYTES:
        st.error(f"File too large ({uploaded.size/1024/1024:.1f} MB). Maximum is 5 MB."); st.stop()
    raw_bytes = uploaded.read()
    try: raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try: raw_text = raw_bytes.decode("latin-1")
        except Exception: st.error("Could not decode file. Save as UTF-8."); st.stop()
else:
    raw_text = DEFAULT_CSV

try:
    df, slope, r2 = load_data(raw_text)
except ValueError as exc:
    st.error(f"⚠️ {safe(str(exc))}"); st.info("Falling back to built-in demo dataset.")
    df, slope, r2 = load_data(DEFAULT_CSV)

# ═══════════════════════════════════════════════
#  DERIVED STATS
# ═══════════════════════════════════════════════
current      = float(df["Weight"].iloc[-1])
start_w      = float(df["Weight"].iloc[0])
total_chg    = current - start_w
max_w        = float(df["Weight"].max())
min_w        = float(df["Weight"].min())
slope_recent = float(df["slope_rolling"].iloc[-1])
holt_alpha   = float(df["_holt_alpha"].iloc[-1])
holt_beta    = float(df["_holt_beta"].iloc[-1])

with st.sidebar:
    st.markdown("---")
    st.caption(f"Holt α = {holt_alpha:.3f} · β = {holt_beta:.3f}  \n_(fitted via SSE minimisation)_")

# ── Projection ──
last_day   = int(df["DayNum"].iloc[-1])
last_date  = df["Date"].iloc[-1]
proj_x     = np.arange(1, proj_days + 1)
proj_dates = [last_date + pd.Timedelta(days=int(d)) for d in proj_x]

if proj_model == "Holt (trend-aware)":
    l0 = float(df["_holt_level"].iloc[-1]); b0 = float(df["_holt_trend"].iloc[-1])
    proj_w      = np.array([l0 + b0 * t for t in proj_x])
    eff_slope   = b0
    proj_label  = f"{b0*7:+.3f} kg/wk (Holt, α={holt_alpha:.2f} β={holt_beta:.2f})"
    boot_fitted = df["Holt"].values;      boot_last = float(df["Holt"].iloc[-1]);      boot_slope = b0
else:
    intercept_val = float(df["Trend_global"].iloc[0])
    proj_w        = intercept_val + slope * (last_day + proj_x)
    eff_slope     = slope
    proj_label    = f"{slope*7:+.3f} kg/wk (OLS)"
    boot_fitted   = df["Trend_global"].values; boot_last = float(df["Trend_global"].iloc[-1]); boot_slope = slope

boot_lo, boot_hi = bootstrap_forecast(
    tuple(df["Weight"].values), tuple(boot_fitted), proj_days, boot_last, boot_slope)

# ── Goal reachability ──
_toward = (goal > current and eff_slope > 0) or (goal < current and eff_slope < 0)
if abs(eff_slope) < SLOPE_STABLE_THRESH:
    goal_status, days_to_goal = "stable", np.inf
elif not _toward:
    goal_status, days_to_goal = "away", np.inf
else:
    days_to_goal = abs((goal - current) / eff_slope)
    goal_status  = "reachable"

# ── Walk-forward MAE ──
w_series  = df["Weight"].values
mae_naive = _wf_mae_naive(w_series, MAE_WINDOW)
mae_ols   = _wf_mae_ols(w_series, MAE_WINDOW)
mae_holt  = _wf_mae_holt(w_series, MAE_WINDOW, holt_alpha, holt_beta)

# ── Data quality ──
dq = compute_data_quality(df)

# ═══════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════
st.markdown("<h1>Weight Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:#666;font-size:0.85rem;margin-top:-0.5rem;'>"
    f"{safe(df['Date'].iloc[0].strftime('%d %b %Y'))} — "
    f"{safe(df['Date'].iloc[-1].strftime('%d %b %Y'))} "
    f"· {len(df)} days recorded</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════
if goal_status == "reachable":
    goal_sub = f'<span class="metric-flat">{days_to_goal:.0f} days away</span>'
elif goal_status == "away":
    goal_sub = '<span class="metric-up">Trend diverging</span>'
else:
    goal_sub = '<span class="metric-flat">Weight stable</span>'

chg_css = "metric-up" if total_chg > 0 else "metric-down"
kpi_cols = st.columns(5)
kpi_rows = [
    ("Current", f"{current:.2f} kg",       _chip(float(df["Delta"].iloc[-1])),                              "#c8a97e"),
    ("Change",  f"{total_chg:+.2f} kg",     f'<span class="{chg_css}">from {start_w:.2f} kg</span>',        "#7eafc8"),
    ("Range",   f"{min_w:.1f}–{max_w:.1f}", f'<span class="metric-flat">span {max_w-min_w:.2f} kg</span>',  "#9e7ec8"),
    ("Trend",   f"{eff_slope*7:+.2f}/wk",   f'<span class="metric-flat">R² {r2:.2f}</span>',                "#7ec89e"),
    ("Goal",    f"{goal:.1f} kg",            goal_sub,                                                        "#c87e7e"),
]
for col, (label, val, sub, accent) in zip(kpi_cols, kpi_rows):
    with col:
        st.markdown(
            f'<div class="metric-card" style="--accent:{accent}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════
tabs = st.tabs(["📈 Overview", "📊 Analysis", "📅 Calendar",
                "🔮 Projection", "🔬 Model Integrity", "📋 Data"])

# ─────────────────────────────
#  TAB 1 – OVERVIEW
# ─────────────────────────────
with tabs[0]:
    fig = go.Figure()
    if show_band:
        bb_ok = df["BB_upper"].notna() & df["BB_lower"].notna()
        bd,bu,bl = df.loc[bb_ok,"Date"],df.loc[bb_ok,"BB_upper"],df.loc[bb_ok,"BB_lower"]
        fig.add_trace(go.Scatter(x=pd.concat([bd,bd[::-1]]),y=pd.concat([bu,bl[::-1]]),
            fill="toself",fillcolor="rgba(200,169,126,0.06)",
            line=dict(color="rgba(0,0,0,0)"),name="Bollinger Band",hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bd,y=bu,line=dict(color="rgba(200,169,126,0.25)",width=1,dash="dot"),
            name="BB Upper",hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=bd,y=bl,line=dict(color="rgba(200,169,126,0.25)",width=1,dash="dot"),
            name="BB Lower",hoverinfo="skip",showlegend=False))
    if show_raw:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Weight"],mode="lines+markers",
            line=dict(color="rgba(200,169,126,0.35)",width=1),
            marker=dict(color=GOLD,size=3,opacity=0.6),name="Daily weight",connectgaps=True))
    if show_ma7:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["MA7"],line=dict(color=GOLD,width=2.5),name="MA 7"))
    if show_ma14:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["MA14"],line=dict(color=BLUE,width=2),name="MA 14"))
    if show_ma30:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["MA30"],line=dict(color=GREEN,width=2),name="MA 30"))
    if show_ema:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["EMA7"],line=dict(color=PURPLE,width=2,dash="dot"),name="EMA 7"))
    if show_holt:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Holt"],line=dict(color=TEAL,width=2,dash="dot"),name="Holt smooth"))
    if show_trend:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Trend_global"],line=dict(color=RED,width=1.5,dash="dash"),name="Global OLS"))
    if show_roll:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Trend_rolling"],line=dict(color="#e0a070",width=1.5,dash="dot"),name="Rolling trend (30d)"))
    fig.add_hline(y=goal,line_dash="dot",line_color="rgba(126,200,158,0.4)",
                  annotation_text=f"Goal {goal} kg",annotation_font=dict(color=GREEN,size=10))
    fig.update_layout(**base_layout("Weight Over Time",height=460))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2 = st.columns([2,1])
    with c1:
        cols_d=[GREEN if v<=0 else RED for v in df["Delta"]]
        f2=go.Figure(go.Bar(x=df["Date"],y=df["Delta"],marker_color=cols_d,name="Daily Δ"))
        f2.update_layout(**base_layout("Daily Change (kg)",height=220))
        st.plotly_chart(f2,use_container_width=True)
    with c2:
        dfw=df.set_index("Date").resample("W")["Weight"].agg(["first","last"])
        dfw["chg"]=dfw["last"]-dfw["first"]
        cols_w=[GREEN if v<=0 else RED for v in dfw["chg"]]
        f3=go.Figure(go.Bar(x=dfw.index,y=dfw["chg"],marker_color=cols_w,name="Weekly Δ"))
        f3.update_layout(**base_layout("Weekly Change",height=220))
        st.plotly_chart(f3,use_container_width=True)

# ─────────────────────────────
#  TAB 2 – ANALYSIS
# ─────────────────────────────
with tabs[1]:
    c1,c2=st.columns(2)
    with c1:
        f4=go.Figure()
        f4.add_trace(go.Histogram(x=df["Weight"],nbinsx=25,marker_color=GOLD,opacity=0.75,name="Distribution"))
        f4.add_vline(x=df["Weight"].mean(),line_dash="dash",line_color=BLUE,
            annotation_text=f"Mean {df['Weight'].mean():.2f}",annotation_font=dict(color=BLUE,size=10))
        f4.add_vline(x=df["Weight"].median(),line_dash="dot",line_color=GREEN,
            annotation_text=f"Median {df['Weight'].median():.2f}",annotation_font=dict(color=GREEN,size=10))
        f4.update_layout(**base_layout("Weight Distribution",height=310))
        st.plotly_chart(f4,use_container_width=True)
    with c2:
        f5=go.Figure()
        f5.add_trace(go.Scatter(x=df["Date"],y=df["Volatility"],fill="tozeroy",
            fillcolor="rgba(158,126,200,0.15)",line=dict(color=PURPLE,width=1.5),name="7-day σ"))
        f5.update_layout(**base_layout("7-Day Volatility (σ)",height=310))
        st.plotly_chart(f5,use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg=df.groupby("DOW")["Weight"].mean().reindex(dow_order)
        f6=go.Figure(go.Bar(x=dow_avg.index,y=dow_avg.values,
            marker_color=[GOLD if v==dow_avg.max() else BLUE for v in dow_avg.values],
            text=dow_avg.round(2).values,textposition="outside",textfont=dict(color=MUTED,size=9)))
        f6.update_layout(**base_layout("Avg Weight by Day of Week",height=310))
        st.plotly_chart(f6,use_container_width=True)
    with c4:
        f7=go.Figure()
        for month in df["Month"].unique():
            sub=df[df["Month"]==month]
            f7.add_trace(go.Box(y=sub["Weight"],name=month,marker_color=GOLD,line_color=GOLD,
                fillcolor="rgba(200,169,126,0.15)",boxmean=True,showlegend=False))
        f7.update_layout(**base_layout("Monthly Distribution",height=310))
        st.plotly_chart(f7,use_container_width=True)

    f_rs=go.Figure()
    f_rs.add_trace(go.Scatter(x=df["Date"],y=df["slope_rolling"]*7,mode="lines",
        line=dict(color=TEAL,width=1.8),name="Rolling slope (kg/wk)",
        fill="tozeroy",fillcolor="rgba(126,200,200,0.08)"))
    f_rs.add_hline(y=0,line_color=MUTED,line_width=1)
    f_rs.update_layout(**base_layout("30-Day Rolling Slope (kg/wk)",height=240))
    st.plotly_chart(f_rs,use_container_width=True)

    f8=go.Figure()
    f8.add_trace(go.Scatter(x=df["Date"],y=df["RoC7"],mode="lines",
        line=dict(color=BLUE,width=2),name="7-day RoC",
        fill="tozeroy",fillcolor="rgba(126,175,200,0.1)"))
    f8.add_hline(y=0,line_color=MUTED,line_width=1)
    f8.update_layout(**base_layout("7-Day Rate of Change (kg)",height=220))
    st.plotly_chart(f8,use_container_width=True)

    up_days=int((df["Delta"]>0).sum()); down_days=int((df["Delta"]<0).sum()); flat_days=int((df["Delta"]==0).sum())
    best_day=df.loc[df["Delta"].idxmin()]; worst_day=df.loc[df["Delta"].idxmax()]
    max_dow=safe(str(dow_avg.idxmax()))

    st.markdown(f"""
    <div class="insight-box">
    <strong style="color:#c8a97e;">📊 Key Insights</strong><br><br>
    Out of <strong>{len(df)}</strong> days,
    weight <strong style="color:#6bbf8e;">fell on {down_days}</strong>,
    <strong style="color:#e07070;">rose on {up_days}</strong>, flat on {flat_days}.<br><br>
    Best single-day drop: <strong style="color:#6bbf8e;">{best_day['Delta']:+.2f} kg</strong>
    on {safe(best_day['Date'].strftime('%d %b'))} ·
    Largest gain: <strong style="color:#e07070;">{worst_day['Delta']:+.2f} kg</strong>
    on {safe(worst_day['Date'].strftime('%d %b'))}.<br><br>
    Heaviest day on average: <strong style="color:#c8a97e;">{max_dow}</strong> ·
    Global trend: <strong>{"gaining" if slope>0 else "losing"} {abs(slope*7):.2f} kg/wk</strong>
    (R² = {r2:.2f}) · Recent 30-day slope: <strong>{slope_recent*7:+.2f} kg/wk</strong>.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────
#  TAB 3 – CALENDAR HEATMAP
# ─────────────────────────────
with tabs[2]:
    pivot=df.pivot_table(index="Weekday",columns="Week",values="Weight",aggfunc="mean")
    day_labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    f9=go.Figure(go.Heatmap(z=pivot.values,x=[f"W{w}" for w in pivot.columns],y=day_labels,
        colorscale=[[0,"#1a2e22"],[0.5,"#c8a97e"],[1,"#e07070"]],
        text=np.round(pivot.values,1),texttemplate="%{text}",
        textfont=dict(size=9,color="rgba(255,255,255,0.5)"),
        hovertemplate="Week %{x}, %{y}: <b>%{z:.2f} kg</b><extra></extra>",
        showscale=True,colorbar=dict(tickfont=dict(color=MUTED),outlinewidth=0)))
    f9.update_layout(**base_layout("Weight Heatmap by Week × Day",height=300))
    st.plotly_chart(f9,use_container_width=True)

    pivot_d=df.pivot_table(index="Weekday",columns="Week",values="Delta",aggfunc="mean")
    f10=go.Figure(go.Heatmap(z=pivot_d.values,x=[f"W{w}" for w in pivot_d.columns],y=day_labels,
        colorscale=[[0,"#1a3b2e"],[0.5,"#222228"],[1,"#3b1a1a"]],zmid=0,
        text=np.round(pivot_d.values,2),texttemplate="%{text}",
        textfont=dict(size=8,color="rgba(255,255,255,0.4)"),
        hovertemplate="Week %{x}, %{y}: <b>%{z:+.2f} kg</b><extra></extra>",
        showscale=True,colorbar=dict(tickfont=dict(color=MUTED),outlinewidth=0)))
    f10.update_layout(**base_layout("Daily Change Heatmap",height=300))
    st.plotly_chart(f10,use_container_width=True)

# ─────────────────────────────
#  TAB 4 – PROJECTION
# ─────────────────────────────
with tabs[3]:
    f11=go.Figure()
    f11.add_trace(go.Scatter(x=df["Date"],y=df["Holt"],
        line=dict(color=TEAL,width=1.5,dash="dot"),name="Holt (historical)"))
    f11.add_trace(go.Scatter(x=df["Date"],y=df["MA7"],line=dict(color=GOLD,width=2),name="MA 7"))
    f11.add_trace(go.Scatter(
        x=list(proj_dates)+list(proj_dates[::-1]),y=list(boot_hi)+list(boot_lo[::-1]),
        fill="toself",fillcolor="rgba(200,169,126,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"95% PI (bootstrap, n={BOOTSTRAP_N:,})",hoverinfo="skip"))
    f11.add_trace(go.Scatter(x=proj_dates,y=boot_lo,
        line=dict(color="rgba(200,169,126,0.2)",width=1),showlegend=False,hoverinfo="skip"))
    f11.add_trace(go.Scatter(x=proj_dates,y=boot_hi,
        line=dict(color="rgba(200,169,126,0.2)",width=1),showlegend=False,hoverinfo="skip"))
    f11.add_trace(go.Scatter(x=proj_dates,y=proj_w,
        line=dict(color=GOLD,width=2.5,dash="dash"),
        name=f"Projection ({proj_model.split()[0]})"))
    f11.add_hline(y=goal,line_dash="dot",line_color=GREEN,
                  annotation_text=f"Goal {goal} kg",annotation_font=dict(color=GREEN,size=10))
    f11.add_trace(go.Scatter(x=[df["Date"].iloc[-1]],y=[current],
        mode="markers",marker=dict(color=GOLD,size=10),name="Today"))
    f11.update_layout(**base_layout(f"Projection — {safe(proj_model)} · {proj_days} days",height=460))
    st.plotly_chart(f11,use_container_width=True)

    step_idx=list(range(0,proj_days,7))
    proj_table=pd.DataFrame({
        "Date":           [proj_dates[i].strftime("%d %b %Y") for i in step_idx],
        "Projected (kg)": np.round(proj_w[step_idx],2),
        "Lower 95% PI":   np.round(boot_lo[step_idx],2),
        "Upper 95% PI":   np.round(boot_hi[step_idx],2),
    })
    st.markdown("<div class='section-title'>Weekly Projections</div>",unsafe_allow_html=True)
    st.dataframe(proj_table,use_container_width=True,hide_index=True)
    st.caption(
        f"Prediction intervals via parametric bootstrap resampling of "
        f"{'Holt' if 'Holt' in proj_model else 'OLS'} residuals ({BOOTSTRAP_N:,} iterations). "
        f"Intervals widen over time due to accumulated uncertainty — interpret with caution "
        f"over longer horizons or after lifestyle changes."
    )

    if goal_status=="reachable":
        goal_date=last_date+pd.Timedelta(days=int(days_to_goal))
        st.markdown(f"""
        <div class="insight-box">
        At <strong>{safe(proj_label)}</strong>, you should reach your goal of
        <strong>{goal:.1f} kg</strong> in approximately <strong>{days_to_goal:.0f} days</strong>
        — around <strong>{safe(goal_date.strftime('%d %b %Y'))}</strong>.
        </div>""",unsafe_allow_html=True)
    elif goal_status=="away":
        direction_word="gaining" if eff_slope>0 else "losing"
        st.markdown(f"""
        <div class="warn-box">
        ⚠️ Your current trend is <strong>{direction_word} {abs(eff_slope*7):.3f} kg/week</strong>,
        moving <strong>away</strong> from your goal of {goal:.1f} kg.
        Reverse trajectory to get back on track.
        </div>""",unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box" style="border-left-color:#888;">
        Weight appears stable — trend is below 0.01 kg/week. Set a new goal in the sidebar
        to model a target, or wait for a clearer trend to emerge.
        </div>""",unsafe_allow_html=True)

# ─────────────────────────────
#  TAB 5 – MODEL INTEGRITY
# ─────────────────────────────
with tabs[4]:

    # ── LAYER 1: Data Validity ──────────────────
    st.markdown("<div class='section-title'>Layer 1 — Data Validity</div>",
                unsafe_allow_html=True)
    st.write(
        "These signals determine whether the statistical stack can be trusted. "
        "Gaps corrupt rolling means and Bollinger bands. Low sample counts inflate "
        "bootstrap variance. Unstable volatility suggests the noise regime has shifted, "
        "which breaks the smoothing assumptions."
    )

    dq_cols = st.columns(3)

    gap_colour = dq["gap_status"]
    gap_label  = {"green":"✓ Clean","amber":"⚠ Minor gaps","red":"✗ Significant gaps"}[gap_colour]
    gap_note   = (f"{dq['gaps']} gap(s) &gt;2d · largest gap: {dq['max_gap']}d"
                  if dq["gaps"] > 0 else "No gaps longer than 2 days")

    n_colour = dq["n_status"]
    n_label  = {"green":"✓ Sufficient","amber":"⚠ Limited","red":"✗ Very short"}[n_colour]

    vol_colour = dq["vol_status"]
    vol_label  = {"green":"✓ Stable","amber":"⚠ Some shift","red":"✗ High shift"}[vol_colour]
    vol_note   = f"Volatility CV = {dq['vol_cv']:.2f}"

    with dq_cols[0]:
        st.markdown(f"""
        <div class="integrity-box">
          <div class="metric-label">Measurement Gaps</div>
          <div class="metric-value dq-{gap_colour}">{gap_label}</div>
          <div style="font-size:0.75rem;color:#666;margin-top:0.5rem">{gap_note}</div>
        </div>""",unsafe_allow_html=True)

    with dq_cols[1]:
        st.markdown(f"""
        <div class="integrity-box">
          <div class="metric-label">Sample Size</div>
          <div class="metric-value dq-{n_colour}">{n_label}</div>
          <div style="font-size:0.75rem;color:#666;margin-top:0.5rem">{dq['n']} observations</div>
        </div>""",unsafe_allow_html=True)

    with dq_cols[2]:
        st.markdown(f"""
        <div class="integrity-box">
          <div class="metric-label">Noise Regime Stability</div>
          <div class="metric-value dq-{vol_colour}">{vol_label}</div>
          <div style="font-size:0.75rem;color:#666;margin-top:0.5rem">{vol_note}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LAYER 2: Predictive Honesty ─────────────
    st.markdown("<div class='section-title'>Layer 2 — Predictive Honesty</div>",
                unsafe_allow_html=True)
    st.write(
        f"Walk-forward 1-step MAE over the last **{MAE_WINDOW} days**: "
        "each prediction is made using only data available at that moment, "
        "eliminating in-sample optimism. "
        "The naive baseline (tomorrow = today) is the irreducible noise floor — "
        "a model that can't beat it adds no signal beyond persistence."
    )

    # Determine winner and framing
    best_mae  = min(mae_ols, mae_holt)
    naive_gap = mae_naive - best_mae

    if naive_gap < 0.02:
        comparison_note = (
            "Both OLS and Holt are close to the naive baseline — "
            "daily weight fluctuation dominates trend signal at this horizon. "
            "Projections are directional guides, not precise targets."
        )
        comparison_css = "warn-box"
    else:
        winner = "Holt" if mae_holt < mae_ols else "OLS"
        comparison_note = (
            f"{winner} performs better on recent walk-forward error. "
            f"Improvement over naive: {naive_gap:.3f} kg/day."
        )
        comparison_css = "insight-box"

    mae_cols = st.columns(3)
    mae_items = [
        ("Naive Baseline", mae_naive, "predict y[t] = y[t−1]",            "#888"),
        ("Global OLS",     mae_ols,   "linear trend model",                BLUE),
        ("Holt Smoothing", mae_holt,  f"α={holt_alpha:.2f} β={holt_beta:.2f}", TEAL),
    ]
    for col, (name, val, note, accent) in zip(mae_cols, mae_items):
        is_best  = (val == best_mae)
        best_tag = " <span style='font-size:0.65rem;color:#c8a97e;letter-spacing:0.1em'>▲ BEST</span>" if is_best else ""
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{'#c8a97e' if is_best else '#333'}">
              <div class="metric-label">{name}{best_tag}</div>
              <div class="metric-value">{val:.3f} kg</div>
              <div class="metric-sub"><span class="metric-flat">{note}</span></div>
            </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # MAE bar chart
    mae_fig = go.Figure(go.Bar(
        x=["Naive","OLS","Holt"],
        y=[mae_naive,mae_ols,mae_holt],
        marker_color=[MUTED,BLUE,TEAL],
        text=[f"{v:.3f}" for v in [mae_naive,mae_ols,mae_holt]],
        textposition="outside",textfont=dict(color=MUTED,size=10),
    ))
    mae_fig.add_hline(y=mae_naive,line_dash="dot",line_color=MUTED,
                      annotation_text="Naive floor",annotation_font=dict(color=MUTED,size=9))
    mae_fig.update_layout(**base_layout(
        f"Walk-Forward MAE — last {MAE_WINDOW} days (lower = better)",height=280))
    mae_fig.update_yaxes(title_text="MAE (kg)")
    st.plotly_chart(mae_fig,use_container_width=True)

    st.markdown(f'<div class="{comparison_css}">{comparison_note}</div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # ── LAYER 3: Residuals ──────────────────────
    st.markdown("<div class='section-title'>Residuals Over Time</div>",
                unsafe_allow_html=True)
    st.write(
        "Residuals (actual − fitted) for OLS and Holt. "
        "Random scatter around zero is healthy. "
        "Persistent drift in one direction signals model misspecification."
    )

    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(x=df["Date"],y=df["Weight"]-df["Trend_global"],
        mode="lines",line=dict(color=BLUE,width=1.2),name="OLS residuals",
        fill="tozeroy",fillcolor="rgba(126,175,200,0.06)"))
    res_fig.add_trace(go.Scatter(x=df["Date"],y=df["Weight"]-df["Holt"],
        mode="lines",line=dict(color=TEAL,width=1.2),name="Holt residuals",
        fill="tozeroy",fillcolor="rgba(126,200,200,0.06)"))
    res_fig.add_hline(y=0,line_color=MUTED,line_width=1)
    res_fig.update_layout(**base_layout("",height=240))
    st.plotly_chart(res_fig,use_container_width=True)

    st.caption(
        "Walk-forward MAE: train on [0..t-1], predict t, measure |actual − predicted|, "
        f"average over last {MAE_WINDOW} days. "
        "In-sample MAE (fitted vs actual on same data) is intentionally omitted — "
        "it overstates model quality."
    )

# ─────────────────────────────
#  TAB 6 – DATA TABLE
# ─────────────────────────────
with tabs[5]:
    display_cols = [
        "Date","Weight","Delta","MA7","MA14","MA30",
        "EMA7","Holt","Trend_global","Trend_rolling","Volatility","CumChange",
    ]
    disp = df[display_cols].copy()
    disp["Date"] = disp["Date"].dt.strftime("%d %b %Y")
    for c in display_cols[1:]: disp[c] = disp[c].round(2)
    disp = disp.rename(columns={"Trend_global":"Trend (OLS)","Trend_rolling":"Trend (30d)"})

    st.dataframe(
        disp.style.background_gradient(subset=["Weight"],cmap="RdYlGn_r").format(precision=2),
        use_container_width=True,height=500)

    export = sanitize_for_export(disp)
    st.download_button(
        "⬇ Download enriched CSV",
        data=export.to_csv(index=False),
        file_name="weight_enriched.csv",
        mime="text/csv",
    )
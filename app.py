import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import os
import random
import numpy as np
import psycopg2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

# =====================================================
# CONFIG
# =====================================================
ENTRY_TF = "4h"
SR_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

TP1_R = 0.8
TP2_R = 2.0

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# TELEGRAM
# =====================================================
def send_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except:
        pass

# =====================================================
# DATABASE
# =====================================================
def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signal_history (
        time TEXT,
        symbol TEXT,
        phase TEXT,
        candle TEXT,
        entry REAL,
        sl REAL,
        tp1 REAL,
        tp2 REAL,
        priority INT,
        rating TEXT,
        status TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_results (
        time TEXT,
        symbol TEXT,
        r REAL
    );
    """)

    conn.commit()
    conn.close()

def save_signal(sig):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO signal_history VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, tuple(sig.values()))
    conn.commit()
    conn.close()

def load_signal_history():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM signal_history ORDER BY time DESC", conn)
    conn.close()
    return df

def has_open_signal(symbol):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM signal_history
        WHERE symbol=%s AND status='OPEN'
        LIMIT 1
    """, (symbol,))
    res = cur.fetchone()
    conn.close()
    return res is not None

def save_trade_result(symbol, r):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trade_results VALUES (%s,%s,%s)
    """, (now_wib(), symbol, r))
    conn.commit()
    conn.close()

def load_trade_results():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM trade_results", conn)
    conn.close()
    return df

# =====================================================
# CCXT
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# AUTO UPDATE R
# =====================================================
def update_trade_outcomes(okx):
    df = load_signal_history()
    if df.empty:
        return

    for i, row in df.iterrows():
        if row["status"] != "OPEN":
            continue

        try:
            price = okx.fetch_ticker(row["symbol"])["last"]
        except:
            continue

        r = None
        status = None

        if price <= row["sl"]:
            r, status = -1, "SL HIT"
        elif price >= row["tp2"]:
            r, status = TP2_R, "TP2 HIT"
        elif price >= row["tp1"]:
            r, status = TP1_R, "TP1 HIT"

        if r is not None:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("""
                UPDATE signal_history
                SET status=%s
                WHERE symbol=%s AND status='OPEN'
            """, (status, row["symbol"]))
            conn.commit()
            conn.close()

            save_trade_result(row["symbol"], r)

            send_telegram(
                f"📉 *TRADE CLOSED*\n{row['symbol']}\nStatus: *{status}*\nR: *{r}R*"
            )

# =====================================================
# MARKET DATA
# =====================================================
@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
    r = requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType": "SPOT"},
        timeout=15
    )
    r.raise_for_status()
    syms = [
        d["instId"] for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]
    return random.sample(syms, min(MAX_SCAN_SYMBOLS, len(syms)))

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c=df.high,df.low,df.close
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=period,adjust=False).mean()
    hl2=(h+l)/2
    upper=hl2+mult*atr
    lower=hl2-mult*atr

    stl=pd.Series(index=df.index,dtype=float)
    trend=pd.Series(index=df.index,dtype=int)
    trend.iloc[0]=1
    stl.iloc[0]=lower.iloc[0]

    for i in range(1,len(df)):
        if trend.iloc[i-1]==1:
            stl.iloc[i]=max(lower.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=1 if c.iloc[i]>stl.iloc[i] else -1
        else:
            stl.iloc[i]=min(upper.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=-1 if c.iloc[i]<stl.iloc[i] else 1
    return stl,trend

def volume_oscillator(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def ad_phase(adl, lookback=10):
    slope=adl.iloc[-1]-adl.iloc[-lookback]
    avg=adl.diff().rolling(lookback).mean().iloc[-1]
    strength=slope/(abs(avg)+1e-9)
    if slope>0:
        return "AKUMULASI_KUAT" if strength>2 else "AKUMULASI_LEMAH"
    if slope<0:
        return "DISTRIBUSI"
    return "NETRAL"

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v3.7", layout="wide")
st.title("🚀 OPSI A PRO v3.7 — POSTGRES EDITION")

okx = get_okx()
init_db()
update_trade_outcomes(okx)

tab1, tab2, tab3 = st.tabs(["📡 Live Signal", "📜 Riwayat", "🎲 Monte Carlo"])

with tab1:
    if st.button("🔍 Scan Live Signal"):
        syms = get_liquid_symbols(MIN_USDT_VOLUME)
        prog = st.progress(0)
        status = st.empty()

        found = []
        for i,s in enumerate(syms,1):
            status.text(f"{i}/{len(syms)} {s}")
            if not has_open_signal(s):
                try:
                    df = pd.DataFrame(okx.fetch_ohlcv(s, ENTRY_TF, limit=200),
                        columns=["t","open","high","low","close","volume"])
                    stl,trend = supertrend(df, ATR_PERIOD, MULTIPLIER)
                    vo = volume_oscillator(df.volume, VO_FAST, VO_SLOW)
                    adl = accumulation_distribution(df)
                    phase = ad_phase(adl)

                    if trend.iloc[-1]==1 and vo.iloc[-1]>VO_MIN and phase=="AKUMULASI_KUAT":
                        entry=df.close.iloc[-1]
                        sl=df.low.iloc[-10:].min()
                        risk=entry-sl
                        sig={
                            "time":now_wib(),"symbol":s,"phase":phase,
                            "candle":"Normal",
                            "entry":round(entry,8),
                            "sl":round(sl,8),
                            "tp1":round(entry+risk*TP1_R,8),
                            "tp2":round(entry+risk*TP2_R,8),
                            "priority":4,
                            "rating":"⭐⭐⭐⭐",
                            "status":"OPEN"
                        }
                        save_signal(sig)
                        found.append(sig)
                        send_telegram(f"🚀 *NEW SIGNAL*\n{s}\nEntry: {sig['entry']}")
                except:
                    pass
            prog.progress(i/len(syms))
            time.sleep(RATE_LIMIT_DELAY)

        prog.empty(); status.empty()
        st.success(f"{len(found)} signal ditemukan")

with tab2:
    st.dataframe(load_signal_history(), use_container_width=True)

with tab3:
    df_r = load_trade_results()
    if df_r.empty:
        st.info("Belum ada trade closed.")
    else:
        r = df_r["r"].values
        curves=[]
        for _ in range(500):
            bal=10000; eq=[bal]
            for _ in range(200):
                bal+=bal*0.01*np.random.choice(r)
                eq.append(bal)
            curves.append(eq)
        curves=np.array(curves)

        fig=go.Figure()
        for i in range(min(30,len(curves))):
            fig.add_trace(go.Scatter(y=curves[i],mode="lines",opacity=0.3))
        fig.update_layout(template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

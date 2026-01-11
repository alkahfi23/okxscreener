import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import os
import random
import numpy as np
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

BASE_DIR = os.getcwd()
SIGNAL_LOG_FILE = os.path.join(BASE_DIR, "signal_history.csv")
TRADE_RESULT_FILE = os.path.join(BASE_DIR, "trade_results.csv")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# PRIORITY
# =====================================================
PAIR_PRIORITY = {
    "BCH-USDT": 5,
    "WLFI-USDT": 4,
    "ZEC-USDT": 3,
    "PEPE-USDT": 2
}

# =====================================================
# CCXT
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# FILE HANDLERS
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","Symbol","Phase","Candle",
            "Entry","SL","TP1","TP2",
            "Priority","Rating","Status"
        ])
        df.to_csv(SIGNAL_LOG_FILE, index=False)
    return pd.read_csv(SIGNAL_LOG_FILE)

def save_signal(signal):
    df = load_signal_history()
    df = pd.concat([df, pd.DataFrame([signal])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

def has_open_signal(symbol):
    df = load_signal_history()
    return ((df["Symbol"] == symbol) & (df["Status"] == "OPEN")).any()

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        df = pd.DataFrame(columns=["Time","Symbol","R"])
        df.to_csv(TRADE_RESULT_FILE, index=False)
    return pd.read_csv(TRADE_RESULT_FILE)

# =====================================================
# AUTO UPDATE R FROM TP / SL
# =====================================================
def update_trade_outcomes(okx):
    history = load_signal_history()
    if history.empty:
        return

    results = load_trade_results()
    updated = False

    for i, row in history.iterrows():
        if row["Status"] != "OPEN":
            continue

        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue

        r = None
        status = None

        if price <= row["SL"]:
            r, status = -1, "SL HIT"
        elif price >= row["TP2"]:
            r, status = TP2_R, "TP2 HIT"
        elif price >= row["TP1"]:
            r, status = TP1_R, "TP1 HIT"

        if r is not None:
            history.at[i, "Status"] = status
            results = pd.concat([
                results,
                pd.DataFrame([{
                    "Time": now_wib(),
                    "Symbol": row["Symbol"],
                    "R": r
                }])
            ], ignore_index=True)
            updated = True

    if updated:
        history.to_csv(SIGNAL_LOG_FILE, index=False)
        results.to_csv(TRADE_RESULT_FILE, index=False)

# =====================================================
# SYMBOL FETCH
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
    h,l,c = df.high, df.low, df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index,dtype=float)
    trend = pd.Series(index=df.index,dtype=int)
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
    slope = adl.iloc[-1] - adl.iloc[-lookback]
    avg = adl.diff().rolling(lookback).mean().iloc[-1]
    strength = slope / (abs(avg) + 1e-9)
    if slope > 0:
        return "AKUMULASI_KUAT" if strength > 2 else "AKUMULASI_LEMAH"
    if slope < 0:
        return "DISTRIBUSI"
    return "NETRAL"

# =====================================================
# PRICE ACTION & SUPPORT
# =====================================================
def detect_candle(df):
    o,h,l,c=df.open,df.high,df.low,df.close
    body=abs(c-o); rng=h-l
    if rng.iloc[-1]<df.high.iloc[-20:].mean()*0.3: return "Normal"
    if c.iloc[-1]>o.iloc[-1] and c.iloc[-2]<o.iloc[-2] and c.iloc[-1]>o.iloc[-2]:
        return "Bullish Engulfing"
    if c.iloc[-1]>o.iloc[-1] and (o.iloc[-1]-l.iloc[-1])>2*body.iloc[-1]:
        return "Hammer"
    if body.iloc[-1]/rng.iloc[-1]>0.65 and c.iloc[-1]>o.iloc[-1]:
        return "Strong Bullish"
    return "Normal"

def find_support(df,lb):
    raw=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i]==min(df.low.iloc[i-lb:i+lb+1]):
            raw.append(df.low.iloc[i])
    raw=sorted(set(raw))
    filt=[]
    for s in raw:
        if not filt or abs(s-filt[-1])/s>0.01:
            filt.append(s)
    return filt

# =====================================================
# SIGNAL CHECK
# =====================================================
def check_signal(okx,symbol):
    if has_open_signal(symbol):
        return None

    df4h=pd.DataFrame(okx.fetch_ohlcv(symbol,ENTRY_TF,limit=LIMIT_4H),
        columns=["t","open","high","low","close","volume"])
    df1d=pd.DataFrame(okx.fetch_ohlcv(symbol,SR_TF,limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"])

    stl,trend=supertrend(df4h,ATR_PERIOD,MULTIPLIER)
    vo=volume_oscillator(df4h.volume,VO_FAST,VO_SLOW)
    adl=accumulation_distribution(df4h)
    phase=ad_phase(adl)

    if trend.iloc[-1]!=1 or vo.iloc[-1]<VO_MIN or phase not in ["AKUMULASI_KUAT","AKUMULASI_LEMAH"]:
        return None

    ema200=df1d.close.ewm(span=200).mean()
    if ema200.isna().iloc[-1] or df1d.close.iloc[-1]<ema200.iloc[-1]:
        return None

    entry=df4h.close.iloc[-1]
    supports=[s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
    if not supports:
        return None

    sl=max(supports)*(1-ZONE_BUFFER)
    if entry-sl<entry*0.002:
        return None

    risk=entry-sl
    priority=PAIR_PRIORITY.get(symbol,3)
    if phase=="AKUMULASI_KUAT":
        priority=min(priority+1,5)

    return {
        "Time":now_wib(),"Symbol":symbol,"Phase":phase,
        "Candle":detect_candle(df4h),
        "Entry":round(entry,8),"SL":round(sl,8),
        "TP1":round(entry+risk*TP1_R,8),
        "TP2":round(entry+risk*TP2_R,8),
        "Priority":priority,"Rating":"⭐"*priority,
        "Status":"OPEN"
    }

# =====================================================
# CHART (ON-DEMAND)
# =====================================================
def get_chart_data(okx, symbol):
    df4h = pd.DataFrame(
        okx.fetch_ohlcv(symbol, ENTRY_TF, limit=100),
        columns=["t","open","high","low","close","volume"]
    )
    stl,_ = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    adl = accumulation_distribution(df4h)
    return df4h, stl, adl

def render_chart(df,stl,adl,signal):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3])
    fig.add_candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime"),name="Supertrend"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan"),name="A/D"),row=2,col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=signal[k],line_color=c,row=1)

    fig.update_layout(height=520,template="plotly_dark",xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v3.6.1",layout="wide")
st.title("🚀 OPSI A PRO v3.6.1 — PROGRESS + CHART FIX")

tab1,tab2,tab3=st.tabs(["📡 Live Signal","📜 Riwayat","🎲 Monte Carlo"])
okx=get_okx()

update_trade_outcomes(okx)

with tab1:
    if st.button("🔍 Scan Live Signal"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        total = len(symbols)

        progress = st.progress(0)
        status = st.empty()
        signals = []

        for i,s in enumerate(symbols,1):
            status.text(f"Scanning {s} ({i}/{total})")
            try:
                sig = check_signal(okx,s)
                if sig:
                    save_signal(sig)
                    signals.append(sig)
            except:
                pass
            progress.progress(i/total)
            time.sleep(RATE_LIMIT_DELAY)

        progress.empty()
        status.empty()

        if signals:
            st.success(f"🔥 {len(signals)} SIGNAL DITEMUKAN")
            st.dataframe(pd.DataFrame(signals),use_container_width=True)

            for sig in signals:
                with st.expander(f"📈 {sig['Symbol']} — Chart"):
                    with st.spinner("Loading chart..."):
                        dfc,stlc,adlc = get_chart_data(okx, sig["Symbol"])
                        st.plotly_chart(render_chart(dfc,stlc,adlc,sig),use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    st.dataframe(load_signal_history(),use_container_width=True)

with tab3:
    df_r=load_trade_results()
    if df_r.empty:
        st.info("Belum ada trade closed.")
    else:
        r=df_r["R"].values
        risk=st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades=st.slider("Trades / Simulation",50,500,300)

        if st.button("Run Monte Carlo"):
            curves=[]
            for _ in range(1000):
                bal=10000; eq=[bal]
                for _ in range(trades):
                    bal+=bal*risk*np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)
            curves=np.array(curves)

            st.write(f"Median Final Balance: ${np.median(curves[:,-1]):,.0f}")
            st.write(f"Ruin Probability: {(curves[:,-1]<5000).mean()*100:.2f}%")

            fig=go.Figure()
            for i in range(min(50,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i],mode="lines",opacity=0.3,showlegend=False))
            fig.update_layout(template="plotly_dark",height=400)
            st.plotly_chart(fig,use_container_width=True)

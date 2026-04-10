
"""
ARIMA Modelling — Interactive Learning Lab
The Mountain Path Academy | Prof. V. Ravichandran
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════
GOLD = "#FFD700"; BLUE = "#003366"; MID = "#004d80"; CARD = "#112240"
TXT = "#e6f1ff"; MUTED = "#8892b0"; GRN = "#28a745"; RED = "#dc3545"
LB = "#ADD8E6"; WARN = "#FFC107"; ORANGE = "#fd7e14"; TEAL = "#17a2b8"
PURPLE = "#6f42c1"; BG_GRAD = "linear-gradient(135deg,#1a2332,#243447,#2a3f5f)"

st.set_page_config(page_title="ARIMA Modelling — The Mountain Path Academy", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp {{ background:{BG_GRAD}; font-family:'Source Sans 3',sans-serif; }}
section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#0a1628 0%,#112240 50%,#0a1628 100%) !important; border-right:2px solid {GOLD} !important; }}
section[data-testid="stSidebar"] * {{ color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; }}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] span {{ color:{GOLD} !important; -webkit-text-fill-color:{GOLD} !important; font-weight:600 !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:0.5rem; background:rgba(17,34,64,0.6); border-radius:12px; padding:6px; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED} !important; background:transparent !important; border-radius:8px !important; }}
.stTabs [aria-selected="true"] {{ color:{GOLD} !important; background:rgba(255,215,0,0.12) !important; border-bottom:2px solid {GOLD} !important; }}
[data-testid="stMetric"] {{ background:{CARD} !important; border:1px solid rgba(255,215,0,0.2) !important; border-radius:12px !important; padding:16px !important; }}
[data-testid="stMetricValue"] {{ color:{GOLD} !important; -webkit-text-fill-color:{GOLD} !important; font-family:'JetBrains Mono',monospace !important; font-size:1.6rem !important; }}
[data-testid="stMetricLabel"] {{ color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; }}
.stSlider label,.stNumberInput label,.stSelectbox label {{ color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; }}
.stSlider [data-baseweb="slider"] div {{ color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; }}
footer {{visibility:hidden;}} #MainMenu {{visibility:hidden;}}
</style>""")

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def mp_header(title, sub=""):
    s = f'<div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:1rem;margin-top:4px;user-select:none;">{sub}</div>' if sub else ""
    st.html(f'<div style="user-select:none;margin-bottom:18px;"><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:2rem;font-weight:700;">{title}</div>{s}<div style="height:3px;background:linear-gradient(90deg,{GOLD},transparent);border-radius:2px;margin-top:8px;width:40%;"></div></div>')

def mp_sub(title):
    st.html(f'<div style="color:{LB};-webkit-text-fill-color:{LB};font-family:Playfair Display,serif;font-size:1.35rem;font-weight:600;margin:20px 0 10px 0;user-select:none;">{title}</div>')

def mp_card(content, border=GOLD):
    st.html(f'<div style="background:{CARD};border:1px solid {border};border-left:4px solid {border};border-radius:10px;padding:18px 22px;margin:10px 0;user-select:none;"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.95rem;line-height:1.7;">{content}</div></div>')

def mp_insight(title, content):
    st.html(f'<div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-left:4px solid {GOLD};border-radius:10px;padding:16px 20px;margin:12px 0;user-select:none;"><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:1rem;font-weight:700;margin-bottom:6px;">💡 {title}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.92rem;line-height:1.7;">{content}</div></div>')

def mp_warn(title, content):
    st.html(f'<div style="background:rgba(220,53,69,0.08);border:1px solid rgba(220,53,69,0.3);border-left:4px solid {RED};border-radius:10px;padding:16px 20px;margin:12px 0;user-select:none;"><div style="color:{RED};-webkit-text-fill-color:{RED};font-size:0.95rem;font-weight:700;margin-bottom:6px;">⚠️ {title}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.92rem;line-height:1.7;">{content}</div></div>')

def mp_formula(label, formula, expl=""):
    ex = f'<div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.85rem;margin-top:6px;">{expl}</div>' if expl else ""
    st.html(f'<div style="background:rgba(0,51,102,0.4);border:1px solid rgba(173,216,230,0.25);border-radius:10px;padding:14px 20px;margin:8px 0;user-select:none;"><div style="color:{LB};-webkit-text-fill-color:{LB};font-size:0.85rem;font-weight:600;margin-bottom:4px;">{label}</div><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:JetBrains Mono,monospace;font-size:1.02rem;">{formula}</div>{ex}</div>')

def mp_step(num, title, desc):
    st.html(f'<div style="display:flex;gap:14px;align-items:flex-start;background:{CARD};border-radius:10px;padding:14px 18px;margin:6px 0;border-left:4px solid {ORANGE};user-select:none;"><div style="background:{ORANGE};color:white;-webkit-text-fill-color:white;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;flex-shrink:0;">{num}</div><div><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-weight:600;font-size:0.95rem;">{title}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;margin-top:3px;line-height:1.6;">{desc}</div></div></div>')

def plotly_theme(fig, title="", h=420):
    fig.update_layout(title=dict(text=title, font=dict(family="Playfair Display", size=18, color=GOLD), x=0.5),
        paper_bgcolor="rgba(17,34,64,0.85)", plot_bgcolor="rgba(17,34,64,0.4)",
        font=dict(family="Source Sans 3", color=TXT, size=13), height=h,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)),
        xaxis=dict(gridcolor="rgba(136,146,176,0.15)", zerolinecolor="rgba(136,146,176,0.15)"),
        yaxis=dict(gridcolor="rgba(136,146,176,0.15)", zerolinecolor="rgba(136,146,176,0.15)"))
    return fig

# ══════════════════════════════════════════════════════════
# DATA FROM EXCEL
# ══════════════════════════════════════════════════════════
# Illustration 1: Retail Sales
RETAIL_MONTHS = [f"Jan'23",f"Feb'23",f"Mar'23",f"Apr'23",f"May'23",f"Jun'23",
                 f"Jul'23",f"Aug'23",f"Sep'23",f"Oct'23",f"Nov'23",f"Dec'23",
                 f"Jan'24",f"Feb'24",f"Mar'24",f"Apr'24",f"May'24",f"Jun'24",
                 f"Jul'24",f"Aug'24",f"Sep'24",f"Oct'24",f"Nov'24",f"Dec'24"]
RETAIL_SALES = [120,125,130,128,135,140,138,145,150,148,155,162,
                158,165,170,168,175,180,178,185,190,188,195,202]
RETAIL_FITTED = [None,120,126.95,132.34,129.37,137.52,142.45,139.39,147.53,152.46,
                 149.39,157.53,164.76,159.15,167.48,172.45,169.39,177.53,182.46,179.39,187.53,192.46,189.39,197.53]
RETAIL_FORECAST_M = [f"Jan'25",f"Feb'25",f"Mar'25",f"Apr'25",f"May'25",f"Jun'25"]
RETAIL_FORECAST_V = [204.76,206.92,208.88,210.76,212.62,214.47]

# Illustration 2: Stock Prices
STOCK_DAYS = list(range(1,21))
STOCK_PRICE = [100,101.5,99.8,102.3,101.1,103.7,102.5,105.2,104,106.8,
               105.3,108.1,106.4,109.5,107.9,110.6,109.2,112,110.5,113.3]

# Illustration 3: Ice Cream SARIMA
ICE_YEARS = [2021]*4+[2022]*4+[2023]*4+[2024]*4
ICE_QTRS = ["Q1","Q2","Q3","Q4"]*4
ICE_SALES = [45,120,180,55,52,135,200,62,58,148,218,70,65,160,235,78]
ICE_SEASONAL = {"Q1":0.468,"Q2":1.197,"Q3":1.771,"Q4":0.564}
ICE_FORECAST = [("2025 Q1",72),("2025 Q2",172),("2025 Q3",252),("2025 Q4",86)]

# Case Study 1: Bakery
BAKERY_WEEKS = list(range(1,31))
BAKERY_SALES = [200,214,208,222,218,231,225,240,236,248,243,257,250,265,260,273,268,282,276,290,285,298,292,305,300,314,308,322,316,330]
BAKERY_FORECAST = [(31,335.3),(32,340.0),(33,344.3),(34,348.6)]

# Case Study 2: Airline
AIRLINE_MONTHS_LBL = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
AIRLINE_Y1 = [112,118,132,129,121,135,148,152,136,119,104,118]
AIRLINE_Y2 = [120,126,141,138,131,146,160,165,149,130,112,126]
AIRLINE_Y3 = [128,135,150,148,140,157,172,178,161,141,121,135]
AIRLINE_FCAST = [137,144,160,159,151,169,185,193,175,153,131,144]
AIRLINE_SEASONAL = [("Jan",0.876,"❄️ Low"),("Feb",0.922,"❄️ Low"),("Mar",1.029,"↗️ Spring"),
                    ("Apr",1.009,"↗️ Spring"),("May",0.953,"➖ Shoulder"),("Jun",1.065,"☀️ Surge"),
                    ("Jul",1.167,"☀️ PEAK"),("Aug",1.204,"☀️ PEAK"),("Sep",1.085,"↘️ Fall"),
                    ("Oct",0.949,"↘️ Fall"),("Nov",0.820,"❄️ Low"),("Dec",0.922,"❄️ Holiday")]

# Stationarity Check
STATION_ORIG = list(range(1,25))
STATION_YT = [50,55,52,60,63,58,67,72,68,75,80,76,83,88,85,92,96,91,100,105,99,108,112,107]
STATION_DIFF = [v-STATION_YT[i] for i,v in enumerate(STATION_YT[1:])]
STATION_ACF_ORIG = {"Lag 1":0.958,"Lag 2":0.953,"Lag 3":0.998,"Lag 4":0.946}
STATION_ACF_DIFF = {"Lag 1":-0.492,"Lag 2":-0.495,"Lag 3":0.972,"Lag 4":-0.474}


# ══════════════════════════════════════════════════════════
# ARIMA SIMULATOR
# ══════════════════════════════════════════════════════════
def arima_111_simulate(data, c, phi, theta):
    """Simulate ARIMA(1,1,1) fitted values and residuals."""
    n = len(data)
    fitted = [None]
    residuals = [None]
    diffs = [None] + [data[i]-data[i-1] for i in range(1,n)]
    for t in range(1, n):
        if t == 1:
            pred = data[0]
            fitted.append(pred)
            residuals.append(data[t] - pred)
        else:
            prev_diff = diffs[t-1] if diffs[t-1] is not None else 0
            prev_err = residuals[t-1] if residuals[t-1] is not None else 0
            pred_diff = c + phi * prev_diff + theta * prev_err
            pred_val = data[t-1] + pred_diff
            fitted.append(pred_val)
            residuals.append(data[t] - pred_val)
    return fitted, residuals

def arima_111_forecast(data, residuals, c, phi, theta, steps=4):
    """Forecast future values."""
    last_val = data[-1]
    last_diff = data[-1] - data[-2]
    last_err = residuals[-1] if residuals[-1] is not None else 0
    forecasts = []
    for s in range(steps):
        if s == 0:
            pred_diff = c + phi * last_diff + theta * last_err
        else:
            pred_diff = c + phi * prev_diff  # future errors = 0
        pred_val = last_val + pred_diff
        forecasts.append(pred_val)
        prev_diff = pred_diff
        last_val = pred_val
    return forecasts


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.html(f"""<div style="text-align:center;padding:15px 0 5px 0;user-select:none;">
        <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:1.3rem;font-weight:800;letter-spacing:1px;">THE MOUNTAIN PATH</div>
        <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:0.8rem;letter-spacing:3px;margin-top:2px;">ACADEMY</div>
        <div style="height:2px;background:linear-gradient(90deg,transparent,{GOLD},transparent);margin:10px auto;width:80%;"></div>
        <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.75rem;">World of Finance</div></div>""")
    st.html(f'<div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:1rem;font-weight:700;margin:20px 0 8px 0;user-select:none;">📈 Navigate</div>')
    page = st.radio("Topic", [
        "🏠 Home",
        "1️⃣ ARIMA Fundamentals",
        "2️⃣ Stationarity & Differencing",
        "3️⃣ ACF/PACF & Model Selection",
        "4️⃣ ARIMA(1,1,1) Live Simulator",
        "5️⃣ Illustrations & Case Studies",
        "6️⃣ SARIMA — Seasonal ARIMA",
        "7️⃣ Excel Solver Demo",
        "8️⃣ Where ARIMA Works vs Fails",
        "9️⃣ Q&A Practice",
    ], label_visibility="collapsed")
    st.html(f"""<div style="position:fixed;bottom:0;left:0;width:inherit;padding:12px 16px;background:rgba(10,22,40,0.95);border-top:1px solid rgba(255,215,0,0.2);text-align:center;user-select:none;">
        <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.7rem;">Prof. V. Ravichandran</div>
        <div style="margin-top:4px;"><a href="https://themountainpathacademy.com" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.7rem;text-decoration:none;">themountainpathacademy.com</a></div>
        <div style="margin-top:3px;"><a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.65rem;text-decoration:none;margin-right:8px;">LinkedIn</a><a href="https://github.com/trichyravis" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.65rem;text-decoration:none;">GitHub</a></div></div>""")


# ══════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.html(f"""<div style="text-align:center;padding:30px 20px 10px 20px;user-select:none;">
        <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:0.9rem;letter-spacing:4px;font-weight:600;">THE MOUNTAIN PATH ACADEMY</div>
        <div style="color:white;-webkit-text-fill-color:white;font-family:Playfair Display,serif;font-size:2.8rem;font-weight:800;margin-top:12px;">ARIMA Modelling</div>
        <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:1.15rem;margin-top:10px;">Interactive Learning Lab — AutoRegressive Integrated Moving Average</div>
        <div style="height:3px;background:linear-gradient(90deg,transparent,{GOLD},transparent);margin:20px auto;width:50%;"></div>
        <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.85rem;">Prof. V. Ravichandran &nbsp;|&nbsp; NMIMS Bangalore &nbsp;|&nbsp; BITS Pilani &nbsp;|&nbsp; RV University Bangalore &nbsp;|&nbsp; Goa Institute of Management</div></div>""")

    # Hero chart — Bakery case study
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=BAKERY_WEEKS, y=BAKERY_SALES, mode='lines+markers', name='Actual (30 wks)',
                            line=dict(color=GOLD, width=2.5), marker=dict(size=6)))
    fc_x = [w for w,_ in BAKERY_FORECAST]
    fc_y = [v for _,v in BAKERY_FORECAST]
    fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode='lines+markers', name='ARIMA Forecast',
                            line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=9, symbol='diamond')))
    fig.add_vline(x=30.5, line_dash="dot", line_color=ORANGE, line_width=2)
    fig = plotly_theme(fig, "Golden Crust Bakery — ARIMA(1,1,1) Forecast", 380)
    fig.update_yaxes(title="Loaves Sold")
    fig.update_xaxes(title="Week")
    st.plotly_chart(fig, use_container_width=True)

    mp_sub("📋 What You'll Learn")
    topics = [
        ("🔢", "ARIMA Fundamentals", "p, d, q parameters, the ARIMA equation, AR vs MA components, special cases"),
        ("📊", "Stationarity & Testing", "Why stationarity matters, ADF test, differencing, ACF/PACF interpretation"),
        ("🎛️", "Live Simulator", "Adjust c, φ₁, θ₁ in real-time and watch fitted values and forecasts change instantly"),
        ("🏦", "Case Studies", "Retail sales, stock prices, bakery bread, airline passengers — from the Excel workbook"),
        ("🔬", "SARIMA & Solver", "Seasonal ARIMA, ice cream quarterly data, Excel Solver for optimal parameters"),
        ("⚡", "When to Use & Not Use", "Where ARIMA works, where it fails, and better alternatives"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(topics):
        with cols[i % 3]:
            st.html(f'<div style="background:{CARD};border:1px solid rgba(255,215,0,0.15);border-radius:12px;padding:20px;text-align:center;min-height:150px;margin-bottom:8px;user-select:none;"><div style="font-size:2rem;margin-bottom:6px;">{icon}</div><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:1rem;font-weight:700;">{title}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.85rem;margin-top:6px;line-height:1.6;">{desc}</div></div>')


# ══════════════════════════════════════════════════════════
# ARIMA FUNDAMENTALS
# ══════════════════════════════════════════════════════════
elif page == "1️⃣ ARIMA Fundamentals":
    mp_header("ARIMA Fundamentals", "AutoRegressive Integrated Moving Average — the building blocks")

    mp_card(f"<b style='color:{GOLD};-webkit-text-fill-color:{GOLD};'>ARIMA</b> is one of the most widely used statistical methods for time series forecasting. It captures patterns in historical data (trends, autocorrelations) to predict future values.")

    mp_sub("🔢 The Three Parameters — ARIMA(p, d, q)")
    params = [
        ("p", "AR (AutoRegressive)", "Number of lagged observations used as predictors",
         "\"Today's value depends on yesterday's (and the day before)\"", TEAL),
        ("d", "I (Integrated)", "Number of times data is differenced to become stationary",
         "\"Remove the trend so data fluctuates around a constant mean\"", GOLD),
        ("q", "MA (Moving Average)", "Number of lagged forecast errors used as predictors",
         "\"Today's value is also influenced by recent forecast mistakes\"", ORANGE),
    ]
    cols = st.columns(3)
    for col, (sym, name, meaning, intuition, clr) in zip(cols, params):
        with col:
            st.html(f'<div style="background:{CARD};border-top:4px solid {clr};border-radius:12px;padding:20px;text-align:center;min-height:230px;user-select:none;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-family:JetBrains Mono,monospace;font-size:2.5rem;font-weight:800;">{sym}</div><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:1rem;font-weight:700;margin:6px 0;">{name}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;line-height:1.5;">{meaning}</div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.82rem;font-style:italic;margin-top:8px;">{intuition}</div></div>')

    mp_sub("📐 The ARIMA Equation")
    mp_formula("General ARIMA(p,d,q)",
               "Y'ₜ = c + φ₁Y'ₜ₋₁ + ... + φₚY'ₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θ_qεₜ₋_q",
               "Y'ₜ = differenced series | c = constant | φᵢ = AR coefficients | θⱼ = MA coefficients | εₜ = white noise")

    mp_sub("🎯 Special Cases")
    specials = [
        ("White Noise", "ARIMA(0,0,0)", "Purely random; cannot be forecasted", MUTED),
        ("Random Walk", "ARIMA(0,1,0)", "Best forecast = today's value. Stock prices", RED),
        ("AR(1)", "ARIMA(1,0,0)", "Value depends on immediately preceding value", TEAL),
        ("MA(1)", "ARIMA(0,0,1)", "Value depends on most recent forecast error", ORANGE),
        ("SARIMA", "(p,d,q)(P,D,Q)ₛ", "Seasonal ARIMA with period s", PURPLE),
    ]
    for name, params_str, desc, clr in specials:
        st.html(f'<div style="background:{CARD};border-left:4px solid {clr};border-radius:8px;padding:12px 16px;margin:5px 0;display:flex;gap:14px;align-items:center;user-select:none;"><div style="min-width:110px;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-weight:700;font-size:0.95rem;">{name}</div><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:JetBrains Mono,monospace;font-size:0.78rem;">{params_str}</div></div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;">{desc}</div></div>')

    mp_sub("📊 AR vs MA — What's the Difference?")
    c1, c2 = st.columns(2)
    with c1:
        mp_card(f"<b style='color:{TEAL};-webkit-text-fill-color:{TEAL};'>AR (AutoRegressive)</b><br><br>Current value depends on its <b>own past values</b>.<br><br>AR(1): Yₜ = c + φ₁Yₜ₋₁ + εₜ<br><br><em style='color:{MUTED};-webkit-text-fill-color:{MUTED};'>\"If sales were high last week, they tend to be high this week too.\"</em>", border=TEAL)
    with c2:
        mp_card(f"<b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>MA (Moving Average)</b><br><br>Current value depends on <b>past forecast errors</b>.<br><br>MA(1): Yₜ = c + εₜ + θ₁εₜ₋₁<br><br><em style='color:{MUTED};-webkit-text-fill-color:{MUTED};'>\"If I under-predicted last week, I correct upward this week.\"</em>", border=ORANGE)


# ══════════════════════════════════════════════════════════
# STATIONARITY & DIFFERENCING
# ══════════════════════════════════════════════════════════
elif page == "2️⃣ Stationarity & Differencing":
    mp_header("Stationarity & Differencing", "The critical prerequisite for ARIMA")

    mp_card(f"A time series is <b style='color:{GOLD};-webkit-text-fill-color:{GOLD};'>stationary</b> if its statistical properties (mean, variance, autocorrelation) do <b>not change over time</b>. ARIMA <b>requires</b> stationarity — the <b>d</b> parameter achieves this via differencing.")

    mp_sub("📊 Visual: Non-Stationary vs Stationary")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Original (Non-Stationary — has trend)", "After Differencing (Stationary)"])
    fig.add_trace(go.Scatter(x=STATION_ORIG, y=STATION_YT, mode='lines+markers', line=dict(color=GOLD, width=2.5), marker=dict(size=6), name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(2,25)), y=STATION_DIFF, mode='lines+markers', line=dict(color=GRN, width=2.5), marker=dict(size=6), name='Differenced'), row=1, col=2)
    fig.add_hline(y=np.mean(STATION_DIFF), line_dash="dash", line_color=MUTED, row=1, col=2)
    fig = plotly_theme(fig, "", 380)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    mp_sub("🔍 Differencing Formulas")
    mp_formula("First Difference (d=1)", "ΔYₜ = Yₜ − Yₜ₋₁", "Removes linear trend")
    mp_formula("Second Difference (d=2)", "Δ²Yₜ = ΔYₜ − ΔYₜ₋₁", "If still non-stationary after d=1")
    mp_formula("Seasonal Difference", "Yₜ − Yₜ₋ₛ", "Removes repeating seasonal pattern (e.g., s=4 for quarterly)")

    mp_sub("🧪 Stationarity Test — ADF (Augmented Dickey-Fuller)")
    mp_card(f"<b style='color:{GOLD};-webkit-text-fill-color:{GOLD};'>Null hypothesis:</b> The series has a unit root (non-stationary).<br><br><b style='color:{GRN};-webkit-text-fill-color:{GRN};'>If p-value < 0.05:</b> Reject null → series IS stationary ✅<br><b style='color:{RED};-webkit-text-fill-color:{RED};'>If p-value > 0.05:</b> Fail to reject → series is NOT stationary → difference again ❌", border=LB)

    mp_sub("📋 Stationarity Check — Worked Example (from Excel)")
    c1, c2 = st.columns(2)
    with c1:
        st.html(f'<div style="color:{RED};-webkit-text-fill-color:{RED};font-weight:700;font-size:1rem;margin-bottom:8px;user-select:none;">❌ Before Differencing (ACF)</div>')
        for lag, val in STATION_ACF_ORIG.items():
            st.html(f'<div style="background:{CARD};border-left:4px solid {RED};border-radius:6px;padding:8px 14px;margin:4px 0;display:flex;justify-content:space-between;user-select:none;"><span style="color:{TXT};-webkit-text-fill-color:{TXT};">{lag}</span><span style="color:{RED};-webkit-text-fill-color:{RED};font-family:JetBrains Mono,monospace;font-weight:700;">{val:.3f}</span></div>')
        mp_warn("Diagnosis", "ACF decays <b>slowly</b> (all > 0.9) → strong trend → <b>NOT stationary</b>")
    with c2:
        st.html(f'<div style="color:{GRN};-webkit-text-fill-color:{GRN};font-weight:700;font-size:1rem;margin-bottom:8px;user-select:none;">✅ After Differencing (d=1)</div>')
        for lag, val in STATION_ACF_DIFF.items():
            st.html(f'<div style="background:{CARD};border-left:4px solid {GRN};border-radius:6px;padding:8px 14px;margin:4px 0;display:flex;justify-content:space-between;user-select:none;"><span style="color:{TXT};-webkit-text-fill-color:{TXT};">{lag}</span><span style="color:{GRN};-webkit-text-fill-color:{GRN};font-family:JetBrains Mono,monospace;font-weight:700;">{val:.3f}</span></div>')
        mp_insight("Diagnosis", "ACF drops to near-zero → trend removed → <b>STATIONARY</b> ✅")

    mp_insight("Over-Differencing Warning", "If the first lag of ACF is strongly negative (near −0.5), you may have <b>over-differenced</b>. Try reducing d by 1.")


# ══════════════════════════════════════════════════════════
# ACF/PACF & MODEL SELECTION
# ══════════════════════════════════════════════════════════
elif page == "3️⃣ ACF/PACF & Model Selection":
    mp_header("ACF/PACF & Model Selection", "How to determine p, d, q from the data")

    mp_card(f"<b style='color:{TEAL};-webkit-text-fill-color:{TEAL};'>ACF (AutoCorrelation Function)</b> shows <b>total</b> correlation including indirect effects → determines <b>q</b> (MA order).<br><br><b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>PACF (Partial ACF)</b> shows only <b>direct</b> correlation, removing intermediate lags → determines <b>p</b> (AR order).")

    mp_sub("📊 ACF Pattern Guide")
    patterns = [
        ("Decays slowly (high r at many lags)", "Series has a TREND", "❌ NO", "Apply differencing (d=1 or d=2)", RED),
        ("Drops quickly then near-zero", "Stationary!", "✅ YES", "Ready for ARIMA", GRN),
        ("Spikes at lag s, 2s, 3s", "SEASONAL pattern", "❌ NO", "Seasonal differencing (D=1)", ORANGE),
        ("Alternating +/−", "Possible over-differencing", "⚠️ Check", "Reduce d by 1", WARN),
        ("Single spike at lag 1, rest ~0", "Suggests MA(1)", "✅ YES", "Set q=1", TEAL),
    ]
    for pattern, meaning, stat, action, clr in patterns:
        st.html(f'<div style="background:{CARD};border-left:4px solid {clr};border-radius:8px;padding:12px 16px;margin:5px 0;user-select:none;"><div style="display:flex;gap:12px;align-items:center;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-weight:700;font-size:0.82rem;min-width:28px;">{stat}</div><div style="flex:1;"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.9rem;"><b>{pattern}</b> → {meaning}</div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.82rem;margin-top:2px;">Action: {action}</div></div></div></div>')

    mp_sub("📐 PACF Quick Guide (for p)")
    mp_card(f"• PACF cuts off sharply after lag k → set <b>p = k</b><br>• PACF decays gradually → AR is not dominant; focus on MA (q) via ACF<br>• Both ACF and PACF decay → likely need BOTH AR and MA terms<br>• <b>Significance threshold:</b> ±2/√n (for n=24, threshold = ±0.408)", border=LB)

    mp_sub("📋 The Box-Jenkins 7-Step Methodology")
    steps = [
        ("Visualise the data", "Plot time series. Look for trends, seasonality, outliers."),
        ("Test for stationarity", "Use ADF test. If p > 0.05, apply differencing."),
        ("Determine d", "Difference until stationary. Count how many times = d."),
        ("Examine ACF & PACF", "ACF → q (MA order). PACF → p (AR order). Look for significant spikes."),
        ("Fit candidate models", "Try several (p,d,q). Compare AIC/BIC (lower = better)."),
        ("Diagnose residuals", "Must be white noise (no autocorrelation). Ljung-Box test (p > 0.05)."),
        ("Forecast", "Generate point forecasts and confidence intervals."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        mp_step(str(i), title, desc)

    mp_insight("AIC & BIC", "AIC = 2k − 2ln(L) &nbsp;|&nbsp; BIC = k·ln(n) − 2ln(L)<br>k = parameters, n = observations, L = likelihood. <b>Lower = better.</b> BIC penalises complexity more → prefers simpler models (parsimony).")


# ══════════════════════════════════════════════════════════
# LIVE SIMULATOR
# ══════════════════════════════════════════════════════════
elif page == "4️⃣ ARIMA(1,1,1) Live Simulator":
    mp_header("ARIMA(1,1,1) Live Simulator", "Adjust parameters in real-time and watch the model respond")

    mp_formula("ARIMA(1,1,1) Equation",
               "ΔŶₜ = c + φ₁ × ΔYₜ₋₁ + θ₁ × εₜ₋₁",
               "Then: Fitted Value = Previous Actual + ΔŶₜ")

    mp_sub("🎛️ Adjust Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        c_val = st.slider("Constant (c)", -5.0, 10.0, 2.5, 0.1, help="Base expected change per period")
    with c2:
        phi_val = st.slider("φ₁ (AR coefficient)", -0.99, 0.99, 0.40, 0.01, help="How much last period's change carries forward")
    with c3:
        theta_val = st.slider("θ₁ (MA coefficient)", -0.99, 0.99, -0.25, 0.01, help="Error correction strength")

    mp_sub("📊 Choose Dataset")
    dataset = st.selectbox("Dataset", ["Bakery Bread Sales (30 weeks)", "Retail Sales (24 months)"])

    if "Bakery" in dataset:
        data = BAKERY_SALES; labels = [f"Wk {w}" for w in BAKERY_WEEKS]; x_title = "Week"
    else:
        data = RETAIL_SALES; labels = RETAIL_MONTHS; x_title = "Month"

    fitted, residuals = arima_111_simulate(data, c_val, phi_val, theta_val)
    forecasts = arima_111_forecast(data, residuals, c_val, phi_val, theta_val, steps=6)

    # Metrics
    valid_res = [r for r in residuals if r is not None]
    mae = np.mean([abs(r) for r in valid_res])
    rmse = np.sqrt(np.mean([r**2 for r in valid_res]))
    ssr = sum(r**2 for r in valid_res)
    mean_res = np.mean(valid_res)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("MAE", f"{mae:.2f}")
    with c2: st.metric("RMSE", f"{rmse:.2f}")
    with c3: st.metric("SSR", f"{ssr:.1f}")
    with c4: st.metric("Mean Residual", f"{mean_res:.2f}")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=data, mode='lines+markers', name='Actual',
                            line=dict(color=GOLD, width=2.5), marker=dict(size=6)))
    fit_x = [labels[i] for i in range(len(fitted)) if fitted[i] is not None]
    fit_y = [v for v in fitted if v is not None]
    fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', name='Fitted',
                            line=dict(color=TEAL, width=2, dash='dot')))
    fc_labels = [f"F+{i+1}" for i in range(len(forecasts))]
    fig.add_trace(go.Scatter(x=fc_labels, y=forecasts, mode='lines+markers', name='Forecast',
                            line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=9, symbol='diamond')))
    fig = plotly_theme(fig, f"ARIMA(1,1,1) — c={c_val}, φ₁={phi_val}, θ₁={theta_val}", 450)
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    mp_sub("📉 Residuals (should look like random noise)")
    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(x=labels[1:], y=valid_res,
                            marker_color=[GRN if v > 0 else RED for v in valid_res]))
    fig_res.add_hline(y=0, line_color=MUTED, line_width=1.5)
    fig_res = plotly_theme(fig_res, "Residuals — Good Model = Random Pattern, Near-Zero Mean", 300)
    st.plotly_chart(fig_res, use_container_width=True)

    mp_insight("What Good Residuals Look Like",
        "Residuals should show <b>no pattern</b> (random scatter around zero). If you see waves or trends, the model is missing structure. "
        "The Ljung-Box test formally checks this: p-value > 0.05 means residuals are white noise ✅")


# ══════════════════════════════════════════════════════════
# ILLUSTRATIONS & CASE STUDIES
# ══════════════════════════════════════════════════════════
elif page == "5️⃣ Illustrations & Case Studies":
    mp_header("Illustrations & Case Studies", "From the Excel workbook — real-world ARIMA applications")

    tabs = st.tabs(["🛒 Retail Sales", "📈 Stock Prices", "🍞 Bakery", "✈️ Airline"])

    with tabs[0]:
        mp_sub("Illustration 1: Monthly Retail Sales — ARIMA(1,1,1)")
        mp_card(f"<b>Parameters:</b> c = 1.2, φ₁ = 0.35, θ₁ = −0.20<br><b>Interpretation:</b> Forecast = c + φ₁ × (previous 1st difference) + θ₁ × (previous error)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=RETAIL_MONTHS, y=RETAIL_SALES, mode='lines+markers', name='Actual', line=dict(color=GOLD, width=2.5), marker=dict(size=6)))
        fit_r = [v for v in RETAIL_FITTED if v is not None]
        fig.add_trace(go.Scatter(x=RETAIL_MONTHS[1:], y=fit_r, mode='lines', name='Fitted', line=dict(color=TEAL, width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=RETAIL_FORECAST_M, y=RETAIL_FORECAST_V, mode='lines+markers', name='6-Month Forecast', line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=9, symbol='diamond')))
        fig = plotly_theme(fig, "Monthly Retail Sales ($000s) — ARIMA(1,1,1)", 430)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        mp_sub("Illustration 2: Stock Prices — Why ARIMA Fails")
        mp_warn("Key Insight", "ARIMA on stock prices typically converges to <b>ARIMA(0,1,0) = Random Walk</b>. Tomorrow's best forecast is today's price — a <b>flat line</b>.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=STOCK_DAYS, y=STOCK_PRICE, mode='lines+markers', name='Actual Price', line=dict(color=GOLD, width=2.5), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=[20,21,22], y=[113.3,113.3,113.3], mode='lines+markers', name='Random Walk Forecast', line=dict(color=RED, width=3, dash='dash'), marker=dict(size=8, symbol='x')))
        fig = plotly_theme(fig, "Stock Price — Random Walk Produces Flat Line Forecast", 400)
        st.plotly_chart(fig, use_container_width=True)
        reasons = ["Stock price changes (returns) are nearly unpredictable","Returns show very low autocorrelation (past ≠ future)",
                   "Prices are driven by new information, not past patterns","Consistent with the Efficient Market Hypothesis"]
        for r in reasons:
            st.html(f'<div style="background:{CARD};border-left:3px solid {RED};border-radius:6px;padding:8px 14px;margin:4px 0;user-select:none;"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;">{r}</div></div>')

    with tabs[2]:
        mp_sub("Case Study 1: Golden Crust Bakery — ARIMA(1,1,1)")
        mp_card(f"<b>30 weeks of data</b> showing gradual upward trend.<br><b>Parameters:</b> c = 2.5, φ₁ = 0.40, θ₁ = −0.25<br><b>Equation:</b> ΔŶₜ = 2.5 + 0.40 × ΔYₜ₋₁ + (−0.25) × εₜ₋₁")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=BAKERY_WEEKS, y=BAKERY_SALES, mode='lines+markers', name='Actual', line=dict(color=GOLD, width=2.5), marker=dict(size=5)))
        fc_x = [w for w,_ in BAKERY_FORECAST]; fc_y = [v for _,v in BAKERY_FORECAST]
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode='lines+markers', name='4-Wk Forecast', line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=9, symbol='diamond')))
        fig.add_vline(x=30.5, line_dash="dot", line_color=ORANGE, line_width=2)
        fig = plotly_theme(fig, "Bakery Bread Sales — Actual + ARIMA Forecast", 420)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Mean Residual", "0.50")
        with c2: st.metric("MAE", "10.81 loaves")
        with c3: st.metric("RMSE", "10.86 loaves")
        mp_insight("Key Takeaway", "Forecast growth slows and flattens beyond 4 weeks — typical ARIMA behaviour. Re-estimate regularly with new data.")

    with tabs[3]:
        mp_sub("Case Study 2: SkyLine Airline — SARIMA(0,1,1)(0,1,1)₁₂")
        mp_card(f"<b>36 months of data</b> with clear upward trend + 12-month seasonal cycle (summer peak July–Aug, winter trough Nov).<br><b>SARIMA Equation:</b> Ŷₜ = Yₜ₋₁ + Yₜ₋₁₂ − Yₜ₋₁₃ + θ₁εₜ₋₁ + Θ₁εₜ₋₁₂ + θ₁Θ₁εₜ₋₁₃")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=AIRLINE_MONTHS_LBL, y=AIRLINE_Y3, mode='lines+markers', name='Year 3 Actual', line=dict(color=GOLD, width=2.5), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=AIRLINE_MONTHS_LBL, y=AIRLINE_FCAST, mode='lines+markers', name='Year 4 SARIMA Forecast', line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=8, symbol='diamond')))
        fig = plotly_theme(fig, "Airline Passengers — Year 3 Actual vs Year 4 Forecast", 420)
        fig.update_yaxes(title="Passengers (000s)")
        st.plotly_chart(fig, use_container_width=True)

        mp_sub("Seasonal Profile")
        for month, idx, pattern in AIRLINE_SEASONAL:
            pct = idx * 100
            bar_w = max(int(pct * 0.6), 10)
            clr = GRN if idx > 1.05 else (RED if idx < 0.9 else GOLD)
            st.html(f'<div style="background:{CARD};border-radius:6px;padding:6px 14px;margin:3px 0;display:flex;align-items:center;gap:10px;user-select:none;"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.85rem;min-width:40px;">{month}</div><div style="background:{clr};height:16px;width:{bar_w}px;border-radius:4px;"></div><div style="color:{clr};-webkit-text-fill-color:{clr};font-family:JetBrains Mono,monospace;font-size:0.82rem;">{idx:.3f}</div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.78rem;">{pattern}</div></div>')


# ══════════════════════════════════════════════════════════
# SARIMA
# ══════════════════════════════════════════════════════════
elif page == "6️⃣ SARIMA — Seasonal ARIMA":
    mp_header("SARIMA — Seasonal ARIMA", "Extending ARIMA for data with repeating seasonal patterns")

    mp_formula("SARIMA Notation", "SARIMA(p, d, q)(P, D, Q)ₛ",
               "p,d,q = non-seasonal | P = seasonal AR | D = seasonal differencing | Q = seasonal MA | s = period (4=quarterly, 12=monthly)")

    mp_sub("🍦 Ice Cream Sales — SARIMA(1,1,1)(1,1,0)₄")
    fig = go.Figure()
    x_labels = [f"{ICE_QTRS[i]}'{str(ICE_YEARS[i])[2:]}" for i in range(len(ICE_SALES))]
    fig.add_trace(go.Scatter(x=x_labels, y=ICE_SALES, mode='lines+markers', name='Actual', line=dict(color=GOLD, width=2.5), marker=dict(size=7)))
    fc_x2 = [lbl for lbl, _ in ICE_FORECAST]
    fc_y2 = [v for _, v in ICE_FORECAST]
    fig.add_trace(go.Scatter(x=fc_x2, y=fc_y2, mode='lines+markers', name='SARIMA Forecast', line=dict(color=GRN, width=3, dash='dash'), marker=dict(size=9, symbol='diamond')))
    fig = plotly_theme(fig, "Quarterly Ice Cream Sales — Actual + SARIMA Forecast", 430)
    fig.update_yaxes(title="Sales (000s)")
    st.plotly_chart(fig, use_container_width=True)

    mp_sub("📊 Seasonal Indices")
    cols = st.columns(4)
    colors_q = [RED, ORANGE, GRN, GOLD]
    labels_q = ["Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Fall)"]
    for i, (q, idx) in enumerate(ICE_SEASONAL.items()):
        with cols[i]:
            st.metric(labels_q[i], f"{idx:.3f}x")

    mp_insight("Reading Seasonal Indices", "Q3 (Summer) sales are <b>1.771× the trend level</b> — nearly double! Q1 (Winter) is only <b>0.468×</b>. The SARIMA model captures this repeating summer-peak/winter-trough pattern automatically.")

    mp_sub("🔧 What Makes SARIMA Different from ARIMA?")
    mp_card(f"""SARIMA differences the data <b>both consecutively</b> (Yₜ − Yₜ₋₁) <b>AND seasonally</b> (Yₜ − Yₜ₋ₛ). This captures:
    <br><br>• <b>Short-term dynamics:</b> How this period relates to last period (via p, d, q)
    <br>• <b>Seasonal dynamics:</b> How this period relates to the same period last year (via P, D, Q)
    <br><br>Example: SARIMA(1,1,1)(1,1,0)₄ applies both regular and seasonal differencing, then fits AR and MA terms at both levels.""", border=PURPLE)


# ══════════════════════════════════════════════════════════
# EXCEL SOLVER
# ══════════════════════════════════════════════════════════
elif page == "7️⃣ Excel Solver Demo":
    mp_header("Excel Solver for ARIMA Parameters", "Finding optimal c, φ₁, θ₁ by minimising Sum of Squared Residuals")

    mp_card(f"Instead of guessing ARIMA parameters, <b>Excel Solver</b> finds values that <b>minimise</b> forecast errors. This is mathematically equivalent to <b>Maximum Likelihood Estimation</b> under normally distributed errors.")

    mp_sub("📋 Step-by-Step Solver Instructions")
    solver_steps = [
        ("Enable Solver", "File → Options → Add-ins → Manage: Excel Add-ins → ☑ Solver Add-in"),
        ("Set Objective", "Click 'Set Objective' → select the SSR cell (Σεₜ²)"),
        ("Set to: Min", "Select the 'Min' radio button — we want to MINIMISE total error"),
        ("Changing Variable Cells", "Select c, φ₁, θ₁ cells — these are what Solver will adjust"),
        ("Add Constraints", "φ₁ ≥ −1 and ≤ 1 (stationarity) | θ₁ ≥ −1 and ≤ 1 (invertibility)"),
        ("Solving Method", "Select 'GRG Nonlinear' — ARIMA is nonlinear due to MA terms"),
        ("Click Solve!", "Solver updates parameter cells with optimal values — no more guessing!"),
    ]
    for i, (title, desc) in enumerate(solver_steps, 1):
        mp_step(str(i), title, desc)

    mp_sub("⚙️ What Happens Under the Hood")
    mp_card(f"""<b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>1.</b> Solver tries different values for c, φ₁, and θ₁<br>
    <b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>2.</b> Each combination recalculates ALL fitted values and residuals<br>
    <b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>3.</b> The SSR (Σεₜ²) changes with each iteration<br>
    <b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>4.</b> Solver stops when it finds the combination that makes SSR as small as possible<br>
    <b style='color:{ORANGE};-webkit-text-fill-color:{ORANGE};'>5.</b> The parameter cells now contain the OPTIMAL values""")

    mp_sub("💡 Tips")
    tips = ["Try different initial guesses to confirm Solver finds the same answer (global vs local minimum)",
            "Set Convergence to 0.00001 in Options for more precision",
            "Check residuals after solving — if patterns remain, try a different (p,d,q)",
            "This works for ANY ARIMA(p,d,q) — just add more parameter cells"]
    for t in tips:
        st.html(f'<div style="background:rgba(40,167,69,0.08);border-left:3px solid {GRN};border-radius:6px;padding:8px 14px;margin:4px 0;user-select:none;"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;">{t}</div></div>')


# ══════════════════════════════════════════════════════════
# WHERE ARIMA WORKS VS FAILS
# ══════════════════════════════════════════════════════════
elif page == "8️⃣ Where ARIMA Works vs Fails":
    mp_header("Where ARIMA Works vs Fails", "Assumptions, limitations, and better alternatives")

    tabs = st.tabs(["✅ Works Well", "❌ Should NOT Use", "⚠️ Assumptions", "📊 vs Other Methods"])

    with tabs[0]:
        mp_sub("✅ Where ARIMA Works Well")
        works = [
            ("🛒 Retail sales", "Strong seasonal patterns with steady trends", "Monthly supermarket revenue", GRN),
            ("⚡ Energy demand", "Electricity/gas usage follows clear cycles", "Daily electricity load", TEAL),
            ("🏭 Manufacturing", "Production volumes are relatively stable", "Monthly widget production", GOLD),
            ("🏨 Tourism & hospitality", "Highly seasonal and predictable", "Monthly hotel bookings", ORANGE),
            ("📦 Inventory management", "Demand for staples is stable and autocorrelated", "Weekly grocery restocking", LB),
            ("📊 Economic indicators", "GDP, CPI move gradually with momentum", "Quarterly GDP, monthly CPI", PURPLE),
        ]
        for icon, why, example, clr in works:
            st.html(f'<div style="background:{CARD};border-left:4px solid {clr};border-radius:8px;padding:12px 16px;margin:6px 0;user-select:none;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-weight:700;font-size:0.95rem;">{icon}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;margin-top:3px;">{why}</div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.8rem;">{example}</div></div>')

    with tabs[1]:
        mp_sub("❌ Where ARIMA Should NOT Be Used")
        fails = [
            ("📈 Stock prices", "Returns are near-random; flat-line forecast", "GARCH for volatility; ML models", RED),
            ("₿ Cryptocurrency", "Extreme volatility, regime shifts", "Deep learning (LSTM, Transformer)", RED),
            ("🦠 Pandemic forecasting", "Massive structural break", "SIR models; scenario analysis", ORANGE),
            ("📱 Social media trends", "Exponential spikes; no stable ACF", "Event-driven models; Bayesian", WARN),
            ("🔗 Multi-factor problems", "ARIMA ignores external variables", "ARIMAX, VAR, ML with features", TEAL),
            ("📅 Long-horizon (>12)", "Converges to mean; useless confidence intervals", "Structural models; domain expertise", MUTED),
        ]
        for icon, why, alt, clr in fails:
            st.html(f'<div style="background:{CARD};border-left:4px solid {clr};border-radius:8px;padding:12px 16px;margin:6px 0;user-select:none;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-weight:700;font-size:0.95rem;">{icon}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.88rem;margin-top:3px;">{why}</div><div style="color:{GRN};-webkit-text-fill-color:{GRN};font-size:0.82rem;">Better: {alt}</div></div>')

    with tabs[2]:
        mp_sub("⚠️ 6 Key Assumptions")
        assumptions = [
            ("Stationarity after differencing", "Mean/variance must be constant. If variance changes, ARIMA alone won't handle it."),
            ("Linearity", "Relationships between past and future are LINEAR. Cannot capture exponential growth."),
            ("No structural breaks", "Data-generating process must be consistent. Pandemic/regime shift breaks it."),
            ("Sufficient data", "Typically 50+ observations, ideally 100+. Fewer → unreliable estimates."),
            ("No exogenous shocks", "Uses ONLY past values. Ignores weather, competitors, policy changes."),
            ("Normally distributed errors", "Residuals should be white noise with normal distribution."),
        ]
        for i, (title, desc) in enumerate(assumptions, 1):
            mp_step(str(i), title, desc)

    with tabs[3]:
        mp_sub("📊 ARIMA vs Other Methods")
        methods = [
            ("ARIMA", "Interpretable; well-understood; needs less data", "Univariate; linear only; manual tuning", "50–1000 points; stable", GOLD),
            ("Exp. Smoothing", "Simpler; handles trend/season intuitively", "Less flexible ACF structure", "Quick baseline forecasts", TEAL),
            ("GARCH", "Models time-varying volatility", "Only variance, not level", "Financial risk management", ORANGE),
            ("VAR", "Multivariate; captures interdependencies", "Needs many observations", "Multiple related series", PURPLE),
            ("ML (LSTM, XGB)", "Nonlinear; many features; large data", "Black-box; needs lots of data", ">1000 points; complex", RED),
        ]
        for name, strength, weakness, use, clr in methods:
            st.html(f'<div style="background:{CARD};border-left:4px solid {clr};border-radius:10px;padding:14px 18px;margin:8px 0;user-select:none;"><div style="color:{clr};-webkit-text-fill-color:{clr};font-weight:700;font-size:1rem;">{name}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.85rem;margin-top:4px;"><span style="color:{GRN};-webkit-text-fill-color:{GRN};">✅</span> {strength}</div><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:0.85rem;"><span style="color:{RED};-webkit-text-fill-color:{RED};">❌</span> {weakness}</div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.82rem;">Best for: {use}</div></div>')

        mp_insight("ARIMA vs Machine Learning", "For univariate series with <b>&lt;1000 data points</b>, ARIMA often matches or beats ML models (including LSTM, XGBoost). ML excels with many features, very large datasets, and complex nonlinear patterns. <b>Use the right tool for the problem.</b>")


# ══════════════════════════════════════════════════════════
# Q&A PRACTICE
# ══════════════════════════════════════════════════════════
elif page == "9️⃣ Q&A Practice":
    mp_header("Q&A Practice — Self-Assessment", "35 questions from the Excel workbook covering theory, practice & pitfalls")
    mp_card("💡 Try answering each question in your head <b>before</b> clicking to reveal the answer.")

    qa_sections = {
        "Fundamentals (Q1–Q10)": [
            ("Q1","What does ARIMA stand for?","AutoRegressive Integrated Moving Average. <b style='color:#FFD700;-webkit-text-fill-color:#FFD700;'>AR</b> = regression on own past values. <b style='color:#FFD700;-webkit-text-fill-color:#FFD700;'>I</b> = differencing for stationarity. <b style='color:#FFD700;-webkit-text-fill-color:#FFD700;'>MA</b> = modelling past error terms."),
            ("Q2","What do p, d, q represent?","<b style='color:#17a2b8;-webkit-text-fill-color:#17a2b8;'>p</b> = AR lags. <b style='color:#FFD700;-webkit-text-fill-color:#FFD700;'>d</b> = differencing order. <b style='color:#fd7e14;-webkit-text-fill-color:#fd7e14;'>q</b> = MA lags. ARIMA(1,1,1) uses 1 lag of past values, 1 round of differencing, 1 lag of past errors."),
            ("Q3","What is stationarity?","Constant mean, variance, and autocorrelation over time. ARIMA requires it because coefficients are estimated assuming properties don&rsquo;t change."),
            ("Q4","How do you make a non-stationary series stationary?","<b style='color:#FFD700;-webkit-text-fill-color:#FFD700;'>Differencing:</b> subtract each value from its predecessor. d=1 removes linear trend. Log transform for changing variance. Seasonal differencing (Yₜ &minus; Yₜ₋ₛ) for seasonal patterns."),
            ("Q5","What is the ADF test?","Tests for unit root (non-stationarity). If <b style='color:#28a745;-webkit-text-fill-color:#28a745;'>p-value &lt; 0.05</b>, reject null &rarr; series is stationary. If p &gt; 0.05, need more differencing."),
            ("Q6","What is autocorrelation?","How much a value at time t correlates with values at earlier time points (lags). High autocorrelation at lag 1 means yesterday strongly predicts today."),
            ("Q7","ACF vs PACF?","<b style='color:#17a2b8;-webkit-text-fill-color:#17a2b8;'>ACF:</b> total correlation (including indirect) &rarr; determines q. <b style='color:#fd7e14;-webkit-text-fill-color:#fd7e14;'>PACF:</b> direct correlation only &rarr; determines p. Sharp PACF cutoff after lag k &rarr; p=k."),
            ("Q8","What is a random walk?","ARIMA(0,1,0): Yₜ = Yₜ₋₁ + εₜ. Best forecast = today&rsquo;s value. Stock prices often behave this way. Forecast is a flat line."),
            ("Q9","What is white noise?","Uncorrelated random variables with zero mean and constant variance. If residuals are white noise, the model has captured all predictable structure."),
            ("Q10","AR vs MA components?","<b style='color:#17a2b8;-webkit-text-fill-color:#17a2b8;'>AR:</b> current value depends on past values (Yₜ = c + &phi;₁Yₜ₋₁ + &epsilon;ₜ). <b style='color:#fd7e14;-webkit-text-fill-color:#fd7e14;'>MA:</b> current value depends on past errors (Yₜ = c + &epsilon;ₜ + &theta;₁&epsilon;ₜ₋₁)."),
        ],
        "Practice (Q11–Q20)": [
            ("Q11","How to choose p, d, q?","d: difference until ADF passes. PACF cutoff &rarr; p. ACF cutoff &rarr; q. Fit several candidates, compare AIC/BIC. Or use auto.arima."),
            ("Q12","What are AIC and BIC?","Model fit penalised for complexity. Lower = better. BIC penalises more heavily &rarr; prefers simpler models (parsimony)."),
            ("Q13","How many data points?","Minimum 50, ideally 100+. For SARIMA: at least 2 full seasonal cycles. Fewer &rarr; unreliable estimates, wide confidence intervals."),
            ("Q14","How to check model quality?","1) Residuals = white noise. 2) Ljung-Box test p &gt; 0.05. 3) Compare AIC/BIC. 4) Out-of-sample RMSE/MAE/MAPE."),
            ("Q15","What is the Ljung-Box test?","Tests if residuals are independently distributed. p &gt; 0.05 = residuals are white noise (&radic;). p &lt; 0.05 = model is incomplete."),
            ("Q16","ARIMA vs SARIMA?","SARIMA adds seasonal components: (P,D,Q)ₛ. Captures patterns repeating every s periods. Essential for data with seasonal cycles."),
            ("Q17","Can ARIMA handle missing values?","No. Must impute first: interpolation, LOCF, seasonal average, or Kalman smoothing."),
            ("Q18","Software for ARIMA?","Python: statsmodels, pmdarima. R: forecast package. Excel: Solver + manual setup. SAS, SPSS, Stata all have ARIMA procedures."),
            ("Q19","What is overfitting in ARIMA?","Too many parameters fitting noise, not signal. Signs: great in-sample, poor out-of-sample. Prevention: use BIC, cross-validate, prefer simpler models."),
            ("Q20","How far ahead can ARIMA forecast?","1&ndash;3 periods: good. 4&ndash;12: increasing uncertainty. Beyond 12: converges to mean with very wide intervals. Horizon &le; 20&ndash;25% of data length."),
        ],
        "Advanced (Q21–Q28)": [
            ("Q21","What is ARIMAX?","ARIMA with eXogenous variables. Adds external predictors (e.g., temperature for electricity demand). Addresses the univariate limitation."),
            ("Q22","ARIMA vs Exponential Smoothing?","Both are univariate. ETS uses weighted averages with exponentially decreasing weights. ARIMA uses lagged values and errors. ETS is simpler; ARIMA handles complex autocorrelation better."),
            ("Q23","What is a VAR model?","Vector AutoRegression &mdash; models multiple series simultaneously. Use when Series A and B influence each other (e.g., GDP, inflation, interest rates)."),
            ("Q24","What is GARCH?","Models time-varying <b>volatility</b>. ARIMA models the level; GARCH models how variance changes. Often combined: ARIMA for mean, GARCH for variance. Essential for financial risk."),
            ("Q25","Why does ARIMA fail on stocks?","Returns are near-random. ARIMA converges to a random walk &mdash; flat line forecast. Near-zero autocorrelation in returns. Efficient Market Hypothesis."),
            ("Q26","What if you over-difference?","Introduces artificial negative autocorrelation. Makes series noisier. Sign: ACF lag 1 strongly negative (&asymp; &minus;0.5). Reduce d."),
            ("Q27","Can ARIMA detect causation?","<b style='color:#dc3545;-webkit-text-fill-color:#dc3545;'>No.</b> It is correlation-based. It identifies patterns but cannot determine WHY. For causal inference, use experiments or structural models."),
            ("Q28","What is a structural break?","Sudden permanent change (pandemic, regulation). ARIMA estimates from all history equally, so a break contaminates estimates. Model forecasts the OLD regime."),
        ],
        "Bonus (Q29–Q35)": [
            ("Q29","Confidence intervals &mdash; why do they widen?","Forecast uncertainty accumulates each period. 1-step = tight. 10+ periods = very wide. This is a fundamental property, not a flaw."),
            ("Q30","ARIMA vs Machine Learning?","For univariate &lt;1000 points, ARIMA often matches or beats ML. ML excels with many features, large data, nonlinear patterns. ARIMA: interpretable, less data, easier to validate."),
            ("Q31","What is Box-Jenkins methodology?","3-step approach: 1) <b>Identification</b> (ACF/PACF, determine p,d,q). 2) <b>Estimation</b> (fit via MLE). 3) <b>Diagnostic checking</b> (residuals = white noise?). If fail, return to step 1."),
            ("Q32","Can you combine ARIMA with other methods?","Yes! ARIMA + GARCH for volatility. ARIMA + Neural Network for nonlinear residuals. ARIMAX for external predictors. Ensemble averaging of ARIMA + ETS."),
            ("Q33","Principle of parsimony?","Prefer the simplest adequate model. ARIMA(1,1,1) over ARIMA(3,1,3) if both fit similarly. BIC enforces this naturally."),
            ("Q34","What is a unit root?","Makes a series non-stationary. Shocks have permanent effect (memory never fades). Differencing removes it. ADF test detects it."),
            ("Q35","When to use vs NOT use ARIMA?","<b style='color:#28a745;-webkit-text-fill-color:#28a745;'>USE:</b> 50+ points, stable patterns, short-term (1&ndash;12 periods), no critical external factors.<br><b style='color:#dc3545;-webkit-text-fill-color:#dc3545;'>DON&rsquo;T:</b> structural breaks, volatile/nonlinear data, long horizon (&gt;12), external factors dominate, &lt;30 points."),
        ]
    }

    for sec_name, questions in qa_sections.items():
        mp_sub(sec_name)
        qa_html = ""
        for qid, question, answer in questions:
            qa_html += f"""<details style="background:#112240;border:1px solid rgba(255,215,0,0.18);border-radius:10px;margin:8px 0;overflow:hidden;">
                <summary style="padding:14px 20px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px;user-select:none;">
                    <span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-family:JetBrains Mono,monospace;font-size:0.82rem;font-weight:700;min-width:32px;">{qid}</span>
                    <span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-size:0.92rem;line-height:1.5;">{question}</span>
                    <span style="margin-left:auto;color:#FFD700;-webkit-text-fill-color:#FFD700;font-size:1.1rem;">▶</span>
                </summary>
                <div style="padding:0 20px 16px 20px;border-top:1px solid rgba(255,215,0,0.12);">
                    <div style="background:rgba(0,51,102,0.35);border-left:4px solid #FFD700;border-radius:0 8px 8px 0;padding:14px 18px;margin-top:12px;">
                        <div style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-size:0.75rem;font-weight:700;letter-spacing:1px;margin-bottom:8px;">ANSWER</div>
                        <div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-size:0.92rem;line-height:1.85;">{answer}</div>
                    </div>
                </div>
            </details>"""
        st.html(f'<style>details summary::-webkit-details-marker{{display:none;}}details[open] summary span:last-child{{transform:rotate(90deg);}}</style>{qa_html}')


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.html(f"""<div style="text-align:center;padding:30px 0 15px 0;margin-top:40px;border-top:1px solid rgba(255,215,0,0.2);user-select:none;">
    <div style="height:2px;background:linear-gradient(90deg,transparent,{GOLD},transparent);margin:0 auto 18px auto;width:40%;"></div>
    <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;">The Mountain Path Academy</div>
    <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.78rem;margin-top:4px;">World of Finance — Prof. V. Ravichandran</div>
    <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:0.72rem;margin-top:2px;">NMIMS Bangalore | BITS Pilani | RV University Bangalore | Goa Institute of Management</div>
    <div style="margin-top:10px;"><a href="https://themountainpathacademy.com" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.82rem;text-decoration:none;font-weight:600;">themountainpathacademy.com</a></div>
    <div style="margin-top:6px;"><a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.75rem;text-decoration:none;margin-right:12px;">LinkedIn</a><a href="https://github.com/trichyravis" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:0.75rem;text-decoration:none;">GitHub</a></div></div>""")

"""
=============================================================================
PROJECT 1: Walmart Demand Forecasting & Inventory Optimization Dashboard
=============================================================================
Author : Pallapolu Bhuvan Chandra — BITS Pilani
Dataset: Walmart Store Sales (Kaggle)
Stack  : Python | pandas | statsmodels | matplotlib | Streamlit

Business Problem:
    Walmart operates 45 stores across regions. Demand is driven by
    seasonality, markdowns, holidays (Super Bowl, Thanksgiving, Christmas)
    and macro factors (CPI, fuel price, unemployment).
    
    This dashboard:
    1. Forecasts weekly department-level demand using SARIMA
    2. Derives optimal inventory policy (EOQ + Safety Stock)
    3. Quantifies the cost of under/over-stocking at each service level
    4. Lets users stress-test assumptions interactively
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCM Demand Forecasting | Bhuvan Chandra",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:700; color:#0D1F3C; }
    .sub-title    { font-size:1.1rem; color:#64748B; margin-top:-10px; }
    .metric-card  { background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:10px; padding:16px; text-align:center; }
    .metric-val   { font-size:1.8rem; font-weight:700; color:#2563EB; }
    .metric-label { font-size:0.85rem; color:#64748B; }
    .insight-box  { background:#EFF6FF; border-left:4px solid #2563EB;
                    padding:12px 16px; border-radius:4px; font-size:0.92rem; }
    .warn-box     { background:#FEF3C7; border-left:4px solid #F59E0B;
                    padding:12px 16px; border-radius:4px; font-size:0.92rem; }
    section[data-testid="stSidebar"] { background:#0D1F3C; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA — Walmart Store Sales dataset (Kaggle, 421,570 rows)
# Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_walmart_data():
    df = pd.read_csv('walmart data/train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Weekly_Sales'] > 0].copy()
    return df


@st.cache_data
def run_sarima(series, forecast_weeks=12):
    """Fit SARIMA(1,1,1)(1,1,0,52) and return forecast + confidence interval.
    Also returns a 52-week forecast for forecast-driven EOQ/ROP calculations."""
    model  = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,0,52),
                     enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    fc     = result.get_forecast(steps=forecast_weeks)
    mean   = fc.predicted_mean
    ci     = fc.conf_int(alpha=0.05)
    # 52-week forecast for annual demand (forecast-driven EOQ)
    fc_52  = result.get_forecast(steps=52).predicted_mean.clip(lower=0)
    return mean, ci, result, fc_52


@st.cache_data
def run_rolling_rop(series, lead_time, unit_cost, std_demand, service_level,
                    roll_weeks=52, min_train=52):
    """
    Rolling-window ROP: for each of the last `roll_weeks` weeks,
    refit SARIMA on data up to that point and compute a forecast-driven ROP.
    Returns a DataFrame with Date and ROP columns.
    """
    z_map = {0.90: 1.282, 0.91: 1.341, 0.92: 1.405,
             0.93: 1.476, 0.94: 1.555, 0.95: 1.645,
             0.96: 1.751, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}
    z = z_map.get(round(service_level, 2), 1.645)
    safety_stk = z * std_demand * np.sqrt(lead_time)

    n = len(series)
    start = max(min_train, n - roll_weeks)
    records = []

    for i in range(start, n):
        train = series.iloc[:i]
        try:
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,0,52),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fc_lt = res.get_forecast(steps=lead_time).predicted_mean.clip(lower=0)
            demand_lt = fc_lt.sum() / unit_cost
            rop = demand_lt + safety_stk
        except Exception:
            rop = np.nan
        records.append({"Date": series.index[i], "ROP": rop})

    return pd.DataFrame(records).set_index("Date")


@st.cache_data
def run_forward_rop(fc_52, lead_time, unit_cost, std_demand, service_level, forecast_weeks):
    """
    Forward-looking ROP table: for each forecast week,
    use a rolling lead_time window of the 52-week forecast to compute ROP.
    """
    z_map = {0.90: 1.282, 0.91: 1.341, 0.92: 1.405,
             0.93: 1.476, 0.94: 1.555, 0.95: 1.645,
             0.96: 1.751, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}
    z = z_map.get(round(service_level, 2), 1.645)
    safety_stk = z * std_demand * np.sqrt(lead_time)

    records = []
    for i in range(min(forecast_weeks, len(fc_52))):
        window = fc_52.iloc[i: i + lead_time]
        demand_lt = window.sum() / unit_cost
        rop = demand_lt + safety_stk
        records.append({
            "Forecast Week": i + 1,
            "Week Start": fc_52.index[i].strftime("%Y-%m-%d") if hasattr(fc_52.index[i], 'strftime') else str(fc_52.index[i]),
            "Forecasted Demand (units)": round(fc_52.iloc[i] / unit_cost, 1),
            "ROP (units)": round(rop, 0),
            "Safety Stock (units)": round(safety_stk, 0),
        })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_eoq(annual_demand, ordering_cost, holding_cost_per_unit):
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)


def calc_safety_stock(std_demand, lead_time_weeks, service_level):
    z_map = {0.90: 1.282, 0.91: 1.341, 0.92: 1.405,
             0.93: 1.476, 0.94: 1.555, 0.95: 1.645,
             0.96: 1.751, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}
    z = z_map.get(round(service_level, 2), 1.645)
    return z * std_demand * np.sqrt(lead_time_weeks)


def calc_inventory_costs(eoq, annual_demand, ordering_cost,
                          safety_stock, holding_cost_per_unit):
    if eoq <= 0:
        return 0, 0, 0
    n_orders    = annual_demand / eoq
    order_cost  = n_orders * ordering_cost
    holding     = (eoq / 2 + safety_stock) * holding_cost_per_unit
    total       = order_cost + holding
    return round(order_cost, 2), round(holding, 2), round(total, 2)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
df = load_walmart_data()
# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📦 SCM Dashboard")
    st.markdown("**Walmart Demand Forecasting**")
    st.markdown("---")
    
    selected_store = st.selectbox(
        "Select Store", sorted(df["Store"].unique()),
        help="Walmart operates 45 stores across different regions"
    )
    selected_dept = st.selectbox(
        "Select Department", sorted(df["Dept"].unique()),
        help="Each store has up to 99 departments"
    )
    forecast_weeks = st.slider("Forecast Horizon (weeks)", 4, 26, 12)
    
    st.markdown("---")
    st.markdown("### 🏭 Inventory Parameters")
    unit_cost       = st.number_input("Unit Cost ($)", 1.0, 500.0, 25.0, step=1.0)
    ordering_cost   = st.number_input("Ordering Cost ($)", 10.0, 1000.0, 150.0, step=10.0)
    holding_pct     = st.slider("Holding Cost (% of unit cost / yr)", 5, 40, 20)
    lead_time       = st.slider("Lead Time (weeks)", 1, 12, 2)
    service_level   = st.select_slider(
        "Service Level",
        options=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        value=0.95
    )
    
    st.markdown("---")
    st.markdown("*Built by Pallapolu Bhuvan Chandra*")
    st.markdown("*BITS Pilani | SCM Portfolio*")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">📦 Walmart Demand Forecasting & Inventory Optimization</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">SARIMA forecasting → EOQ + Safety Stock optimization → '
    'Cost sensitivity analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# Filter data
mask   = (df["Store"] == selected_store) & (df["Dept"] == selected_dept)
series = df[mask].set_index("Date")["Weekly_Sales"].sort_index()

if len(series) < 52:
    st.error("Not enough data for this Store/Dept combination.")
    st.stop()

# ── Fit SARIMA once — reused by forecast tab AND inventory calculations ──
with st.spinner("Fitting SARIMA model..."):
    mean_fc, ci_fc, sarima_result, fc_52 = run_sarima(series, forecast_weeks)

# ── Top KPI row ──────────────────────────────────────────────────────────
avg_weekly = series.mean()

# FORECAST-DRIVEN: annual demand from 52-week SARIMA forecast, not historical avg
annual_demand_units = fc_52.sum() / unit_cost

holding_cost_pu = unit_cost * (holding_pct / 100)
eoq         = calc_eoq(annual_demand_units, ordering_cost, holding_cost_pu)
std_demand  = series.std() / unit_cost
safety_stk  = calc_safety_stock(std_demand, lead_time, service_level)

# FORECAST-DRIVEN: ROP uses forecasted demand over lead time window, not historical avg
fc_leadtime = fc_52.iloc[:lead_time].sum() / unit_cost
reorder_pt  = fc_leadtime + safety_stk

oc, hc, tc  = calc_inventory_costs(eoq, annual_demand_units,
                                    ordering_cost, safety_stk, holding_cost_pu)

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (f"${avg_weekly:,.0f}", "Avg Weekly Sales"),
    (f"{eoq:,.0f} units", "Optimal Order Qty (EOQ)"),
    (f"{safety_stk:,.0f} units", f"Safety Stock ({int(service_level*100)}% SL)"),
    (f"{reorder_pt:,.0f} units", "Reorder Point"),
    (f"${tc:,.0f}", "Total Annual Inv. Cost"),
]
for col, (val, label) in zip([col1,col2,col3,col4,col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast", "🔬 Time Series Analysis",
    "📊 Inventory Optimization", "💰 Cost Sensitivity",
    "🔄 Rolling ROP"
])

# ── TAB 1: SARIMA Forecast ───────────────────────────────────────────────
with tab1:
    st.subheader(f"SARIMA Demand Forecast — Store {selected_store}, Dept {selected_dept}")

    future_dates = pd.date_range(series.index[-1], periods=forecast_weeks+1, freq="W-FRI")[1:]
    
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    
    # Historical
    ax.plot(series.index[-78:], series.values[-78:],
            color="#1E40AF", linewidth=1.8, label="Historical Sales", zorder=3)
    # Forecast
    ax.plot(future_dates, mean_fc.values,
            color="#EF4444", linewidth=2, linestyle="--", label="SARIMA Forecast", zorder=4)
    # CI band
    ax.fill_between(future_dates, ci_fc.iloc[:,0], ci_fc.iloc[:,1],
                    alpha=0.15, color="#EF4444", label="95% Confidence Interval")
    # Holiday shading
    holiday_mask = df[mask].set_index("Date")["IsHoliday"]
    holiday_mask.index = pd.to_datetime(holiday_mask.index)
    for d in series.index[-78:][holiday_mask.reindex(series.index[-78:]).fillna(False).astype(bool)]:
        ax.axvline(d, color="#F59E0B", alpha=0.3, linewidth=1)
    
    ax.axvline(series.index[-1], color="#64748B", linestyle=":", linewidth=1.5, label="Forecast Start")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Weekly Sales ($)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Business insight
    fc_trend = ((mean_fc.values[-1] - mean_fc.values[0]) / mean_fc.values[0]) * 100
    direction = "📈 increasing" if fc_trend > 0 else "📉 decreasing"
    st.markdown(f"""
    <div class="insight-box">
    <b>🔍 Business Insight:</b> Demand for Store {selected_store}, 
    Dept {selected_dept} is forecasted to be <b>{direction} by {abs(fc_trend):.1f}%</b> 
    over the next {forecast_weeks} weeks. 
    Average forecasted weekly sales: <b>${mean_fc.mean():,.0f}</b>.
    Yellow lines mark holiday weeks (Super Bowl, Thanksgiving, Christmas) — 
    these consistently drive 15–25% demand spikes requiring proactive inventory build-up.
    </div>""", unsafe_allow_html=True)
    
    # Model diagnostics
    with st.expander("📋 Model Diagnostics (AIC, BIC, Residuals)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("AIC", f"{sarima_result.aic:.1f}")
        c2.metric("BIC", f"{sarima_result.bic:.1f}")
        c3.metric("Log Likelihood", f"{sarima_result.llf:.1f}")

        # ADF test on residuals
        resid = sarima_result.resid.dropna()
        adf_stat, adf_p, *_ = adfuller(resid)
        if adf_p < 0.05:
            st.success(f"✅ Residuals are stationary (ADF p={adf_p:.4f}) — model is well-specified.")
        else:
            st.warning(f"⚠️ Residuals may not be fully stationary (ADF p={adf_p:.4f}).")


# ── TAB 2: Decomposition ────────────────────────────────────────────────
with tab2:
    st.subheader("Time Series Decomposition — Trend, Seasonality, Residual")
    
    decomp = seasonal_decompose(series, model="additive", period=52)
    
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor("#F8FAFC")
    
    components = [
        (series,          "#1E40AF", "Observed Sales ($)"),
        (decomp.trend,    "#10B981", "Trend Component ($)"),
        (decomp.seasonal, "#8B5CF6", "Seasonal Component ($)"),
        (decomp.resid,    "#EF4444", "Residual / Noise ($)"),
    ]
    for ax, (data, color, ylabel) in zip(axes, components):
        ax.set_facecolor("#F8FAFC")
        ax.plot(data, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)
    
    axes[-1].set_xlabel("Week")
    fig.suptitle("Seasonal Decomposition of Walmart Weekly Sales", fontsize=13, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    seasonal_amplitude = (decomp.seasonal.max() - decomp.seasonal.min())
    pct_amplitude = (seasonal_amplitude / series.mean()) * 100

    st.markdown(f"""
    <div class="insight-box">
    <b>🔍 Business Insight:</b> Seasonality accounts for 
    <b>±{pct_amplitude/2:.1f}% swing</b> around the mean demand. 
    This means safety stock calculations <i>must</i> account for seasonal peaks — 
    a flat average-based reorder point would cause stockouts during holiday 
    weeks and excess holding costs in slow periods.
    </div>""", unsafe_allow_html=True)


# ── TAB 3: Inventory Optimization ───────────────────────────────────────
with tab3:
    st.subheader("EOQ + Safety Stock Inventory Policy")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### 📐 Optimal Inventory Policy")
        policy_data = {
            "Parameter"    : ["Annual Demand (units)", "EOQ (units/order)",
                               "Orders per Year", "Safety Stock (units)",
                               "Reorder Point (units)", "Max Inventory (units)"],
            "Value"        : [f"{annual_demand_units:,.0f}", f"{eoq:,.0f}",
                               f"{annual_demand_units/max(eoq,1):,.1f}", f"{safety_stk:,.0f}",
                               f"{reorder_pt:,.0f}", f"{reorder_pt + eoq:,.0f}"]
        }
        st.dataframe(pd.DataFrame(policy_data), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:12px">
        <b>Policy Logic:</b> Place an order of <b>{eoq:,.0f} units</b> every time 
        inventory drops to <b>{reorder_pt:,.0f} units</b>. 
        The <b>{safety_stk:,.0f} unit safety buffer</b> protects against 
        demand spikes during the {lead_time}-week lead time at a 
        <b>{int(service_level*100)}% service level</b>. 
        <i>ROP and EOQ are computed from the SARIMA forecast, not a historical average — 
        so they adjust automatically to upcoming seasonal demand.</i>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 💸 Annual Cost Breakdown")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        
        sizes  = [oc, hc]
        labels = [f"Ordering Cost\n${oc:,.0f}", f"Holding Cost\n${hc:,.0f}"]
        colors = ["#2563EB", "#10B981"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2}
        )
        for t in autotexts:
            t.set_fontsize(11); t.set_fontweight("bold"); t.set_color("white")
        ax.set_title(f"Total Annual Cost: ${tc:,.0f}", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # Service level vs safety stock table
    st.markdown("#### 📊 Service Level Comparison Table")
    sl_levels = [0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    rows_sl = []
    for sl in sl_levels:
        ss   = calc_safety_stock(std_demand, lead_time, sl)
        rp   = fc_leadtime + ss  # forecast-driven
        _, _, tot = calc_inventory_costs(eoq, annual_demand_units,
                                          ordering_cost, ss, holding_cost_pu)
        rows_sl.append({
            "Service Level" : f"{int(sl*100)}%",
            "Safety Stock"  : f"{ss:,.0f} units",
            "Reorder Point" : f"{rp:,.0f} units",
            "Annual Cost"   : f"${tot:,.0f}",
            "vs 95% SL"     : "baseline" if sl == 0.95 else
                               f"+${tot - calc_inventory_costs(eoq, annual_demand_units, ordering_cost, calc_safety_stock(std_demand, lead_time, 0.95), holding_cost_pu)[2]:,.0f}"
        })
    st.dataframe(pd.DataFrame(rows_sl), use_container_width=True, hide_index=True)


# ── TAB 4: Cost Sensitivity ─────────────────────────────────────────────
with tab4:
    st.subheader("Cost Sensitivity Analysis — What drives inventory cost?")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Holding Cost % vs Total Annual Cost")
        hc_range = np.arange(5, 45, 5)
        costs_hc = []
        for h in hc_range:
            hc_pu   = unit_cost * (h/100)
            eoq_h   = calc_eoq(annual_demand_units, ordering_cost, hc_pu)
            ss_h    = calc_safety_stock(std_demand, lead_time, service_level)
            _, _, tc_h = calc_inventory_costs(eoq_h, annual_demand_units,
                                               ordering_cost, ss_h, hc_pu)
            costs_hc.append(tc_h)
        
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor("#F8FAFC"); ax.set_facecolor("#F8FAFC")
        ax.plot(hc_range, costs_hc, color="#2563EB", linewidth=2.5, marker="o", markersize=5)
        ax.axvline(holding_pct, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Current: {holding_pct}%")
        ax.set_xlabel("Holding Cost (% of unit cost)"); ax.set_ylabel("Total Annual Cost ($)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Lead Time (weeks) vs Safety Stock Cost")
        lt_range = np.arange(1, 13)
        ss_costs = []
        for lt in lt_range:
            ss_lt = calc_safety_stock(std_demand, lt, service_level)
            _, hc_lt, _ = calc_inventory_costs(eoq, annual_demand_units,
                                                 ordering_cost, ss_lt, holding_cost_pu)
            ss_costs.append(hc_lt)
        
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor("#F8FAFC"); ax.set_facecolor("#F8FAFC")
        ax.bar(lt_range, ss_costs, color="#10B981", edgecolor="white", linewidth=0.8)
        ax.axvline(lead_time, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Current: {lead_time}wk")
        ax.set_xlabel("Lead Time (weeks)"); ax.set_ylabel("Holding Cost ($)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    # Business insight box
    eoq_95  = eoq
    eoq_99  = calc_eoq(annual_demand_units, ordering_cost,
                        unit_cost * (holding_pct/100))
    ss_95   = calc_safety_stock(std_demand, lead_time, 0.95)
    ss_99   = calc_safety_stock(std_demand, lead_time, 0.99)
    _, _, tc_95 = calc_inventory_costs(eoq_95, annual_demand_units, ordering_cost, ss_95, holding_cost_pu)
    _, _, tc_99 = calc_inventory_costs(eoq_99, annual_demand_units, ordering_cost, ss_99, holding_cost_pu)
    uplift      = ((tc_99 - tc_95) / tc_95) * 100

    st.markdown(f"""
    <div class="insight-box">
    <b>🔍 Key Business Insight:</b> Moving from 95% to 99% service level 
    increases annual inventory cost by <b>{uplift:.1f}%</b> 
    (${tc_95:,.0f} → ${tc_99:,.0f}). 
    Lead time is the highest-leverage variable — reducing it from {lead_time} to 1 week 
    cuts safety stock by <b>{((1 - 1/lead_time**0.5)*100):.0f}%</b>. 
    This explains why Amazon and Flipkart invest billions in last-mile infrastructure — 
    shorter lead times directly reduce working capital requirements.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box" style="margin-top:12px">
    <b>⚡ Stress Test:</b> If a holiday markdown drives demand +25%, 
    the current reorder point of <b>{reorder_pt:,.0f} units</b> would be breached 
    in approximately <b>{max(1, int(safety_stk / (avg_weekly/unit_cost * 0.25)))} week(s)</b>. 
    Pre-positioning inventory 3 weeks before major holidays is recommended.
    </div>""", unsafe_allow_html=True)


# ── TAB 5: Rolling ROP ───────────────────────────────────────────────────
with tab5:
    st.subheader("🔄 Rolling Window Reorder Point")
    st.markdown("""
    <div class="insight-box">
    <b>What this shows:</b> Instead of one fixed ROP, the model refits SARIMA every week 
    on all data up to that point, then forecasts the next <b>lead time</b> weeks to compute 
    a dynamic ROP. This is how a real production inventory system would behave — 
    the trigger point updates as the demand signal evolves.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Running rolling window SARIMA (last 52 weeks) — this takes ~40 seconds..."):
        rolling_rop_df = run_rolling_rop(
            series, lead_time, unit_cost, std_demand, service_level,
            roll_weeks=52, min_train=52
        )

    # ── Chart: historical rolling ROP ───────────────────────────────────
    st.markdown("#### 📉 Historical Rolling ROP (last 52 weeks)")

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    # Actual sales (right-axis scaled to units)
    ax2 = ax.twinx()
    hist = series[rolling_rop_df.index[0]:]
    ax2.fill_between(hist.index, hist.values / unit_cost,
                     alpha=0.12, color="#94A3B8", label="Actual Demand (units)")
    ax2.set_ylabel("Actual Demand (units)", fontsize=9, color="#94A3B8")
    ax2.tick_params(axis="y", labelcolor="#94A3B8")

    # Rolling ROP line
    ax.plot(rolling_rop_df.index, rolling_rop_df["ROP"],
            color="#2563EB", linewidth=2, label="Dynamic ROP", zorder=3)
    ax.axhline(reorder_pt, color="#EF4444", linestyle="--", linewidth=1.5,
               label=f"Static ROP (current): {reorder_pt:,.0f}")

    # Holiday markers
    holiday_dates = series.index[
        df[mask].set_index("Date")["IsHoliday"]
        .reindex(series.index).fillna(False).astype(bool)
    ]
    for d in holiday_dates:
        if d >= rolling_rop_df.index[0]:
            ax.axvline(d, color="#F59E0B", alpha=0.3, linewidth=1)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Week"); ax.set_ylabel("Reorder Point (units)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Dynamic vs Static ROP — Store {selected_store}, Dept {selected_dept}",
                 fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Insight: how much did ROP vary?
    rop_min = rolling_rop_df["ROP"].min()
    rop_max = rolling_rop_df["ROP"].max()
    rop_swing = ((rop_max - rop_min) / rop_min) * 100
    st.markdown(f"""
    <div class="insight-box" style="margin-top:8px">
    <b>🔍 Key Insight:</b> The dynamic ROP swung between 
    <b>{rop_min:,.0f}</b> and <b>{rop_max:,.0f} units</b> — 
    a <b>{rop_swing:.1f}% range</b> over the last 52 weeks. 
    A static ROP of {reorder_pt:,.0f} would have been too low during demand peaks 
    (risking stockouts) and unnecessarily high during slow periods (excess holding cost). 
    Yellow lines mark holiday weeks where ROP spikes are expected.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Table: forward-looking ROP per forecast week ─────────────────────
    st.markdown("#### 📅 Forward-Looking ROP — Next {} Weeks".format(forecast_weeks))
    st.markdown("Each row shows what the ROP *should be* in that forecast week, "
                "based on SARIMA's demand estimate for the following lead time window.")

    fwd_df = run_forward_rop(
        fc_52, lead_time, unit_cost, std_demand, service_level, forecast_weeks
    )
    st.dataframe(fwd_df, use_container_width=True, hide_index=True)

    fwd_max_rop  = fwd_df["ROP (units)"].max()
    fwd_max_week = fwd_df.loc[fwd_df["ROP (units)"].idxmax(), "Forecast Week"]
    st.markdown(f"""
    <div class="warn-box" style="margin-top:8px">
    <b>⚡ Action Signal:</b> Peak ROP over the next {forecast_weeks} weeks is 
    <b>{fwd_max_rop:,.0f} units</b> in forecast week {fwd_max_week}. 
    Ensure inventory is above this level before that week to avoid a stockout.
    </div>""", unsafe_allow_html=True)

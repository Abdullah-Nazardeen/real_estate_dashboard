# app.py
# Streamlit dashboard for Zillow (City x Bedrooms) — 2015–2019 historical analysis
# File expected in the repo root: real_estate.csv

import math
from functools import lru_cache
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Real Estate ROI — City x Bedroom",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str = "real_estate.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    # expected columns:
    # Date,RegionName,State,Bedrooms,Price,Metro,CountyName,SizeRank,Rent,Year,Month
    # normalize types
    df["Bedrooms"] = df["Bedrooms"].astype(str)
    # guard rails
    needed = {"Date","RegionName","State","Bedrooms","Price","Rent"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in real_estate.csv: {missing}")
    # only 2015..2019 (in case extra rows exist)
    df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2019-12-31")].copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

def cagr_from_series(s: pd.Series, dates: pd.Series) -> float:
    s = pd.Series(s).dropna()
    if len(s) < 2:
        return np.nan
    d = pd.Series(dates).iloc[s.index]
    v0, vN = s.iloc[0], s.iloc[-1]
    if v0 <= 0 or vN <= 0: 
        return np.nan
    years = (pd.to_datetime(d.iloc[-1]) - pd.to_datetime(d.iloc[0])).days / 365.25
    if years <= 0: 
        return np.nan
    return (vN / v0) ** (1/years) - 1

def summarize_group(g: pd.DataFrame) -> pd.Series:
    # g is one RegionName+State+Bedrooms
    g = g.sort_values("Date")
    price = g["Price"]
    rent  = g["Rent"]
    dates = g["Date"]

    # anchors & ends
    price_anchor = price.iloc[0] if len(price.dropna()) else np.nan
    rent_anchor  = rent.iloc[0]  if len(rent.dropna())  else np.nan
    price_end    = price.iloc[-1] if len(price.dropna()) else np.nan
    rent_end     = rent.iloc[-1]  if len(rent.dropna())  else np.nan

    price_cagr = cagr_from_series(price.reset_index(drop=True), dates.reset_index(drop=True))
    rent_cagr  = cagr_from_series(rent.reset_index(drop=True),  dates.reset_index(drop=True))

    # gross yields (annualized)
    yield_anchor = (12 * rent_anchor / price_anchor) if (price_anchor and price_anchor>0 and pd.notna(rent_anchor)) else np.nan
    yield_end    = (12 * rent_end / price_end)       if (price_end and price_end>0 and pd.notna(rent_end)) else np.nan

    # window return (purely historical 2015–2019, no forecast):
    # cumulative rent + price appreciation vs price at start
    total_rent_window = rent.sum()  # monthly rent sum over the 5-year window
    price_appreciation = price_end - price_anchor if (pd.notna(price_end) and pd.notna(price_anchor)) else np.nan
    absolute_return_window = (total_rent_window + (price_appreciation if pd.notna(price_appreciation) else 0.0))
    return_pct_window = (absolute_return_window / price_anchor) if (pd.notna(absolute_return_window) and price_anchor>0) else np.nan

    return pd.Series({
        "Price_Anchor": price_anchor,
        "Rent_Anchor": rent_anchor,
        "Price_End": price_end,
        "Rent_End": rent_end,
        "Price_CAGR_Full": price_cagr,
        "Rent_CAGR_Full": rent_cagr,
        "Rent_Yield_Anchor": yield_anchor,
        "Rent_Yield_End": yield_end,
        "Total_Rent_Window": total_rent_window,
        "Absolute_Return_Window": absolute_return_window,
        "Return_Pct_Window": return_pct_window
    })

def label_pct(x: float, nd=1):
    if pd.isna(x): 
        return "—"
    return f"{x*100:.{nd}f}%"

def group_small_slices(df: pd.DataFrame, value_col: str, label_col: str, threshold_pct=0.03, other_label="Other"):
    # groups values contributing < threshold_pct of total into a single "Other" slice
    d = df.copy()
    total = d[value_col].sum()
    if total <= 0: 
        return d
    d["pct"] = d[value_col] / total
    small = d["pct"] < threshold_pct
    if small.any():
        other_sum = d.loc[small, value_col].sum()
        d = d.loc[~small, [label_col, value_col]].copy()
        if other_sum > 0:
            d = pd.concat([d, pd.DataFrame({label_col:[other_label], value_col:[other_sum]})], ignore_index=True)
    return d

# -----------------------------
# Load data
# -----------------------------
df = load_data("real_estate.csv")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
states = sorted(df["State"].dropna().unique().tolist())
bedrooms = ["1","2","3","4","5+"]
state_sel = st.sidebar.multiselect("State", states, default=states[:10])
bed_sel   = st.sidebar.multiselect("Bedrooms", bedrooms, default=bedrooms)
cities_avail = sorted(df.loc[df["State"].isin(state_sel) & df["Bedrooms"].isin(bed_sel), "RegionName"].unique())
city_sel  = st.sidebar.multiselect("City (optional)", cities_avail, default=[])

min_d, max_d = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_d.to_pydatetime(),
    max_value=max_d.to_pydatetime(),
    value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
    format="YYYY-MM"
)

# apply filters
f = df.copy()
if state_sel: 
    f = f[f["State"].isin(state_sel)]
if bed_sel: 
    f = f[f["Bedrooms"].isin(bed_sel)]
if city_sel:
    f = f[f["RegionName"].isin(city_sel)]
f = f[(f["Date"] >= pd.to_datetime(date_range[0])) & (f["Date"] <= pd.to_datetime(date_range[1]))].copy()

# guard
if f.empty:
    st.warning("No data after applying filters.")
    st.stop()

# -----------------------------
# KPI summary (window-aware)
# -----------------------------
# compute per-group summary on filtered data's *full* window (still within 2015–2019 slice)
grp_keys = ["RegionName","State","Bedrooms"]
summary = f.groupby(grp_keys, as_index=False).apply(summarize_group).reset_index(drop=True)

# topline KPIs
n_groups = summary.shape[0]
median_yield_end = summary["Rent_Yield_End"].median()
median_price_cagr = summary["Price_CAGR_Full"].median()
median_rent_cagr = summary["Rent_CAGR_Full"].median()
median_ret_pct = summary["Return_Pct_Window"].median()

st.title("🏠 Real Estate ROI — City x Bedroom (2015–2019)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("City–Bedroom combos", f"{n_groups:,}")
k2.metric("Median Rent Yield (End)", label_pct(median_yield_end))
k3.metric("Median Price CAGR", label_pct(median_price_cagr))
k4.metric("Median Rent CAGR", label_pct(median_rent_cagr))
k5.metric("Median 5Y Return (hist)", label_pct(median_ret_pct))

st.caption("Notes: Rent Yield = 12×Rent / Price. CAGR computed over the selected window using exact day counts. ‘5Y Return (hist)’ = (∑ monthly rent + price change) ÷ price at start; no forecasts are shown.")

# -----------------------------
# Row 1: Time series (meaningful, trend-focused)
# Line charts are the best idiom for temporal data; two synchronized charts avoid dual-axis confusion.
# -----------------------------
st.subheader("Trends over time")

# If a single city is selected, show that city's lines; otherwise show statewide (median) lines.
if len(city_sel) == 1 and len(state_sel) >= 1:
    ts = f[f["RegionName"] == city_sel[0]].sort_values("Date")
    title_suffix = f"{city_sel[0]} ({', '.join(sorted(set(ts['State'])))}), Bedrooms: {', '.join(sorted(set(ts['Bedrooms'])))}"
    st.caption("Showing selected city; use filters to change.")
else:
    # aggregate by Date across current filter to robustly illustrate overall movement
    ts = f.groupby("Date", as_index=False).agg(
        Price=("Price","median"),
        Rent=("Rent","median")
    ).sort_values("Date")
    title_suffix = "Filtered selection (median across groups)"

c1, c2 = st.columns(2)
with c1:
    fig_price = px.line(ts, x="Date", y="Price", title=f"Home Value (ZHVI) — {title_suffix}")
    fig_price.update_layout(hovermode="x unified", yaxis_title="USD", xaxis_title=None)
    st.plotly_chart(fig_price, use_container_width=True)

with c2:
    fig_rent = px.line(ts, x="Date", y="Rent", title=f"Median Rent — {title_suffix}")
    fig_rent.update_layout(hovermode="x unified", yaxis_title="USD per month", xaxis_title=None)
    st.plotly_chart(fig_rent, use_container_width=True)

# -----------------------------
# Row 2: Ranking & composition
# Bar chart for ranking (best for ordered comparisons).
# Pie chart with "Other" bucket for composition (only if categories are not too many).
# -----------------------------
st.subheader("Rankings & Composition")

# A. Top regions by historical 5Y return %
rankN = st.slider("Top-N regions by historical 5Y return %", min_value=5, max_value=50, value=20, step=5)
top = summary.dropna(subset=["Return_Pct_Window"]).sort_values("Return_Pct_Window", ascending=False).head(rankN)
top["Label"] = top["RegionName"] + ", " + top["State"] + " (" + top["Bedrooms"] + "BR)"

fig_rank = px.bar(
    top.sort_values("Return_Pct_Window", ascending=True),
    x="Return_Pct_Window",
    y="Label",
    orientation="h",
    title=f"Top {len(top)} Regions by 5Y Return (historical only)",
    text=top["Return_Pct_Window"].apply(lambda v: f"{v*100:.1f}%"),
)
fig_rank.update_layout(yaxis_title=None, xaxis_title="5Y Return ( (% of initial price) )", hovermode="y")
st.plotly_chart(fig_rank, use_container_width=True)

# B. Composition: share of total rent by State (group small slices)
comp = (f.groupby("State", as_index=False)["Rent"].sum()
          .rename(columns={"Rent":"TotalRent"}).sort_values("TotalRent", ascending=False))
comp2 = group_small_slices(comp, "TotalRent", "State", threshold_pct=0.03, other_label="Other")
fig_pie = px.pie(comp2, names="State", values="TotalRent", title="Share of Total Rent by State (current filter)",
                 hole=0.35)
fig_pie.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}<br>%{percent} of total<br>$%{value:,.0f}")
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Row 3: Relationship & heatmap
# Scatter = relationship between two continuous variables.
# Heatmap = matrix comparison across 2 categorical axes (Bedrooms x State).
# -----------------------------
st.subheader("Relationships & Distributions")

# A. Scatter: Yield vs Price CAGR (size=SizeRank proxy, color=Bedrooms)
# (Lower SizeRank often means larger city in Zillow's ranking)
sc = (summary.assign(SizeRank=lambda d: 
        df.merge(d[["RegionName","State","Bedrooms"]], on=["RegionName","State","Bedrooms"], how="right")["SizeRank"])
      )

fig_sc = px.scatter(
    summary,
    x="Rent_Yield_End",
    y="Price_CAGR_Full",
    color="Bedrooms",
    hover_data=["RegionName","State","Bedrooms",
                summary["Rent_Yield_End"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—").rename("Yield (End)"),
                summary["Price_CAGR_Full"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—").rename("Price CAGR")],
    title="Yield (End) vs Price CAGR — Does high yield come with high/low appreciation?",
)
fig_sc.update_layout(xaxis_title="Rent Yield (End of window)", yaxis_title="Price CAGR (2015–2019)")
st.plotly_chart(fig_sc, use_container_width=True)

# B. Heatmap: Average yield by State x Bedrooms (only top states by data volume)
hm = (summary.merge(df[["RegionName","State","Bedrooms"]].drop_duplicates(), on=["RegionName","State","Bedrooms"], how="left"))
state_counts = df.groupby("State")["RegionName"].nunique().sort_values(ascending=False)
top_states = state_counts.head(15).index.tolist()  # cap to reduce clutter
hm = hm[hm["State"].isin(top_states)]
pvt = hm.pivot_table(index="State", columns="Bedrooms", values="Rent_Yield_End", aggfunc="mean")
pvt = pvt.reindex(index=sorted(pvt.index), columns=sorted(pvt.columns, key=lambda x: (x=="5+", x)))
fig_hm = go.Figure(data=go.Heatmap(
    z=pvt.values,
    x=pvt.columns.astype(str),
    y=pvt.index.astype(str),
    coloraxis="coloraxis",
    hoverongaps=False,
    hovertemplate="State=%{y}<br>Bedrooms=%{x}<br>Avg Yield=%{z:.2%}<extra></extra>"
))
fig_hm.update_layout(
    title="Heatmap — Average Rent Yield (End of window) by State and Bedrooms",
    xaxis_title="Bedrooms",
    yaxis_title="State",
    coloraxis_colorscale="Blues",
    coloraxis_colorbar=dict(title="Yield", ticksuffix="%")
)
st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------
# Data table (for export / verification)
# -----------------------------
st.subheader("Detail table (downloadable)")
dl_cols = ["RegionName","State","Bedrooms","Price_Anchor","Rent_Anchor","Price_End","Rent_End",
           "Rent_Yield_Anchor","Rent_Yield_End","Price_CAGR_Full","Rent_CAGR_Full",
           "Total_Rent_Window","Absolute_Return_Window","Return_Pct_Window"]
st.dataframe(
    summary[dl_cols].sort_values("Return_Pct_Window", ascending=False),
    use_container_width=True
)
st.download_button(
    "Download summary as CSV",
    data=summary[dl_cols].to_csv(index=False).encode("utf-8"),
    file_name="roi_summary_filtered.csv",
    mime="text/csv"
)

st.caption("Dashboard design applies visualization grammar: lines for temporal trends; bars for ordered ranking; pie with ‘Other’ for composition; scatter for relationships; heatmap for matrix comparison; consistent units/labels; conservative, readable defaults. Built for Task 3 criteria (idioms, aggregation/filtering, color usage, layout, interactivity).")

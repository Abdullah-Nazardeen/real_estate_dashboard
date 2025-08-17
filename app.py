# app.py
# Streamlit dashboard for Zillow (City x Bedrooms) — 2015–2019 historical analysis + optional forecast to 2025
# Expects real_estate.csv in repo root with columns:
# Date,RegionName,State,Bedrooms,Price,Metro,CountyName,SizeRank,Rent,Year,Month

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Real Estate ROI — City x Bedroom", page_icon="🏠", layout="wide")

PCT_FMT = ".1%"  # consistent percent formatting everywhere
CURRENCY_FMT = ",.0f"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str = "real_estate.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    # normalize types
    df["Bedrooms"] = df["Bedrooms"].astype(str)
    need = {"Date","RegionName","State","Bedrooms","Price","Rent"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in real_estate.csv: {miss}")
    # enforce historical window
    df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2019-12-31")].copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

def cagr_from_series(vals: pd.Series, dates: pd.Series) -> float:
    s = vals.dropna()
    if len(s) < 2: return np.nan
    v0, vN = s.iloc[0], s.iloc[-1]
    if v0 <= 0 or vN <= 0: return np.nan
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0: return np.nan
    return (vN / v0) ** (1/years) - 1

def summarize_group(g: pd.DataFrame) -> pd.Series:
    # g is one RegionName+State+Bedrooms across 2015–2019
    g = g.sort_values("Date")
    price, rent, dates = g["Price"], g["Rent"], g["Date"]

    price_anchor = price.iloc[0] if len(price.dropna()) else np.nan
    rent_anchor  = rent.iloc[0]  if len(rent.dropna())  else np.nan
    price_end    = price.iloc[-1] if len(price.dropna()) else np.nan
    rent_end     = rent.iloc[-1]  if len(rent.dropna())  else np.nan

    price_cagr = cagr_from_series(price.reset_index(drop=True), dates.reset_index(drop=True))
    rent_cagr  = cagr_from_series(rent.reset_index(drop=True),  dates.reset_index(drop=True))

    # annualized gross yields
    yield_anchor = (12 * rent_anchor / price_anchor) if (price_anchor and price_anchor>0 and pd.notna(rent_anchor)) else np.nan
    yield_end    = (12 * rent_end / price_end)       if (price_end and price_end>0 and pd.notna(rent_end)) else np.nan

    # historical window metrics (5 years)
    total_rent_window = rent.sum()  # sum of monthly rents
    price_appreciation_abs = (price_end - price_anchor) if (pd.notna(price_end) and pd.notna(price_anchor)) else np.nan
    price_appreciation_pct = (price_end / price_anchor - 1.0) if (pd.notna(price_end) and pd.notna(price_anchor) and price_anchor>0) else np.nan

    # 5Y Return (profit relative to initial price)
    # profit = total_rent_window + price_appreciation_abs
    absolute_return_window = total_rent_window + (price_appreciation_abs if pd.notna(price_appreciation_abs) else 0.0)
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
        "Price_Appreciation_Abs": price_appreciation_abs,
        "Price_Appreciation_Pct": price_appreciation_pct,
        "Absolute_Return_Window": absolute_return_window,
        "Return_Pct_Window": return_pct_window
    })

def group_small_slices(df: pd.DataFrame, value_col: str, label_col: str, threshold_pct=0.03, other_label="Other"):
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

def pct_text(x: float, nd=1):
    return "—" if pd.isna(x) else f"{x*100:.{nd}f}%"

def money_text(x: float):
    return "—" if pd.isna(x) else f"${x:,.0f}"

# -----------------------------
# Load data
# -----------------------------
df = load_data("real_estate.csv")

# -----------------------------
# Sidebar (global filters)
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

# Apply filters
f = df.copy()
if state_sel: f = f[f["State"].isin(state_sel)]
if bed_sel:   f = f[f["Bedrooms"].isin(bed_sel)]
if city_sel:  f = f[f["RegionName"].isin(city_sel)]
f = f[(f["Date"] >= pd.to_datetime(date_range[0])) & (f["Date"] <= pd.to_datetime(date_range[1]))].copy()

if f.empty:
    st.warning("No data after applying filters.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab_hist, tab_fcst = st.tabs(["📈 Historical (2015–2019)", "🔮 Forecast to 2025"])

# =============================
# HISTORICAL TAB
# =============================
with tab_hist:
    st.title("📈 Historical analysis (2015–2019)")

    # Group-level summary for selected slice
    grp_keys = ["RegionName","State","Bedrooms"]
    summary = f.groupby(grp_keys, as_index=False).apply(summarize_group).reset_index(drop=True)

    # Topline KPIs (use medians across selected groups)
    n_groups = summary.shape[0]
    median_yield_end = summary["Rent_Yield_End"].median()
    median_price_cagr = summary["Price_CAGR_Full"].median()
    median_rent_cagr = summary["Rent_CAGR_Full"].median()
    median_ret_pct = summary["Return_Pct_Window"].median()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("City–Bedroom combos", f"{n_groups:,}")
    k2.metric("Median Rent Yield (End)", pct_text(median_yield_end))
    k3.metric("Median Price CAGR", pct_text(median_price_cagr))
    k4.metric("Median Rent CAGR", pct_text(median_rent_cagr))
    k5.metric("Median 5Y Return (hist)", pct_text(median_ret_pct))
    st.caption("5Y Return (hist) = (Σ monthly rent + price change 2015→2019) / initial price (2015-01). A value of 129.7% means ~1.297× your purchase price in *profit*, i.e., final value ≈ 2.297× initial.")

    # ---- Time series: use a single city if chosen, else median across slice
    st.subheader("Trends over time")
    if len(city_sel) == 1:
        ts = f[f["RegionName"] == city_sel[0]].sort_values("Date")
        title_suffix = f"{city_sel[0]} ({', '.join(sorted(set(ts['State'])))}), Bedrooms: {', '.join(sorted(set(ts['Bedrooms'])))}"
        st.caption("Showing selected city; change filters to compare.")
    else:
        ts = f.groupby("Date", as_index=False).agg(Price=("Price","median"), Rent=("Rent","median")).sort_values("Date")
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

    # ---- Rankings: price appreciation, rent, combined return
    st.subheader("Rankings")
    rankN = st.slider("Top-N to display", min_value=5, max_value=50, value=20, step=5)

    # A) Top by price appreciation (%)
    top_app = summary.dropna(subset=["Price_Appreciation_Pct"]).sort_values("Price_Appreciation_Pct", ascending=False).head(rankN)
    top_app["Label"] = top_app["RegionName"] + ", " + top_app["State"] + " (" + top_app["Bedrooms"] + "BR)"
    fig_app = px.bar(
        top_app.sort_values("Price_Appreciation_Pct", ascending=True),
        x="Price_Appreciation_Pct", y="Label", orientation="h",
        title=f"Top {len(top_app)} by Price Appreciation (2015→2019)",
        text=top_app["Price_Appreciation_Pct"].apply(lambda v: f"{v*100:.1f}%"),
    )
    fig_app.update_layout(yaxis_title=None, xaxis_title="Price Appreciation (2015→2019)", hovermode="y")
    fig_app.update_xaxes(tickformat=PCT_FMT)
    st.plotly_chart(fig_app, use_container_width=True)

    # B) Top by total rent collected (absolute $ over the window)
    top_rent = summary.dropna(subset=["Total_Rent_Window"]).sort_values("Total_Rent_Window", ascending=False).head(rankN)
    top_rent["Label"] = top_rent["RegionName"] + ", " + top_rent["State"] + " (" + top_rent["Bedrooms"] + "BR)"
    fig_rent_top = px.bar(
        top_rent.sort_values("Total_Rent_Window", ascending=True),
        x="Total_Rent_Window", y="Label", orientation="h",
        title=f"Top {len(top_rent)} by Total Rent Collected (2015–2019)",
        text=top_rent["Total_Rent_Window"].apply(lambda v: f"${v:,.0f}"),
    )
    fig_rent_top.update_layout(yaxis_title=None, xaxis_title="Total Rent (USD, 5-year sum)", hovermode="y")
    st.plotly_chart(fig_rent_top, use_container_width=True)

    # C) Top by combined 5Y return (% of initial price)
    top_ret = summary.dropna(subset=["Return_Pct_Window"]).sort_values("Return_Pct_Window", ascending=False).head(rankN)
    top_ret["Label"] = top_ret["RegionName"] + ", " + top_ret["State"] + " (" + top_ret["Bedrooms"] + "BR)"
    fig_ret = px.bar(
        top_ret.sort_values("Return_Pct_Window", ascending=True),
        x="Return_Pct_Window", y="Label", orientation="h",
        title=f"Top {len(top_ret)} by 5Y Return (historical only)",
        text=top_ret["Return_Pct_Window"].apply(lambda v: f"{v*100:.1f}%"),
    )
    fig_ret.update_layout(yaxis_title=None, xaxis_title="5Y Return (profit ÷ initial price)", hovermode="y")
    fig_ret.update_xaxes(tickformat=PCT_FMT)
    st.plotly_chart(fig_ret, use_container_width=True)

    # ---- Composition pie (share of rent by state) with "Other"
    st.subheader("Composition: Rent by State")
    comp = (f.groupby("State", as_index=False)["Rent"].sum().rename(columns={"Rent":"TotalRent"})
              .sort_values("TotalRent", ascending=False))
    comp2 = group_small_slices(comp, "TotalRent", "State", threshold_pct=0.03, other_label="Other")
    fig_pie = px.pie(comp2, names="State", values="TotalRent", title="Share of Total Rent by State (current filter)", hole=0.35)
    fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                          hovertemplate="%{label}<br>%{percent} of total<br>$%{value:,.0f}")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ---- Relationships: Yield vs Price CAGR (both percent axes)
    st.subheader("Relationship: Yield vs Price Growth")
    fig_sc = px.scatter(
        summary, x="Rent_Yield_End", y="Price_CAGR_Full", color="Bedrooms",
        hover_data={
            "RegionName":True, "State":True, "Bedrooms":True,
            "Rent_Yield_End":":.1%",
            "Price_CAGR_Full":":.1%"
        },
        title="Does higher yield come with higher/lower price growth?"
    )
    fig_sc.update_xaxes(title="Rent Yield (End of window)", tickformat=PCT_FMT)
    fig_sc.update_yaxes(title="Price CAGR (2015–2019)", tickformat=PCT_FMT)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ---- Table for export
    st.subheader("Detail table (downloadable)")
    dl_cols = [
        "RegionName","State","Bedrooms","Price_Anchor","Rent_Anchor","Price_End","Rent_End",
        "Rent_Yield_Anchor","Rent_Yield_End","Price_CAGR_Full","Rent_CAGR_Full",
        "Price_Appreciation_Abs","Price_Appreciation_Pct","Total_Rent_Window",
        "Absolute_Return_Window","Return_Pct_Window"
    ]
    st.dataframe(summary[dl_cols].sort_values("Return_Pct_Window", ascending=False), use_container_width=True)
    st.download_button(
        "Download summary CSV",
        data=summary[dl_cols].to_csv(index=False).encode("utf-8"),
        file_name="roi_summary_filtered.csv",
        mime="text/csv"
    )

# =============================
# FORECAST TAB
# =============================
with tab_fcst:
    st.title("🔮 Forecast: Buy at 2019-12, project to 2025-12 (CAGR-based)")

    # Build a purchase-at-end-2019 baseline per group from the historical summary
    grp_keys = ["RegionName","State","Bedrooms"]
    base = f.groupby(grp_keys, as_index=False).apply(summarize_group).reset_index(drop=True)

    # Forecast assumptions
    YEARS = 6  # 2020-01 .. 2025-12
    MONTHS = YEARS * 12

    # Forecast functions
    def monthly_path(start_value: float, annual_cagr: float, months: int) -> np.ndarray:
        if pd.isna(start_value) or pd.isna(annual_cagr) or start_value <= 0: 
            return np.array([np.nan]*months)
        m = (1 + annual_cagr) ** (1/12) - 1
        return np.array([start_value * ((1+m) ** (i+1)) for i in range(months)])

    # Allow user to select a single city to illustrate the path; otherwise, show top tables
    st.caption("This tab uses each group’s 2015–2019 CAGR to project a monthly path from 2020-01 to 2025-12. It’s illustrative (constant growth), not a statistical model.")
    sel_city = st.selectbox("Pick a city to show forecast paths (optional)", ["(None)"] + sorted(base["RegionName"].unique().tolist()))
    if sel_city != "(None)":
        base_city = base[base["RegionName"] == sel_city]
        # If multiple states/bedrooms exist for same city, let user pick
        st_cols = sorted(base_city["State"].unique().tolist())
        bd_cols = sorted(base_city["Bedrooms"].unique().tolist(), key=lambda x: (x=="5+", x))
        sel_state = st.selectbox("State", st_cols)
        sel_bd    = st.selectbox("Bedrooms", bd_cols)
        row = base_city[(base_city["State"]==sel_state) & (base_city["Bedrooms"]==sel_bd)].head(1)
        if row.empty:
            st.warning("No data for that combination.")
        else:
            price0 = float(row["Price_End"])
            rent0  = float(row["Rent_End"])
            pcagr  = float(row["Price_CAGR_Full"])
            rcagr  = float(row["Rent_CAGR_Full"])

            price_path = monthly_path(price0, pcagr, MONTHS)
            rent_path  = monthly_path(rent0,  rcagr, MONTHS)

            months = pd.period_range("2020-01", periods=MONTHS, freq="M").to_timestamp()
            dfp = pd.DataFrame({"Date": months, "Price_Fcst": price_path, "Rent_Fcst": rent_path})

            # KPIs at 2025-12
            price_2025 = np.nan if len(dfp["Price_Fcst"].dropna())==0 else float(dfp["Price_Fcst"].iloc[-1])
            total_rent_2020_2025 = float(np.nansum(dfp["Rent_Fcst"]))
            final_total_2025 = price_2025 + total_rent_2020_2025 if (pd.notna(price_2025)) else np.nan
            total_return_pct = (final_total_2025 - price0) / price0 if (pd.notna(final_total_2025) and price0>0) else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Purchase price (2019-12)", money_text(price0))
            k2.metric("Projected price (2025-12)", money_text(price_2025))
            k3.metric("Projected rent sum (2020–2025)", money_text(total_rent_2020_2025))
            k4.metric("Projected total return", pct_text(total_return_pct))

            c1, c2 = st.columns(2)
            with c1:
                fig_p = px.line(dfp, x="Date", y="Price_Fcst", title=f"Projected Home Value — {sel_city}, {sel_state} ({sel_bd}BR)")
                fig_p.update_layout(hovermode="x unified", yaxis_title="USD", xaxis_title=None)
                st.plotly_chart(fig_p, use_container_width=True)
            with c2:
                fig_r = px.line(dfp, x="Date", y="Rent_Fcst", title=f"Projected Monthly Rent — {sel_city}, {sel_state} ({sel_bd}BR)")
                fig_r.update_layout(hovermode="x unified", yaxis_title="USD per month", xaxis_title=None)
                st.plotly_chart(fig_r, use_container_width=True)

            # Decomposition bar: appreciation vs rent (absolute $)
            st.subheader("Projected value decomposition (absolute $)")
            comp_df = pd.DataFrame({
                "Component":["Price appreciation","Total rent (2020–2025)"],
                "USD":[(price_2025 - price0) if pd.notna(price_2025) else np.nan, total_rent_2020_2025]
            })
            fig_dec = px.bar(comp_df, x="USD", y="Component", orientation="h", text=comp_df["USD"].apply(lambda v: f"${v:,.0f}"),
                             title="Where does the projected value come from?")
            fig_dec.update_layout(yaxis_title=None, xaxis_title="USD")
            st.plotly_chart(fig_dec, use_container_width=True)
    else:
        # No city selected → ranking tables based on projected totals using each group’s CAGR
        # Compute projections (end value only & total rent)
        def project_row(r):
            price0 = r["Price_End"]; rent0 = r["Rent_End"]; pcagr = r["Price_CAGR_Full"]; rcagr = r["Rent_CAGR_Full"]
            if any(pd.isna([price0, rent0, pcagr, rcagr])) or price0<=0 or rent0<=0:
                return pd.Series({"Price_2025":np.nan,"RentSum_2020_2025":np.nan,"TotalReturnPct_2025":np.nan})
            m_pc = (1+pcagr)**(1/12)-1
            m_rc = (1+rcagr)**(1/12)-1
            price_2025 = price0 * ((1+m_pc)**MONTHS)
            # geometric series sum of monthly rent
            if abs(m_rc) < 1e-12:
                rent_sum = rent0 * MONTHS
            else:
                rent_sum = rent0 * ((1+m_rc)**(MONTHS+1) - (1+m_rc)) / m_rc  # sum from month1..monthN of rent0*(1+m)^k
            total_return_pct = ((price_2025 + rent_sum) - price0) / price0
            return pd.Series({"Price_2025":price_2025,"RentSum_2020_2025":rent_sum,"TotalReturnPct_2025":total_return_pct})

        proj = base.copy()
        proj = pd.concat([proj, proj.apply(project_row, axis=1)], axis=1)
        proj["Label"] = proj["RegionName"] + ", " + proj["State"] + " (" + proj["Bedrooms"] + "BR)"

        st.subheader("Projected rankings (using 2015–2019 CAGRs)")
        rankN2 = st.slider("Top-N to display (projection)", min_value=5, max_value=50, value=20, step=5, key="rank_proj")

        # Top by projected total return %
        top_proj = proj.dropna(subset=["TotalReturnPct_2025"]).sort_values("TotalReturnPct_2025", ascending=False).head(rankN2)
        fig_proj = px.bar(
            top_proj.sort_values("TotalReturnPct_2025", ascending=True),
            x="TotalReturnPct_2025", y="Label", orientation="h",
            title=f"Top {len(top_proj)} by Projected Return to 2025 (buy at 2019-12)",
            text=top_proj["TotalReturnPct_2025"].apply(lambda v: f"{v*100:.1f}%"),
        )
        fig_proj.update_layout(yaxis_title=None, xaxis_title="Projected Total Return (profit ÷ price at purchase)", hovermode="y")
        fig_proj.update_xaxes(tickformat=PCT_FMT)
        st.plotly_chart(fig_proj, use_container_width=True)

        c1, c2 = st.columns(2)
        # Top by projected price appreciation %
        top_app_fc = proj.dropna(subset=["Price_2025","Price_End"]).copy()
        top_app_fc["AppreciationPct_2025"] = (top_app_fc["Price_2025"] / top_app_fc["Price_End"] - 1)
        top_app_fc = top_app_fc.sort_values("AppreciationPct_2025", ascending=False).head(rankN2)
        fig_app_fc = px.bar(
            top_app_fc.sort_values("AppreciationPct_2025", ascending=True),
            x="AppreciationPct_2025", y="Label", orientation="h",
            title=f"Top {len(top_app_fc)} by Projected Price Appreciation to 2025",
            text=top_app_fc["AppreciationPct_2025"].apply(lambda v: f"{v*100:.1f}%"),
        )
        fig_app_fc.update_layout(yaxis_title=None, xaxis_title="Projected Price Appreciation", hovermode="y")
        fig_app_fc.update_xaxes(tickformat=PCT_FMT)
        with c1:
            st.plotly_chart(fig_app_fc, use_container_width=True)

        # Top by projected total rent collected (absolute $)
        top_rent_fc = proj.dropna(subset=["RentSum_2020_2025"]).sort_values("RentSum_2020_2025", ascending=False).head(rankN2)
        fig_rent_fc = px.bar(
            top_rent_fc.sort_values("RentSum_2020_2025", ascending=True),
            x="RentSum_2020_2025", y="Label", orientation="h",
            title=f"Top {len(top_rent_fc)} by Projected Total Rent (2020–2025)",
            text=top_rent_fc["RentSum_2020_2025"].apply(lambda v: f"${v:,.0f}"),
        )
        fig_rent_fc.update_layout(yaxis_title=None, xaxis_title="Projected Total Rent (USD)", hovermode="y")
        with c2:
            st.plotly_chart(fig_rent_fc, use_container_width=True)

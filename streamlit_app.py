# streamlit_app.py
import os
import math
import datetime
from datetime import datetime as dt

import streamlit as st
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh

# -----------------------
# Auto-refresh (every 60s)
# -----------------------
# This causes Streamlit to rerun the script every minute.
_refresh_count = st_autorefresh(interval=60 * 1000, key="fuel_dash_autorefresh")

# -----------------------
# Page config & small theme
# -----------------------
st.set_page_config(
    page_title="Fuel Ops Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS tweaks for look & feel
st.markdown(
    """
<style>
/* Card like boxes */
.last-updated-box {
    background-color: #1E88E5;
    color: #fff;
    padding: 10px 16px;
    border-radius: 10px;
    font-weight: 600;
    display: inline-block;
}

/* KPI small caps label */
.kpi-label {
    font-size: 14px;
    color: #666;
}

/* reduce padding around Streamlit components to fit more */
.block-container .css-1lcbmhc.e1fqkh3o3 {
    padding-top: 1rem;
}

/* make table font slightly smaller */
[data-testid="stDataFrameWrapper"] table {
    font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# Utility: last-updated getter
# -----------------------
def get_last_updated_text(preferred_file="last_updated.txt", fallback_file="company_fuel_report.csv"):
    # 1) preferred_file (written by ETL)
    if os.path.exists(preferred_file):
        try:
            with open(preferred_file, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt + " (ETL)"
        except Exception:
            pass
    # 2) fallback: file mtime of a main CSV
    if os.path.exists(fallback_file):
        mtime = os.path.getmtime(fallback_file)
        return dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S") + " (CSV mtime)"
    return "Unknown"

# -----------------------
# CACHED DATA LOADER (reads CSVs) - refreshes every 60s
# -----------------------
@st.cache_data(ttl=60)
def load_data_from_csv(base_path="."):
    # Load CSVs used by the dashboard. If missing, return empty DataFrames
    def safe_read(path, **kwargs):
        full = os.path.join(base_path, path)
        if os.path.exists(full):
            try:
                return pd.read_csv(full, **kwargs)
            except Exception as e:
                st.error(f"Failed to read {path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    coupon = safe_read("coupon_sales.csv", parse_dates=["sale_date"])
    card = safe_read("card_sales.csv", parse_dates=["sale_date"])
    cash = safe_read("cash_sales.csv", parse_dates=["sale_date"])
    swipe = safe_read("swipe_sales.csv", parse_dates=["sale_date"])
    stock = safe_read("site_stock.csv", parse_dates=["date"])
    price = safe_read("price_history.csv", parse_dates=["date"])
    discounts = safe_read("discounted_transactions.csv", parse_dates=["created_at"])
    exp_coupons = safe_read("expired_coupons_report.csv", parse_dates=["activation_date"])
    company_fuel = safe_read("company_fuel_report.csv", parse_dates=["date"])
    companies_litres = safe_read("companies_daily_litres_sales.csv", parse_dates=["MONTH"])

    # Normalise some columns safely (avoid crashes if missing)
    for df in [coupon, card, cash, swipe, company_fuel, companies_litres]:
        if "site_id" in df.columns:
            df["site_id"] = df["site_id"].astype(str)
        if "service_station_id" in df.columns:
            df["service_station_id"] = df["service_station_id"].astype(str)
        if "company_name" in df.columns:
            df["company_name"] = df["company_name"].astype(str)

    return {
        "coupon": coupon,
        "card": card,
        "cash": cash,
        "swipe": swipe,
        "stock": stock,
        "price": price,
        "discounts": discounts,
        "exp_coupons": exp_coupons,
        "company_fuel": company_fuel,
        "companies_litres": companies_litres,
    }

# Load
DATA = load_data_from_csv()

# -----------------------
# Sidebar: role, date filters, navigation
# -----------------------
with st.sidebar:
    st.header("Controls")
    # Role toggle (simple)
    role = st.selectbox("Role", options=["Viewer", "Admin"], index=0)
    st.write("")  # spacer
    # Date range filter (default: yesterday - today)
    today = pd.Timestamp.today().normalize()
    default_start = today - pd.Timedelta(days=1)
    start_date, end_date = st.date_input(
        "Date range (sale/transacted date)",
        value=(default_start, today),
        min_value=pd.Timestamp("2000-01-01"),
        max_value=today,
    )
    # Page navigation
    st.markdown("---")
    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Sales (All channels)",
            "Company Fuel",
            "Discounts",
            "Expired Coupons",
            "Stock & Prices",
            "Lubricants",
        ],
    )
    st.markdown("---")
    st.caption("Auto-refresh every 60s • ETL updates CSVs in background")

# -----------------------
# Helpers: filter by date safely
# -----------------------
def safe_filter(df, date_col, start, end):
    if df is None or df.empty:
        return df
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]

# -----------------------
# Small UI: top row with last-updated + refresh counter + role badge
# -----------------------
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("⛽ Fuel Operations Dashboard")
with col2:
    last_txt = get_last_updated_text()
    st.markdown(f'<div class="last-updated-box">🔄 Last Updated<br><small>{last_txt}</small></div>', unsafe_allow_html=True)
with col3:
    st.metric(label="Auto refresh ticks", value=_refresh_count)
    st.markdown(f"**Role:** {role}")

# -----------------------
# PAGE: Dashboard overview
# -----------------------
def page_dashboard():
    st.subheader("Overview / KPIs")
    # gather filtered data
    coupon = safe_filter(DATA["coupon"], "sale_date", start_date, end_date)
    card = safe_filter(DATA["card"], "sale_date", start_date, end_date)
    cash = safe_filter(DATA["cash"], "sale_date", start_date, end_date)
    swipe = safe_filter(DATA["swipe"], "sale_date", start_date, end_date)

    # metrics calculations (safe)
    def sum_safe(df, col):
        if df is None or df.empty or col not in df.columns:
            return 0
        return df[col].sum()

    card_litres = sum_safe(card, "total_litres")
    coupon_litres = sum_safe(coupon, "total_litres")
    cash_litres = sum_safe(cash, "total_litres")
    swipe_litres = sum_safe(swipe, "total_litres")

    total_revenue = (
        sum_safe(card, "total_amount")
        + sum_safe(coupon, "total_amount")
        + sum_safe(cash, "total_amount")
        + sum_safe(swipe, "total_amount")
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.4])
    k1.metric("💳 Card Litres", f"{card_litres:,.0f}")
    k2.metric("🎟️ Coupon Litres", f"{coupon_litres:,.0f}")
    k3.metric("💵 Cash Litres", f"{cash_litres:,.0f}")
    k4.metric("💻 Swipe Litres", f"{swipe_litres:,.0f}")
    k5.metric("🧾 Total Revenue (local)", f"${total_revenue:,.0f}")

    st.markdown("### Sales trend (combined channels)")
    combined = pd.concat(
        [
            coupon.assign(channel="Coupon") if not coupon.empty else coupon,
            card.assign(channel="Card") if not card.empty else card,
            cash.assign(channel="Cash") if not cash.empty else cash,
            swipe.assign(channel="Swipe") if not swipe.empty else swipe,
        ],
        ignore_index=True,
        sort=False,
    )

    if combined.empty:
        st.info("No sales data for selected range.")
        return

    # prepare for chart
    if "sale_date" in combined.columns:
        grouped = combined.groupby(["sale_date", "channel"], dropna=False)["total_litres"].sum().reset_index()
    else:
        grouped = pd.DataFrame()

    if not grouped.empty:
        # animated chart: use 'frame' simulation via windowed data (Altair does not do streaming natively here)
        line = (
            alt.Chart(grouped)
            .mark_line(point=True)
            .encode(
                x=alt.X("sale_date:T", title="Date"),
                y=alt.Y("total_litres:Q", title="Litres"),
                color="channel:N",
                tooltip=["sale_date:T", "channel:N", "total_litres:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(line.interactive(), use_container_width=True)

    st.markdown("### Top sites by litres (selected range)")
    agg_site = combined.groupby(["service_station_name", "channel"], dropna=False)["total_litres"].sum().reset_index()
    if not agg_site.empty:
        top_sites = agg_site.groupby("service_station_name")["total_litres"].sum().reset_index().sort_values("total_litres", ascending=False).head(10)
        st.bar_chart(top_sites.set_index("service_station_name")["total_litres"])
    else:
        st.write("No site data available.")

# -----------------------
# PAGE: Sales - All Channels (detailed)
# -----------------------
def page_sales():
    st.header("Sales — All Channels")
    coupon = safe_filter(DATA["coupon"], "sale_date", start_date, end_date)
    card = safe_filter(DATA["card"], "sale_date", start_date, end_date)
    cash = safe_filter(DATA["cash"], "sale_date", start_date, end_date)
    swipe = safe_filter(DATA["swipe"], "sale_date", start_date, end_date)

    tabs = st.tabs(["Combined", "By Channel", "Downloads"])
    with tabs[0]:
        dfs = [df for df in [coupon, card, cash, swipe] if (df is not None and not df.empty)]
        if dfs:
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            st.dataframe(combined)
        else:
            st.info("No sales data for the selected range.")
    with tabs[1]:
        st.subheader("Card")
        st.dataframe(card)
        st.subheader("Coupon")
        st.dataframe(coupon)
        st.subheader("Cash")
        st.dataframe(cash)
        st.subheader("Swipe")
        st.dataframe(swipe)
    with tabs[2]:
        # Downloads
        for name, df in [("coupon_sales.csv", coupon), ("card_sales.csv", card), ("cash_sales.csv", cash), ("swipe_sales.csv", swipe)]:
            if df is None or df.empty:
                st.write(f"{name}: (no data)")
            else:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {name}", data=csv, file_name=name, mime="text/csv")

# -----------------------
# PAGE: Company Fuel
# -----------------------
def page_company_fuel():
    st.header("Company Fuel Report")
    df = safe_filter(DATA["company_fuel"], "date", start_date, end_date)
    if df is None or df.empty:
        st.info("No company-fuel data for the selected range.")
        return

    st.metric("Rows", f"{len(df):,}")
    st.dataframe(df)

    # Company trends (top 10)
    agg = df.groupby("company_name").agg({"diesel_usd_amount": "sum", "petrol_usd_amount": "sum"}).fillna(0)
    agg["total_usd"] = agg["diesel_usd_amount"] + agg["petrol_usd_amount"]
    top = agg.sort_values("total_usd", ascending=False).head(15).reset_index()
    st.subheader("Top companies by USD sales")
    st.bar_chart(top.set_index("company_name")["total_usd"])

# -----------------------
# PAGE: Discounts
# -----------------------
def page_discounts():
    st.header("Discounted Transactions")
    df = safe_filter(DATA["discounts"], "created_at", start_date, end_date)
    if df is None or df.empty:
        st.info("No discounted transactions.")
        return
    st.dataframe(df)
    # simple summary
    total_disc = df["discount"].sum() if "discount" in df.columns else 0
    st.metric("Total Discount Amount", f"${total_disc:,.2f}")

# -----------------------
# PAGE: Expired Coupons
# -----------------------
def page_expired_coupons():
    st.header("Expired / Active Coupons (aged)")
    df = safe_filter(DATA["exp_coupons"], "activation_date", start_date, end_date)
    if df is None or df.empty:
        st.info("No coupon activity.")
        return
    st.dataframe(df)

# -----------------------
# PAGE: Stock & Price
# -----------------------
def page_stock_prices():
    st.header("Stock & Price History")
    stock = safe_filter(DATA["stock"], "date", start_date, end_date)
    price = safe_filter(DATA["price"], "date", start_date, end_date)

    st.subheader("Stock snapshot (recent)")
    st.dataframe(stock)

    st.subheader("Price history")
    st.dataframe(price)

# -----------------------
# PAGE: Lubricants
# -----------------------
def page_lubricants():
    st.header("Lubricants — Cash & Card")
    cash = DATA.get("cash", pd.DataFrame())
    # We used lubricants export files in ETL: lubricants_cash_report.csv & lubricants_card_report.csv
    # Read them on demand (not cached here)
    lub_cash_path = "lubricants_cash_report.csv"
    lub_card_path = "lubricants_card_report.csv"

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(lub_cash_path):
            df_cash = pd.read_csv(lub_cash_path, parse_dates=["created_at"], low_memory=False)
            df_cash = safe_filter(df_cash, "created_at", start_date, end_date)
            st.subheader("Cash Lubricants")
            st.dataframe(df_cash)
            csv = df_cash.to_csv(index=False).encode("utf-8")
            st.download_button("Download cash lubricants", data=csv, file_name=lub_cash_path)
        else:
            st.info("lubricants_cash_report.csv not present.")

    with col2:
        if os.path.exists(lub_card_path):
            df_card = pd.read_csv(lub_card_path, parse_dates=["created_at"], low_memory=False)
            df_card = safe_filter(df_card, "created_at", start_date, end_date)
            st.subheader("Card Lubricants")
            st.dataframe(df_card)
            csv = df_card.to_csv(index=False).encode("utf-8")
            st.download_button("Download card lubricants", data=csv, file_name=lub_card_path)
        else:
            st.info("lubricants_card_report.csv not present.")

# -----------------------
# Router
# -----------------------
PAGE_FN = {
    "Dashboard": page_dashboard,
    "Sales (All channels)": page_sales,
    "Company Fuel": page_company_fuel,
    "Discounts": page_discounts,
    "Expired Coupons": page_expired_coupons,
    "Stock & Prices": page_stock_prices,
    "Lubricants": page_lubricants,
}

# Render page
try:
    PAGE_FN.get(page, page_dashboard)()
except Exception as e:
    st.error(f"Error rendering page: {e}")

# -----------------------
# Admin area (only visible to Admin role)
# -----------------------
if role == "Admin":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Admin Tools")
    col_a, col_b = st.sidebar.columns([1, 1])
    if col_a.button("Reload CSVs now"):
        # Clear the cached load_data function and reload
        load_data_from_csv.clear()
        DATA.update(load_data_from_csv())
        st.experimental_rerun()
    if col_b.button("Show data folder"):
        st.sidebar.write(os.listdir("."))

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built for operations — ETL writes CSVs; dashboard auto-refreshes every 60s. Contact dev for customizations.")

import os
from datetime import datetime as dt
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import create_engine, text
from queries import (
    coupon_sales_query, card_sales_query, cash_sales_query, swipe_sales_query,
    stock_query, price_query, discounted_transaction_query, exp_coupons_query,
Lubs_card_query, Lubs_cash_query, daily_fuel_sales
)

# -----------------------
# DB CONNECTIONS
# -----------------------
MARIADB_CONN_STR = "mysql+pymysql://reports:PcbPkHvrQDUJZG53@41.72.151.66:3306/trek_prod"
POSTGRES_CONN_STR = "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets?options=-csearch_path=public"

mariadb_engine = create_engine(MARIADB_CONN_STR, pool_pre_ping=True)
postgres_engine = create_engine(POSTGRES_CONN_STR, pool_pre_ping=True)

# -----------------------
# Auto-refresh
# -----------------------
_refresh_count = st_autorefresh(interval=60*1000, key="fuel_dash_autorefresh")

# -----------------------
# Page config & CSS
# -----------------------
st.set_page_config(page_title="Fuel Ops Dashboard", layout="wide")
st.markdown("""
<style>
.last-updated-box { background-color: #1E88E5; color: #fff; padding: 10px 16px; border-radius: 10px; font-weight: 600; display: inline-block; }
[data-testid="stDataFrameWrapper"] table { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load data from DB safely
# -----------------------
@st.cache_data(ttl=60)
def load_data():
    data = {}
    def query_db(query, engine):
        try:
            return pd.read_sql(text(query), engine)
        except Exception:
            return pd.DataFrame()
    data["coupon"] = query_db(coupon_sales_query, mariadb_engine)
    data["card"] = query_db(card_sales_query, mariadb_engine)
    data["cash"] = query_db(cash_sales_query, postgres_engine)
    data["swipe"] = query_db(swipe_sales_query, postgres_engine)
    data["stock"] = query_db(stock_query, postgres_engine)
    data["price"] = query_db(price_query, postgres_engine)
    data["discounts"] = query_db(discounted_transaction_query, mariadb_engine)
    data["exp_coupons"] = query_db(exp_coupons_query, mariadb_engine)
    data["lubs_card"] = query_db(Lubs_card_query, mariadb_engine)
    data["lubs_cash"] = query_db(Lubs_cash_query, postgres_engine)
    data["daily_fuel_sales"] = query_db(daily_fuel_sales, mariadb_engine)

    return data

DATA = load_data()

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Controls")
    role = st.selectbox("Role", ["Viewer", "Admin"])
    today = pd.Timestamp.today().normalize()
    start_date, end_date = st.date_input("Date range", value=(today-pd.Timedelta(days=1), today))
    st.markdown("---")
    page = st.radio("Go to", ["Dashboard", "Sales (All channels)", "Company Fuel", "Discounts", "Expired Coupons", "Stock & Prices", "Lubricants"])

# -----------------------
# Helpers
# -----------------------
def safe_filter(df, date_col, start, end):
    if df.empty or date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]

def sum_safe(df, col):
    return df[col].sum() if not df.empty and col in df.columns else 0

# -----------------------
# Top row
# -----------------------
col1, col2, col3 = st.columns([3,1,1])
with col1:
    st.title("⛽ Fuel Operations Dashboard")
with col2:
    last_txt = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="last-updated-box">🔄 Last Updated<br><small>{last_txt}</small></div>', unsafe_allow_html=True)
with col3:
    st.metric("Auto refresh ticks", _refresh_count)
    st.markdown(f"**Role:** {role}")

# -----------------------
# PAGE: Dashboard
# -----------------------
def page_dashboard():
    st.subheader("Overview / KPIs")
    coupon = safe_filter(DATA["coupon"], "sale_date", start_date, end_date)
    card = safe_filter(DATA["card"], "sale_date", start_date, end_date)
    cash = safe_filter(DATA["cash"], "sale_date", start_date, end_date)
    swipe = safe_filter(DATA["swipe"], "sale_date", start_date, end_date)

    k1, k2, k3, k4, k5 = st.columns([1.2]*4 + [1.4])
    k1.metric("💳 Card Litres", f"{sum_safe(card,'total_litres'):,.0f}")
    k2.metric("🎟️ Coupon Litres", f"{sum_safe(coupon,'total_litres'):,.0f}")
    k3.metric("💵 Cash Litres", f"{sum_safe(cash,'total_litres'):,.0f}")
    k4.metric("💻 Swipe Litres", f"{sum_safe(swipe,'total_litres'):,.0f}")
    k5.metric("🧾 Total Revenue", f"${sum_safe(card,'total_amount')+sum_safe(coupon,'total_amount')+sum_safe(cash,'total_amount')+sum_safe(swipe,'total_amount'):,.0f}")

    st.markdown("### Combined Sales Trend")
    combined = pd.concat([coupon.assign(channel="Coupon"),
                          card.assign(channel="Card"),
                          cash.assign(channel="Cash"),
                          swipe.assign(channel="Swipe")], ignore_index=True, sort=False)
    if not combined.empty:
        grouped = combined.groupby(["sale_date", "channel"], dropna=False)["total_litres"].sum().reset_index()
        line = alt.Chart(grouped).mark_line(point=True).encode(
            x="sale_date:T", y="total_litres:Q", color="channel:N",
            tooltip=["sale_date:T","channel:N","total_litres:Q"]
        ).properties(height=320)
        st.altair_chart(line.interactive(), use_container_width=True)

# -----------------------
# PAGE: Sales (All channels)
# -----------------------
def page_sales():
    st.header("Sales — All Channels")
    coupon = safe_filter(DATA["coupon"], "sale_date", start_date, end_date)
    card = safe_filter(DATA["card"], "sale_date", start_date, end_date)
    cash = safe_filter(DATA["cash"], "sale_date", start_date, end_date)
    swipe = safe_filter(DATA["swipe"], "sale_date", start_date, end_date)

    tabs = st.tabs(["Combined","By Channel","Downloads"])
    with tabs[0]:
        combined = pd.concat([coupon, card, cash, swipe], ignore_index=True, sort=False)
        st.dataframe(combined)
    with tabs[1]:
        st.subheader("Card"); st.dataframe(card)
        st.subheader("Coupon"); st.dataframe(coupon)
        st.subheader("Cash"); st.dataframe(cash)
        st.subheader("Swipe"); st.dataframe(swipe)
    with tabs[2]:
        for name, df in [("coupon_sales.csv", coupon), ("card_sales.csv", card), ("cash_sales.csv", cash), ("swipe_sales.csv", swipe)]:
            if not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {name}", data=csv, file_name=name, mime="text/csv")

# -----------------------
# PAGE: Company Fuel
# -----------------------
def page_company_fuel():
    st.header("Company Fuel — Daily Breakdown")

    df = safe_filter(DATA["daily_fuel_sales"], "date", start_date, end_date)

    if df.empty:
        st.warning("No company fuel data found for the selected date range.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⛽ Diesel USD (Litres)", f"{df['diesel_usd_litres'].sum():,.0f}")
    col2.metric("💵 Diesel USD (Amount)", f"${df['diesel_usd_amount'].sum():,.2f}")
    col3.metric("⛽ Petrol USD (Litres)", f"{df['petrol_usd_litres'].sum():,.0f}")
    col4.metric("💵 Petrol USD (Amount)", f"${df['petrol_usd_amount'].sum():,.2f}")

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Company Fuel CSV", csv, "company_fuel.csv", "text/csv")

    # Table
    st.subheader("Daily Company Fuel Summary")
    st.dataframe(df)





# -----------------------
# PAGE: Discounts
# -----------------------
def page_discounts():
    st.header("Discounted Transactions")
    df = safe_filter(DATA["discounts"], "created_at", start_date, end_date)
    st.dataframe(df)

# -----------------------
# PAGE: Expired Coupons
# -----------------------
def page_expired_coupons():
    st.header("Expired / Active Coupons")
    df = safe_filter(DATA["exp_coupons"], "activation_date", start_date, end_date)
    st.dataframe(df)

# -----------------------
# PAGE: Stock & Prices
# -----------------------
def page_stock_prices():
    st.header("Stock & Price History")
    stock = safe_filter(DATA["stock"], "date", start_date, end_date)
    price = safe_filter(DATA["price"], "date", start_date, end_date)
    st.subheader("Stock snapshot"); st.dataframe(stock)
    st.subheader("Price history"); st.dataframe(price)

# -----------------------
# PAGE: Lubricants
# -----------------------
def page_lubricants():
    st.header("Lubricants — Cash & Card")

    lubs_card = safe_filter(DATA["lubs_card"], "created_at", start_date, end_date)
    lubs_cash = safe_filter(DATA["lubs_cash"], "created_at", start_date, end_date)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🛢️ Card Lubes Revenue", f"${sum_safe(lubs_card, 'amount'):,.2f}")
    with col2:
        st.metric("🛢️ Cash Lubes Revenue", f"${sum_safe(lubs_cash, 'amount'):,.2f}")

    tabs = st.tabs(["Card Lubricants", "Cash Lubricants", "Downloads"])

    with tabs[0]:
        st.subheader("Card Lubricants Sales")
        st.dataframe(lubs_card)

    with tabs[1]:
        st.subheader("Cash Lubricants Sales")
        st.dataframe(lubs_cash)

    with tabs[2]:
        if not lubs_card.empty:
            st.download_button(
                "Download Card Lubricants CSV",
                lubs_card.to_csv(index=False).encode("utf-8"),
                "lubs_card.csv",
                "text/csv"
            )
        if not lubs_cash.empty:
            st.download_button(
                "Download Cash Lubricants CSV",
                lubs_cash.to_csv(index=False).encode("utf-8"),
                "lubs_cash.csv",
                "text/csv"
            )

# -----------------------
# Router
# -----------------------
PAGES = {
    "Dashboard": page_dashboard,
    "Sales (All channels)": page_sales,
    "Company Fuel": page_company_fuel,
    "Discounts": page_discounts,
    "Expired Coupons": page_expired_coupons,
    "Stock & Prices": page_stock_prices,
    "Lubricants": page_lubricants,
}

try:
    PAGES.get(page, page_dashboard)()
except Exception as e:
    st.error(f"Error rendering page: {e}")

import os
from datetime import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text

# -----------------------
# Your original queries (imported)
# -----------------------
from queries import (
    coupon_sales_query, card_sales_query, cash_sales_query, swipe_sales_query,
    stock_query, price_query, discounted_transaction_query, exp_coupons_query,
    Lubs_card_query, Lubs_cash_query, daily_litres_sale_query, daily_fuel_sales_query
)

# -----------------------
# DB CONNECTIONS (use env if available)
# -----------------------
MARIADB_CONN_STR = os.getenv("MARIADB_CONN_STR", "mysql+pymysql://reports:PcbPkHvrQDUJZG53@41.72.151.66:3306/trek_prod")
POSTGRES_CONN_STR = os.getenv("POSTGRES_CONN_STR", "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets?options=-csearch_path=public")

mariadb_engine = create_engine(MARIADB_CONN_STR, pool_pre_ping=True)
postgres_engine = create_engine(POSTGRES_CONN_STR, pool_pre_ping=True)

# -----------------------
# Cache directory for incremental snapshots
# -----------------------
CACHE_DIR = Path(".cache_data")
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------
# Streamlit config + CSS (preserve look but colorful)
# -----------------------
st.set_page_config(page_title="Fuel Ops Dashboard", layout="wide")
st.markdown("""
<style>
body { background: linear-gradient(135deg, #0f172a, #1e293b); }
.last-updated-box {
  background-color: #6d28d9;
  color: #fff;
  padding: 10px 16px;
  border-radius: 10px;
  font-weight: 600;
  display: inline-block;
  box-shadow: 0 0 12px rgba(109,40,217,0.7);
}
[data-testid="stDataFrameWrapper"] table {
  font-size: 13px;
  border: 1px solid #6366f1;
  border-radius: 8px;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1e1b4b, #312e81);
  color: white;
}
section[data-testid="stSidebar"] * { color: black !important; }
/* =====  Animations, Neon, Metrics, Shimmer, Hover, Glow  */

/* Smooth fade-in for the entire app */
body, .main, [data-testid="stAppViewContainer"] {
  animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Neon glowing metrics */
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
  text-shadow: 0 0 6px rgba(255,255,255,0.6), 0 0 12px rgba(99,102,241,0.8);
  color: #ffffff !important;
}

/* Gradient metric containers */
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  padding: 12px 16px;
  border-radius: 14px;
  box-shadow: 0 0 16px rgba(124,58,237,0.55);
  transition: 0.25s ease;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-4px) scale(1.02);
  box-shadow: 0 4px 20px rgba(139,92,246,0.75);
}

/* Neon borders around charts */
[data-testid="stVerticalBlock"] .vega-embed {
  border: 1px solid #6366f1;
  box-shadow: 0 0 15px rgba(99,102,241,0.4);
  border-radius: 10px;
  padding: 8px;
  animation: glowPulse 3s infinite alternate;
}
@keyframes glowPulse {
  from { box-shadow: 0 0 10px rgba(99,102,241,0.3); }
  to   { box-shadow: 0 0 22px rgba(167,139,250,0.75); }
}

/* Colorful DataFrames (tables) */
[data-testid="stDataFrameWrapper"] table {
  border: 2px solid #4c1d95;
  border-radius: 10px;
  overflow: hidden;
  background: rgba(49,46,129,0.6);
  color: #fff !important;
}
[data-testid="stDataFrameWrapper"] tbody tr:hover {
  background: rgba(139,92,246,0.4) !important;
  cursor: pointer;
}

/* Sidebar glow and hover */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1e1b4b, #312e81);
  box-shadow: 0 0 25px rgba(124,58,237,0.6);
}
section[data-testid="stSidebar"] label:hover {
  color: #a78bfa !important;
  transition: 0.25s;
}

/* Buttons: neon + ripple */
button[kind="secondary"], button[kind="primary"] {
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  border: none !important;
  color: white !important;
  border-radius: 10px !important;
  box-shadow: 0 0 12px rgba(124,58,237,0.6);
  transition: 0.25s ease;
  overflow: hidden;
  position: relative;
}
button:hover {
  transform: translateY(-3px);
  box-shadow: 0 0 18px rgba(167,139,250,0.8) !important;
}
button:active {
  transform: scale(0.97);
}

/* Shimmer loading blocks */
.shimmer {
  width: 100%;
  height: 22px;
  border-radius: 6px;
  background: linear-gradient(90deg, #2e2b55 0%, #4c1d95 40%, #2e2b55 80%);
  background-size: 200% 100%;
  animation: shimmerMove 1.5s infinite;
}
@keyframes shimmerMove {
  from { background-position: 200% 0; }
  to   { background-position: -200% 0; }
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Auto-refresh
# -----------------------
_refresh_count = st.experimental_get_query_params().get("refresh_tick", [0])[0]
# keep original behavior via st_autorefresh if desired; leave as manual for speed

# -----------------------
# Helpers: fast filter + safe sum
# -----------------------

def safe_filter(df, date_col, start, end):
    if df is None or df.empty or date_col not in df.columns:
        return df if df is not None else pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)]


def sum_safe(df, col):
    return df[col].sum() if df is not None and not df.empty and col in df.columns else 0

# -----------------------
# Incremental loader (max speed)
# - stores snapshot CSV and a marker timestamp
# - only fetches new rows (WHERE date_col > last_marker)
# - works for both MariaDB and Postgres
# -----------------------

def _read_marker(marker_file: Path):
    if not marker_file.exists():
        return None
    txt = marker_file.read_text().strip()
    return pd.to_datetime(txt, errors='coerce') if txt else None


def _write_marker(marker_file: Path, ts):
    marker_file.write_text(str(pd.to_datetime(ts)))


def _snapshot_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _marker_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_marker.txt"


def load_incremental(name: str, query: str, date_col: str, engine):
    """Load table incrementally. If snapshot missing -> load full. Else load only rows newer than marker."""
    snap = _snapshot_path(name)
    marker = _marker_path(name)

    # If no snapshot: full load
    if not snap.exists():
        try:
            df = pd.read_sql(text(query), engine)
        except Exception:
            return pd.DataFrame()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if not df.empty:
                _write_marker(marker, df[date_col].max())
        df.to_csv(snap, index=False)
        return df

    # snapshot exists
    try:
        df_snap = pd.read_csv(snap)
    except Exception:
        df_snap = pd.DataFrame()

    last_ts = _read_marker(marker)
    if last_ts is None:
        # safety: treat as full reload
        try:
            df_full = pd.read_sql(text(query), engine)
            if date_col in df_full.columns:
                df_full[date_col] = pd.to_datetime(df_full[date_col], errors='coerce')
                _write_marker(marker, df_full[date_col].max())
            df_full.to_csv(snap, index=False)
            return df_full
        except Exception:
            return df_snap

    # Build incremental SQL: wrap original query as subquery to add date filter
    incremental_sql = f"SELECT * FROM ({query}) AS t WHERE t.{date_col} > :last_ts"
    try:
        df_new = pd.read_sql(text(incremental_sql), engine, params={"last_ts": last_ts})
    except Exception:
        df_new = pd.DataFrame()

    if df_new.empty:
        # ensure types
        if date_col in df_snap.columns and not pd.api.types.is_datetime64_any_dtype(df_snap[date_col]):
            df_snap[date_col] = pd.to_datetime(df_snap[date_col], errors='coerce')
        return df_snap

    # combine
    if date_col in df_new.columns:
        df_new[date_col] = pd.to_datetime(df_new[date_col], errors='coerce')

    df_combined = pd.concat([df_snap, df_new], ignore_index=True, sort=False)
    if date_col in df_combined.columns:
        df_combined.sort_values(by=date_col, inplace=True)
        _write_marker(marker, df_combined[date_col].max())
    df_combined.to_csv(snap, index=False)
    return df_combined

# -----------------------
# Map pages to queries + engines + date columns
# -----------------------
PAGE_QUERIES = {
    "Dashboard": [
        ("coupon", coupon_sales_query, mariadb_engine, "sale_date"),
        ("card", card_sales_query, mariadb_engine, "sale_date"),
        ("cash", cash_sales_query, postgres_engine, "sale_date"),
        ("swipe", swipe_sales_query, postgres_engine, "sale_date"),
    ],
    "Sales (All channels)": [
        ("coupon", coupon_sales_query, mariadb_engine, "sale_date"),
        ("card", card_sales_query, mariadb_engine, "sale_date"),
        ("cash", cash_sales_query, postgres_engine, "sale_date"),
        ("swipe", swipe_sales_query, postgres_engine, "sale_date"),
    ],
    "Company Fuel": [
        ("daily_fuel_sales", daily_fuel_sales_query, mariadb_engine, "date"),
        ("daily_litres_sale", daily_litres_sale_query, mariadb_engine, "sale_date"),
    ],
    "Discounts": [
        ("discounts", discounted_transaction_query, mariadb_engine, "created_at"),
    ],
    "Expired Coupons": [
        ("exp_coupons", exp_coupons_query, mariadb_engine, "activation_date"),
    ],
    "Stock & Prices": [
        ("stock", stock_query, postgres_engine, "date"),
        ("price", price_query, postgres_engine, "date"),
    ],
    "Lubricants": [
        ("lubs_card", Lubs_card_query, mariadb_engine, "created_at"),
        ("lubs_cash", Lubs_cash_query, postgres_engine, "created_at"),
    ],
    "Daily Litres sales": [
        ("daily_litres_sale", daily_litres_sale_query, mariadb_engine, "sale_date"),
    ]
}

# -----------------------
# Sidebar controls (preserve)
# -----------------------
with st.sidebar:
    st.header("Controls")
    role = st.selectbox("Role", ["Viewer", "Admin"])
    today = pd.Timestamp.today().normalize()
    start_date, end_date = st.date_input("Date range", value=(today - pd.Timedelta(days=1), today))
    st.markdown("---")
    page = st.radio("Go to", list(PAGE_QUERIES.keys()))

# -----------------------
# Load only queries for the selected page in parallel (fast)
# -----------------------
required = PAGE_QUERIES.get(page, [])
DATA = {}
if required:
    with ThreadPoolExecutor(max_workers=min(6, len(required))) as ex:
        futures = {ex.submit(load_incremental, name, q, date_col, engine): name for name, q, engine, date_col in required}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                DATA[name] = fut.result()
            except Exception:
                DATA[name] = pd.DataFrame()

# -----------------------
# Top row (preserve visuals)
# -----------------------
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("⛽ Fuel Operations Dashboard")
with col2:
    last_txt = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="last-updated-box">🔄 Last Updated<br><small>{last_txt}</small></div>', unsafe_allow_html=True)
with col3:
    st.metric("Auto refresh ticks", 0)
    st.markdown(f"**Role:** {role}")

# -----------------------
# Page implementations (preserve behavior)
# -----------------------

def page_dashboard():
    st.subheader("Overview / KPIs")
    coupon = safe_filter(DATA.get("coupon", pd.DataFrame()), "sale_date", start_date, end_date)
    card = safe_filter(DATA.get("card", pd.DataFrame()), "sale_date", start_date, end_date)
    cash = safe_filter(DATA.get("cash", pd.DataFrame()), "sale_date", start_date, end_date)
    swipe = safe_filter(DATA.get("swipe", pd.DataFrame()), "sale_date", start_date, end_date)

    k1, k2, k3, k4, k5 = st.columns([1.2]*4 + [1.4])
    k1.metric("💳 Card Litres", f"{sum_safe(card,'total_litres'):,.0f}")
    k2.metric("🎟️ Coupon Litres", f"{sum_safe(coupon,'total_litres'):,.0f}")
    k3.metric("💵 Cash Litres", f"{sum_safe(cash,'total_litres'):,.0f}")
    k4.metric("💻 Swipe Litres", f"{sum_safe(swipe,'total_litres'):,.0f}")
    k5.metric("🧾 Total Revenue", f"${sum_safe(card,'total_amount')+sum_safe(coupon,'total_amount')+sum_safe(cash,'total_amount')+sum_safe(swipe,'total_amount'):,.0f}")

    st.markdown("### Combined Sales Trend")
    combined = pd.concat([coupon.assign(channel="Coupon"), card.assign(channel="Card"), cash.assign(channel="Cash"), swipe.assign(channel="Swipe")], ignore_index=True, sort=False)
    if not combined.empty:
        if not pd.api.types.is_datetime64_any_dtype(combined["sale_date"]):
            combined["sale_date"] = pd.to_datetime(combined["sale_date"], errors='coerce')
        grouped = combined.groupby(["sale_date", "channel"], dropna=False)["total_litres"].sum().reset_index()
        line = alt.Chart(grouped).mark_line(point=True).encode(x="sale_date:T", y="total_litres:Q", color="channel:N", tooltip=["sale_date:T","channel:N","total_litres:Q"]).properties(height=320)
        st.altair_chart(line.interactive(), use_container_width=True)


def page_sales():
    st.header("Sales — All Channels")
    coupon = safe_filter(DATA.get("coupon", pd.DataFrame()), "sale_date", start_date, end_date)
    card = safe_filter(DATA.get("card", pd.DataFrame()), "sale_date", start_date, end_date)
    cash = safe_filter(DATA.get("cash", pd.DataFrame()), "sale_date", start_date, end_date)
    swipe = safe_filter(DATA.get("swipe", pd.DataFrame()), "sale_date", start_date, end_date)

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


def page_company_fuel():
    st.header("Company Fuel — Daily Breakdown")
    df = safe_filter(DATA.get("daily_fuel_sales", pd.DataFrame()), "date", start_date, end_date)
    if df.empty:
        st.warning("No company fuel data found for the selected date range.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⛽ Diesel USD (Litres)", f"{df.get('diesel_usd_litres', pd.Series()).sum():,.0f}")
    col2.metric("💵 Diesel USD (Amount)", f"${df.get('diesel_usd_amount', pd.Series()).sum():,.2f}")
    col3.metric("⛽ Petrol USD (Litres)", f"{df.get('petrol_usd_litres', pd.Series()).sum():,.0f}")
    col4.metric("💵 Petrol USD (Amount)", f"${df.get('petrol_usd_amount', pd.Series()).sum():,.2f}")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Company Fuel CSV", csv, "company_fuel.csv", "text/csv")
    st.subheader("Daily Company Fuel Summary")
    st.dataframe(df)


def page_discounts():
    st.header("Discounted Transactions")
    df = safe_filter(DATA.get("discounts", pd.DataFrame()), "created_at", start_date, end_date)
    st.dataframe(df)


def page_expired_coupons():
    st.header("Expired / Active Coupons")
    df = safe_filter(DATA.get("exp_coupons", pd.DataFrame()), "activation_date", start_date, end_date)
    st.dataframe(df)


def page_stock_prices():
    st.header("Stock & Price History")
    stock = safe_filter(DATA.get("stock", pd.DataFrame()), "date", start_date, end_date)
    price = safe_filter(DATA.get("price", pd.DataFrame()), "date", start_date, end_date)
    st.subheader("Stock snapshot"); st.dataframe(stock)
    st.subheader("Price history"); st.dataframe(price)


def page_lubricants():
    st.header("Lubricants — Cash & Card")
    lubs_card = safe_filter(DATA.get("lubs_card", pd.DataFrame()), "created_at", start_date, end_date)
    lubs_cash = safe_filter(DATA.get("lubs_cash", pd.DataFrame()), "created_at", start_date, end_date)
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
            st.download_button("Download Card Lubricants CSV", lubs_card.to_csv(index=False).encode("utf-8"), "lubs_card.csv", "text/csv")
        if not lubs_cash.empty:
            st.download_button("Download Cash Lubricants CSV", lubs_cash.to_csv(index=False).encode("utf-8"), "lubs_cash.csv", "text/csv")


def page_daily_litres_sale():
    st.header("Daily Litres Sale Report")
    df = safe_filter(DATA.get("daily_litres_sale", pd.DataFrame()), "sale_date", start_date, end_date)
    if df.empty:
        st.warning("No daily litres sale data found for the selected date range.")
        return
    st.subheader("Daily Litres Sales")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Daily Litres Sales CSV", csv, "daily_litres_sales.csv", "text/csv")

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
    "Daily Litres sales": page_daily_litres_sale
}

try:
    PAGES.get(page, page_dashboard)()
except Exception as e:
    st.error(f"Error rendering page: {e}")

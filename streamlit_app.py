# dashboard.py
import os
from datetime import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Fuel Ops Dashboard", layout="wide")
CACHE_DIR = Path(".cache_data")
CACHE_DIR.mkdir(exist_ok=True)

# DB connection strings (use env variables in deployment)
MARIADB_CONN_STR = os.getenv(
    "MARIADB_CONN_STR",
    "mysql+pymysql://reports:PcbPkHvrQDUJZG53@41.72.151.66:3306/trek_prod",
)
POSTGRES_CONN_STR = os.getenv(
    "POSTGRES_CONN_STR",
    "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets",
)

# Engines
def get_engines():
    mariadb = create_engine(MARIADB_CONN_STR, pool_pre_ping=True)
    postgres = create_engine(POSTGRES_CONN_STR, pool_pre_ping=True)
    return mariadb, postgres

mariadb_engine, postgres_engine = get_engines()

# -----------------------
# QUERIES (ALL timezone-fixed)
# -----------------------
# Note: MariaDB: convert tz from UTC -> +02:00. Postgres: AT TIME ZONE conversions.
# All `... >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)` ensures UTC-based cutoff.
coupon_sales_query = """
SELECT
    DATE(CONVERT_TZ(created_at, '+00:00', '+02:00')) AS sale_date,
    service_station_id,
    service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM trek_prod.coupon_transaction
WHERE deleted = 0
  AND response_description LIKE '%Success%'
  AND created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
GROUP BY sale_date, service_station_id, service_station_name, product;
"""

card_sales_query = """
SELECT
    DATE(CONVERT_TZ(created_at, '+00:00', '+02:00')) AS sale_date,
    service_station_id,
    service_station AS service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM trek_prod.transaction
WHERE deleted = 0 
  AND debit_txn = 1
  AND created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
GROUP BY sale_date, service_station_id, service_station, product;
"""

cash_sales_query = """
SELECT
    DATE(transacted_at AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare') AS sale_date,
    service_stationid AS site_id,
    service_station AS site_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM public.cash_sale
WHERE transacted_at >= NOW() - INTERVAL '100 days'
GROUP BY sale_date, service_stationid, service_station, product;
"""

swipe_sales_query = """
SELECT
    DATE(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare') AS sale_date,
    site AS site_id,
    site AS site_name,
    product,
    SUM(litres)/100.0 AS total_litres,
    SUM(amount)/100.0 AS total_amount
FROM public.transactions
WHERE type LIKE '%SWIPE%'
  AND created_at >= NOW() - INTERVAL '100 days'
GROUP BY sale_date, site, product;
"""

stock_query = """
SELECT
    (date AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare')::date AS date_local,
    service_station,
    product,
    SUM(amount) AS closing_stock_litres
FROM public.site_stock
WHERE date >= NOW() - INTERVAL '100 days'
GROUP BY date_local, service_station, product
ORDER BY date_local DESC;
"""

price_query = """
SELECT
    (date AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare')::date AS date_local,
    site,
    product,
    AVG(competitor_price) AS price
FROM public.price_comparisons
WHERE date >= NOW() - INTERVAL '100 days'
GROUP BY date_local, site, product
ORDER BY date_local DESC;
"""

discounted_transaction_query = """
SELECT
    CONVERT_TZ(t.created_at, '+00:00', '+02:00') AS created_at_local,
    co.name AS company_name,
    c.first_name,
    t.description,
    t.amount/100 AS amount,
    t.discount/100 AS discount,
    t.discount_type,
    t.litres AS litres,
    t.pan,
    t.unit_price/100 AS unit_price,
    t.discount_litre/100 AS discount_litre
FROM transaction t
LEFT JOIN company co ON t.company_id = co.id
LEFT JOIN customer c ON t.customer_id = c.id
WHERE t.discount_litre NOT LIKE '%0.00%'
  AND t.created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY);
"""

exp_coupons_query = """
SELECT
    c.barcode,
    co.booklet_number,
    CONVERT_TZ(c.activation_date, '+00:00', '+02:00') AS activation_date_local,
    c1.name AS company_name,
    c.status AS coupon_status
FROM coupon c
JOIN coupon_booklet co ON c.coupon_booklet_id = co.id
JOIN company c1 ON co.company_id = c1.id
WHERE c.activation_date IS NOT NULL
  AND c.activation_date >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
  AND c.status LIKE '%ACTIVE%';
"""

Lubs_card_query = """
SELECT
    CONVERT_TZ(created_at, '+00:00', '+02:00') AS created_at_local,
    service_station,
    amount / 100 AS amount,
    litres,
    product,
    description
FROM transaction t
WHERE tid IS NOT NULL
  AND created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
  AND (
    product NOT LIKE '%diesel%' AND
    product NOT LIKE '%petrol%' AND
    product NOT LIKE '%blend%' AND
    description NOT LIKE '%diesel%' AND
    description NOT LIKE '%petrol%' AND
    description NOT LIKE '%blend%' AND
    description NOT LIKE '%MUNC%' AND
    description NOT LIKE '%M&M%'
  );
"""

Lubs_cash_query = """
SELECT
    (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Harare') AS created_at_local,
    product,
    amount / 100 AS amount,
    litres AS quantity
FROM cash_sale
WHERE created_at >= NOW() - INTERVAL '100 days'
  AND product NOT LIKE '%PETROL%'
  AND product NOT LIKE '%DIESEL%'
  AND product NOT LIKE '%BLEND%';
"""

daily_fuel_sales_query = """
SELECT 
    DATE(CONVERT_TZ(t.created_at, '+00:00', '+02:00')) as date_local,
    c.name AS company_name,
    SUM(CASE WHEN t.product LIKE '%USD DIESEL%' 
                OR t.product = 'CRIPPS DIESEL USD' 
                OR t.product = 'GRANITESIDE DIESEL USD'
           THEN t.amount ELSE 0 END) / 100 AS diesel_usd_amount,
    SUM(CASE WHEN t.product LIKE '%USD DIESEL%' 
                OR t.product = 'CRIPPS DIESEL USD' 
                OR t.product = 'GRANITESIDE DIESEL USD'
           THEN t.litres ELSE 0 END) AS diesel_usd_litres,
    SUM(CASE WHEN t.product LIKE '%DIESEL LITRES%'
           THEN t.amount ELSE 0 END) / 100 AS diesel_litres_amount,
    SUM(CASE WHEN t.product LIKE '%DIESEL LITRES%'
           THEN t.litres ELSE 0 END) AS diesel_litres_litres,
    SUM(CASE WHEN t.product LIKE '%USD PETROL%'
           THEN t.amount ELSE 0 END) / 100 AS petrol_usd_amount,
    SUM(CASE WHEN t.product LIKE '%USD PETROL%'
           THEN t.litres ELSE 0 END) AS petrol_usd_litres,
    SUM(CASE WHEN t.product LIKE '%PETROL LITRES%'
           THEN t.amount ELSE 0 END) / 100 AS petrol_litres_amount,
    SUM(CASE WHEN t.product LIKE '%PETROL LITRES%'
           THEN t.litres ELSE 0 END) AS petrol_litres_litres
FROM company c
LEFT JOIN transaction t ON c.id = t.company_id
WHERE t.debit_txn = 1
  AND t.transaction_type = 'SALE'
  AND t.created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
GROUP BY date_local, c.name
ORDER BY date_local, name;
"""

daily_litres_sale_query = """
SELECT
    CONVERT_TZ(t.created_at, '+00:00', '+02:00') AS sale_date_local,
    c.name AS company_name,
    t.service_station AS site,
    t.description AS description,
    t.pan AS card_number,
    t.amount / 100 AS amount,
    t.unit_price / 100 AS price,
    t.product AS product
FROM transaction t
LEFT JOIN company c ON t.company_id = c.id
WHERE t.tid IS NOT NULL
  AND t.created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 100 DAY)
  AND t.description LIKE '%SALE%'
  AND (t.pan LIKE '%DSL%' OR t.pan LIKE '%PTL%')
ORDER BY t.created_at DESC;
"""

# -----------------------
# Helper functions
# -----------------------
def _snapshot_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"

def _marker_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_marker.txt"

def _read_marker(marker_file: Path):
    if not marker_file.exists():
        return None
    txt = marker_file.read_text().strip()
    return pd.to_datetime(txt, errors='coerce') if txt else None

def _write_marker(marker_file: Path, ts):
    marker_file.write_text(str(pd.to_datetime(ts)))

def is_aggregated_query(q: str, date_col: str):
    u = (q or "").upper()
    if "GROUP BY" in u or " DATE(" in u or date_col.lower() in ("sale_date", "date", "date_local"):
        return True
    return False

# Improved incremental loader
def load_incremental(name: str, query: str, date_col: str, engine):
    snap = _snapshot_path(name)
    marker = _marker_path(name)

    # if snapshot missing -> full load baseline
    def save_snapshot(df):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            if not df.empty and not df[date_col].isna().all():
                _write_marker(marker, df[date_col].max())
        df.to_csv(snap, index=False)

    try:
        if not snap.exists():
            df_full = pd.read_sql(text(query), engine)
            save_snapshot(df_full)
            return df_full
    except Exception as e:
        st.error(f"Initial load error for {name}: {e}")
        return pd.DataFrame()

    # load snapshot
    try:
        df_snap = pd.read_csv(snap)
    except Exception:
        df_snap = pd.DataFrame()

    # aggregated: full refresh each time (keeps same-day aggregations accurate)
    if is_aggregated_query(query, date_col):
        try:
            df_full = pd.read_sql(text(query), engine)
            save_snapshot(df_full)
            return df_full
        except Exception as e:
            st.warning(f"Aggregated refresh failed for {name}, returning snapshot: {e}")
            if date_col in df_snap.columns and not pd.api.types.is_datetime64_any_dtype(df_snap[date_col]):
                df_snap[date_col] = pd.to_datetime(df_snap[date_col], errors="coerce")
            return df_snap

    # transactional: incremental fetch
    last_ts = _read_marker(marker)
    if last_ts is None:
        try:
            df_full = pd.read_sql(text(query), engine)
            save_snapshot(df_full)
            return df_full
        except Exception as e:
            st.warning(f"Transactional full reload failed for {name}: {e}")
            return df_snap

    incremental_sql = f"SELECT * FROM ({query}) AS t WHERE t.{date_col} > :last_ts"
    try:
        df_new = pd.read_sql(text(incremental_sql), engine, params={"last_ts": last_ts})
    except Exception as e:
        st.warning(f"Incremental fetch failed for {name}: {e}")
        df_new = pd.DataFrame()

    if df_new.empty:
        if date_col in df_snap.columns and not pd.api.types.is_datetime64_any_dtype(df_snap[date_col]):
            df_snap[date_col] = pd.to_datetime(df_snap[date_col], errors="coerce")
        return df_snap

    # combine
    if date_col in df_new.columns:
        df_new[date_col] = pd.to_datetime(df_new[date_col], errors="coerce")
    df_combined = pd.concat([df_snap, df_new], ignore_index=True, sort=False)
    if date_col in df_combined.columns:
        df_combined.sort_values(by=date_col, inplace=True)
        _write_marker(marker, df_combined[date_col].max())
    df_combined.to_csv(snap, index=False)
    return df_combined

# -----------------------
# Page -> queries map (both DBs used per your input)
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
        ("daily_fuel_sales", daily_fuel_sales_query, mariadb_engine, "date_local"),
        ("daily_litres_sale", daily_litres_sale_query, mariadb_engine, "sale_date_local"),
    ],
    "Discounts": [
        ("discounts", discounted_transaction_query, mariadb_engine, "created_at_local"),
    ],
    "Expired Coupons": [
        ("exp_coupons", exp_coupons_query, mariadb_engine, "activation_date_local"),
    ],
    "Stock & Prices": [
        ("stock", stock_query, postgres_engine, "date_local"),
        ("price", price_query, postgres_engine, "date_local"),
    ],
    "Lubricants": [
        ("lubs_card", Lubs_card_query, mariadb_engine, "created_at_local"),
        ("lubs_cash", Lubs_cash_query, postgres_engine, "created_at_local"),
    ],
    "Daily Litres sales": [
        ("daily_litres_sale", daily_litres_sale_query, mariadb_engine, "sale_date_local"),
    ],
}

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    role = st.selectbox("Role", ["Viewer", "Admin"])
    today = pd.Timestamp.today().normalize()
    start_date, end_date = st.date_input("Date range", value=(today - pd.Timedelta(days=1), today))
    st.markdown("---")
    page = st.radio("Go to", list(PAGE_QUERIES.keys()))
    st.markdown("---")
    if st.button("Force full refresh (clear snapshots)"):
        # delete snapshots
        for f in CACHE_DIR.glob("*.csv"):
            try:
                f.unlink()
            except Exception:
                pass
        for f in CACHE_DIR.glob("*_marker.txt"):
            try:
                f.unlink()
            except Exception:
                pass
        st.experimental_rerun()
    st.markdown("Debug / Connection test")
    if st.button("Test DB connections"):
        try:
            with mariadb_engine.connect() as c:
                r = c.execute(text("SELECT NOW()")).fetchone()
                st.success(f"MySQL OK: {r[0]}")
        except Exception as e:
            st.error(f"MySQL error: {e}")
        try:
            with postgres_engine.connect() as c:
                r = c.execute(text("SELECT NOW()")).fetchone()
                st.success(f"Postgres OK: {r[0]}")
        except Exception as e:
            st.error(f"Postgres error: {e}")

# -----------------------
# Load data for the selected page (parallel)
# -----------------------
required = PAGE_QUERIES.get(page, [])
DATA = {}
if required:
    with ThreadPoolExecutor(max_workers=min(6, len(required))) as ex:
        futures = {
            ex.submit(load_incremental, name, q, date_col, engine): name
            for name, q, engine, date_col in required
        }
        for fut in as_completed(futures):
            nm = futures[fut]
            try:
                DATA[nm] = fut.result()
            except Exception as e:
                st.warning(f"Failed loading {nm}: {e}")
                DATA[nm] = pd.DataFrame()

# -----------------------
# Top bar
# -----------------------
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("⛽ Fuel Operations Dashboard")
with col2:
    last_txt = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div style=\"background:linear-gradient(90deg,#6d28d9,#00bfa6); color:#fff; padding:10px; border-radius:8px;\">🔄 Last Updated<br><small>{last_txt}</small></div>', unsafe_allow_html=True)
with col3:
    st.metric("Auto refresh ticks", 0)
    st.markdown(f"**Role:** {role}")

# -----------------------
# Helper filter and sum
# -----------------------
def safe_filter_local(df, date_col, start, end):
    if df is None or df.empty or date_col not in df.columns:
        return df if df is not None else pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)]

def sum_safe_local(df, col):
    return float(df[col].sum()) if df is not None and not df.empty and col in df.columns else 0.0

# -----------------------
# Page implementations (same layout)
# -----------------------
def page_dashboard():
    st.subheader("Overview / KPIs")
    coupon = safe_filter_local(DATA.get("coupon", pd.DataFrame()), "sale_date", start_date, end_date)
    card = safe_filter_local(DATA.get("card", pd.DataFrame()), "sale_date", start_date, end_date)
    cash = safe_filter_local(DATA.get("cash", pd.DataFrame()), "sale_date", start_date, end_date)
    swipe = safe_filter_local(DATA.get("swipe", pd.DataFrame()), "sale_date", start_date, end_date)

    k1, k2, k3, k4, k5 = st.columns([1.2]*4 + [1.4])
    k1.metric("💳 Card Litres", f"{sum_safe_local(card,'total_litres'):,.0f}")
    k2.metric("🎟️ Coupon Litres", f"{sum_safe_local(coupon,'total_litres'):,.0f}")
    k3.metric("💵 Cash Litres", f"{sum_safe_local(cash,'total_litres'):,.0f}")
    k4.metric("💻 Swipe Litres", f"{sum_safe_local(swipe,'total_litres'):,.0f}")
    total_rev = sum_safe_local(card,'total_amount') + sum_safe_local(coupon,'total_amount') + sum_safe_local(cash,'total_amount') + sum_safe_local(swipe,'total_amount')
    k5.metric("🧾 Total Revenue", f"${total_rev:,.0f}")

    st.markdown("### Combined Sales Trend")
    combined = pd.concat([
        coupon.assign(channel="Coupon"),
        card.assign(channel="Card"),
        cash.assign(channel="Cash"),
        swipe.assign(channel="Swipe")
    ], ignore_index=True, sort=False)
    if not combined.empty:
        if not pd.api.types.is_datetime64_any_dtype(combined["sale_date"]):
            combined["sale_date"] = pd.to_datetime(combined["sale_date"], errors="coerce")
        grouped = combined.groupby(["sale_date", "channel"], dropna=False)["total_litres"].sum().reset_index()
        line = alt.Chart(grouped).mark_line(point=True).encode(
            x="sale_date:T", y="total_litres:Q", color="channel:N", tooltip=["sale_date:T","channel:N","total_litres:Q"]
        ).properties(height=320)
        st.altair_chart(line.interactive(), use_container_width=True)

def page_sales():
    st.header("Sales — All Channels")
    coupon = safe_filter_local(DATA.get("coupon", pd.DataFrame()), "sale_date", start_date, end_date)
    card = safe_filter_local(DATA.get("card", pd.DataFrame()), "sale_date", start_date, end_date)
    cash = safe_filter_local(DATA.get("cash", pd.DataFrame()), "sale_date", start_date, end_date)
    swipe = safe_filter_local(DATA.get("swipe", pd.DataFrame()), "sale_date", start_date, end_date)

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
            if df is not None and not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {name}", data=csv, file_name=name, mime="text/csv")

def page_company_fuel():
    st.header("Company Fuel — Daily Breakdown")
    df = safe_filter_local(DATA.get("daily_fuel_sales", pd.DataFrame()), "date_local", start_date, end_date)
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
    df = safe_filter_local(DATA.get("discounts", pd.DataFrame()), "created_at_local", start_date, end_date)
    st.dataframe(df)

def page_expired_coupons():
    st.header("Expired / Active Coupons")
    df = safe_filter_local(DATA.get("exp_coupons", pd.DataFrame()), "activation_date_local", start_date, end_date)
    st.dataframe(df)

def page_stock_prices():
    st.header("Stock & Price History")
    stock = safe_filter_local(DATA.get("stock", pd.DataFrame()), "date_local", start_date, end_date)
    price = safe_filter_local(DATA.get("price", pd.DataFrame()), "date_local", start_date, end_date)
    st.subheader("Stock snapshot"); st.dataframe(stock)
    st.subheader("Price history"); st.dataframe(price)

def page_lubricants():
    st.header("Lubricants — Cash & Card")
    lubs_card = safe_filter_local(DATA.get("lubs_card", pd.DataFrame()), "created_at_local", start_date, end_date)
    lubs_cash = safe_filter_local(DATA.get("lubs_cash", pd.DataFrame()), "created_at_local", start_date, end_date)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🛢️ Card Lubes Revenue", f"${sum_safe_local(lubs_card, 'amount'):,.2f}")
    with col2:
        st.metric("🛢️ Cash Lubes Revenue", f"${sum_safe_local(lubs_cash, 'amount'):,.2f}")
    tabs = st.tabs(["Card Lubricants", "Cash Lubricants", "Downloads"])
    with tabs[0]:
        st.subheader("Card Lubricants Sales")
        st.dataframe(lubs_card)
    with tabs[1]:
        st.subheader("Cash Lubricants Sales")
        st.dataframe(lubs_cash)
    with tabs[2]:
        if lubs_card is not None and not lubs_card.empty:
            st.download_button("Download Card Lubricants CSV", lubs_card.to_csv(index=False).encode("utf-8"), "lubs_card.csv", "text/csv")
        if lubs_cash is not None and not lubs_cash.empty:
            st.download_button("Download Cash Lubricants CSV", lubs_cash.to_csv(index=False).encode("utf-8"), "lubs_cash.csv", "text/csv")

def page_daily_litres_sale():
    st.header("Daily Litres Sale Report")
    df = safe_filter_local(DATA.get("daily_litres_sale", pd.DataFrame()), "sale_date_local", start_date, end_date)
    if df.empty:
        st.warning("No daily litres sale data found for the selected date range.")
        return
    st.subheader("Daily Litres Sales")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Daily Litres Sales CSV", csv, "daily_litres_sales.csv", "text/csv")

# Router
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

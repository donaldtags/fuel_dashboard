import streamlit as st
import pandas as pd
import altair as alt

# ---------- Load Data ----------
@st.cache_data
def load_data():
    coupon = pd.read_csv("coupon_sales.csv", parse_dates=["sale_date"])
    card = pd.read_csv("card_sales.csv", parse_dates=["sale_date"])
    cash = pd.read_csv("cash_sales.csv", parse_dates=["sale_date"])
    stock = pd.read_csv("site_stock.csv", parse_dates=["date"])
    price = pd.read_csv("price_history.csv", parse_dates=["date"])
    swipe = pd.read_csv("swipe_sales.csv", parse_dates=["sale_date"])
    discounts = pd.read_csv("discounted_transactions.csv", parse_dates=["created_at"])
    exp_coupons = pd.read_csv("expired_coupons_report.csv", parse_dates=["activation_date"])
    company_fuel = pd.read_csv("company_fuel_report.csv", parse_dates=["date"])


    return coupon, card, cash, stock, price, swipe, discounts, exp_coupons, company_fuel

coupon_df, card_df, cash_df, stock_df, price_df, swipe_df, discounts_df, exp_coupons_df, company_fuel_df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", [
    "Fuel Dashboard",
    "Sales Report",
    "Discounts Report",
    "Expired Coupons",
    "Company Fuel Report"
])

# ---------- Global Date Filter ----------
min_candidates = []
for df, col in [
    (coupon_df, "sale_date"),
    (card_df, "sale_date"),
    (cash_df, "sale_date"),
    (swipe_df, "sale_date"),
]:
    if not df.empty:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        val = df[col].min()
        if pd.notnull(val):
            min_candidates.append(val)

start_dt = min(min_candidates) if min_candidates else pd.Timestamp.today()
end_dt = pd.Timestamp.today()

def filter_by_date(df, date_column):
    return df[(df[date_column] >= start_dt) & (df[date_column] <= end_dt)]

with st.sidebar:
    st.markdown("### 📅 Date Range Filter")
    date_range = st.date_input(
        "Select date range",
        value=(start_dt.date(), end_dt.date()),
        min_value=start_dt.date(),
        max_value=end_dt.date()
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
    else:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[0])

# ---------- Fuel Dashboard ----------
if page == "Fuel Dashboard":
    st.title("⛽ Fuel Sales & Stock Dashboard")
    st.markdown("---")
    coupon_filtered = filter_by_date(coupon_df, "sale_date")
    card_filtered = filter_by_date(card_df, "sale_date")
    cash_filtered = filter_by_date(cash_df, "sale_date")
    swipe_filtered = filter_by_date(swipe_df, "sale_date")

    total_litres = coupon_filtered['total_litres'].sum() + card_filtered['total_litres'].sum() + \
                   cash_filtered['total_litres'].sum() + swipe_filtered['total_litres'].sum()
    total_revenue = coupon_filtered['total_amount'].sum() + card_filtered['total_amount'].sum() + \
                    cash_filtered['total_amount'].sum() + swipe_filtered['total_amount'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💳 Card Litres", f"{card_filtered['total_litres'].sum():,.0f}")
    col2.metric("🎟️ Coupon Litres", f"{coupon_filtered['total_litres'].sum():,.0f}")
    col3.metric("💵 Cash Litres", f"{cash_filtered['total_litres'].sum():,.0f}")
    col4.metric("💻 Swipe Litres", f"{swipe_filtered['total_litres'].sum():,.0f}")
    col5.metric("🧾 Total Revenue", f"${total_revenue:,.0f}")

    combined_df = pd.concat([
        coupon_filtered.assign(channel='Coupon'),
        card_filtered.assign(channel='Card'),
        cash_filtered.assign(channel='Cash'),
        swipe_filtered.assign(channel='Swipe')
    ])
    grouped = combined_df.groupby(['sale_date', 'channel'])['total_litres'].sum().reset_index()

    st.subheader("📈 Daily Sales Trend")
    st.altair_chart(
        alt.Chart(grouped).mark_line(point=True).encode(
            x='sale_date:T',
            y='total_litres:Q',
            color='channel:N',
            tooltip=['sale_date:T', 'channel:N', 'total_litres:Q']
        ).interactive().properties(height=400),
        use_container_width=True
    )

    st.subheader("📊 Total Litres per Channel")
    bar_data = combined_df.groupby('channel')['total_litres'].sum().reset_index()
    st.altair_chart(
        alt.Chart(bar_data).mark_bar().encode(
            x='channel:N',
            y='total_litres:Q',
            color='channel:N',
            tooltip=['channel:N', 'total_litres:Q']
        ),
        use_container_width=True
    )

# ---------- Sales Report ----------
elif page == "Sales Report":
    st.title("📊 Sales Report")
    st.markdown("---")
    coupon_filtered = filter_by_date(coupon_df, "sale_date")
    card_filtered = filter_by_date(card_df, "sale_date")
    cash_filtered = filter_by_date(cash_df, "sale_date")
    swipe_filtered = filter_by_date(swipe_df, "sale_date")

    sales_report = pd.concat([coupon_filtered, card_filtered, cash_filtered, swipe_filtered], ignore_index=True)
    combined_grouped = sales_report.groupby('sale_date')['total_litres'].sum().reset_index()

    st.subheader("📈 Total Litres Over Time")
    st.altair_chart(
        alt.Chart(combined_grouped).mark_line(point=True).encode(
            x='sale_date:T',
            y='total_litres:Q',
            tooltip=['sale_date:T', 'total_litres:Q']
        ).interactive(),
        use_container_width=True
    )
    st.subheader("🗂 Detailed Sales Data")
    st.dataframe(sales_report)

# ---------- Expired Coupons ----------
elif page == "Expired Coupons":
    st.title("📊 Expired Coupons Report")
    st.markdown("---")
    st.dataframe(exp_coupons_df)

# ---------- Discounts Report ----------
elif page == "Discounts Report":
    st.title("💸 Discounted Transactions")
    st.markdown("---")
    st.dataframe(discounts_df)

# ---------- Company Fuel Report ----------
elif page == "Company Fuel Report":
    st.title("⛽ Company Fuel Sales Report")
    st.markdown("---")

    fuel_filtered = filter_by_date(company_fuel_df, "date")
    st.subheader("🗂 Company Fuel Table")
    st.dataframe(fuel_filtered)

    diesel_usd = fuel_filtered.groupby('date')['diesel_usd_amount'].sum().reset_index()
    petrol_usd = fuel_filtered.groupby('date')['petrol_usd_amount'].sum().reset_index()

    st.subheader("📈 Diesel USD Sales Over Time")
    st.altair_chart(
        alt.Chart(diesel_usd).mark_line(point=True).encode(
            x='date:T',
            y='diesel_usd_amount:Q',
            tooltip=['date:T', 'diesel_usd_amount:Q']
        ).interactive(),
        use_container_width=True
    )

    st.subheader("📈 Petrol USD Sales Over Time")
    st.altair_chart(
        alt.Chart(petrol_usd).mark_line(point=True).encode(
            x='date:T',
            y='petrol_usd_amount:Q',
            tooltip=['date:T', 'petrol_usd_amount:Q']
        ).interactive(),
        use_container_width=True
    )

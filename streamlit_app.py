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
    companies_daily_litres_sales = pd.read_csv("companies_daily_litres_sales.csv", parse_dates=["MONTH"])
    return coupon, card, cash, stock, price, swipe, discounts, exp_coupons, company_fuel, companies_daily_litres_sales

# Load datasets
(
    coupon_df, card_df, cash_df, stock_df, price_df,
    swipe_df, discounts_df, exp_coupons_df, company_fuel_df, companies_daily_litres_sales_df
) = load_data()

# ---------- UI Theme ----------
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    color: #1E88E5 !important;
    font-weight: 900 !important;
    font-size: 28px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
pages = [
    "Fuel Dashboard",
    "Sales Report",
    "Discounts Report",
    "Expired Coupons",
    "Company Fuel Report",
    "Company Litres Report"
]
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", pages)

# ---------- Global Date Filter ----------
today = pd.Timestamp.today().normalize()
yesterday = today - pd.Timedelta(days=1)
start_dt = yesterday
end_dt = today

def filter_by_date(df, date_column):
    return df[(df[date_column] >= start_dt) & (df[date_column] <= end_dt)]

with st.sidebar:
    st.markdown("### 📅 Date Range Filter")
    date_range = st.date_input(
        "Select date range",
        value=(yesterday, today),
        min_value=pd.to_datetime("2000-01-01"),
        max_value=today
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
    else:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[0])

# ---------- Download Button ----------
def download_button(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ════════════════════════════════════════════
#              PAGE 1: FUEL DASHBOARD
# ════════════════════════════════════════════
if page == "Fuel Dashboard":
    st.title("⛽ Fuel Sales & Stock Dashboard")
    st.markdown("---")

    coupon_f = filter_by_date(coupon_df, "sale_date")
    card_f = filter_by_date(card_df, "sale_date")
    cash_f = filter_by_date(cash_df, "sale_date")
    swipe_f = filter_by_date(swipe_df, "sale_date")

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💳 Card Litres", f"{card_f['total_litres'].sum():,.0f}")
    col2.metric("🎟️ Coupon Litres", f"{coupon_f['total_litres'].sum():,.0f}")
    col3.metric("💵 Cash Litres", f"{cash_f['total_litres'].sum():,.0f}")
    col4.metric("💻 Swipe Litres", f"{swipe_f['total_litres'].sum():,.0f}")
    col5.metric("🧾 Total Revenue",
                f"${(coupon_f['total_amount'].sum() + card_f['total_amount'].sum() + cash_f['total_amount'].sum() + swipe_f['total_amount'].sum()):,.0f}"
                )

    # Combined chart
    combined = pd.concat([
        coupon_f.assign(channel="Coupon"),
        card_f.assign(channel="Card"),
        cash_f.assign(channel="Cash"),
        swipe_f.assign(channel="Swipe")
    ])
    grouped = combined.groupby(["sale_date", "channel"])["total_litres"].sum().reset_index()

    st.subheader("📈 Daily Sales Trend")
    st.altair_chart(
        alt.Chart(grouped).mark_line(point=True).encode(
            x=alt.X("sale_date:T", title="Date"),
            y=alt.Y("total_litres:Q", title="Total Litres"),
            color="channel:N",
            tooltip=["sale_date:T", "channel:N", "total_litres:Q"]
        ).interactive(),
        use_container_width=True
    )

    st.subheader("📊 Total Litres per Channel")
    totals = combined.groupby("channel")["total_litres"].sum().reset_index()
    st.altair_chart(
        alt.Chart(totals).mark_bar().encode(
            x="channel:N",
            y="total_litres:Q",
            color="channel:N",
            tooltip=["channel:N", "total_litres:Q"]
        ),
        use_container_width=True
    )

# ════════════════════════════════════════════
#              PAGE 2: SALES REPORT
# ════════════════════════════════════════════
elif page == "Sales Report":
    st.title("📊 Sales Report")
    st.markdown("---")

    dfs = [filter_by_date(df, "sale_date") for df in [coupon_df, card_df, cash_df, swipe_df]]
    filtered_sales = pd.concat(dfs, ignore_index=True)
    grouped = filtered_sales.groupby("sale_date")["total_litres"].sum().reset_index()

    st.subheader("📈 Total Litres Over Time")
    st.altair_chart(
        alt.Chart(grouped).mark_line(point=True).encode(
            x=alt.X("sale_date:T", title="Date"),
            y=alt.Y("total_litres:Q", title="Total Litres"),
            tooltip=["sale_date:T", "total_litres:Q"]
        ).interactive(),
        use_container_width=True
    )

    st.subheader("🗂 Detailed Sales Data")
    download_button(filtered_sales, "sales_filtered.csv")
    st.dataframe(filtered_sales)

# ════════════════════════════════════════════
#          PAGE 3: EXPIRED COUPONS
# ════════════════════════════════════════════
elif page == "Expired Coupons":
    st.title("📊 Expired Coupons Report")
    st.markdown("---")

    exp_f = filter_by_date(exp_coupons_df, "activation_date")
    download_button(exp_f, "expired_coupons_filtered.csv")
    st.dataframe(exp_f)

# ════════════════════════════════════════════
#          PAGE 4: DISCOUNTS REPORT
# ════════════════════════════════════════════
elif page == "Discounts Report":
    st.title("💸 Discounted Transactions")
    st.markdown("---")

    discounts_f = filter_by_date(discounts_df, "created_at")
    download_button(discounts_f, "discounts_filtered.csv")
    st.dataframe(discounts_f)

# ════════════════════════════════════════════
#          PAGE 5: COMPANY FUEL REPORT
# ════════════════════════════════════════════
elif page == "Company Fuel Report":
    st.title("⛽ Company Fuel Sales Report")
    st.markdown("---")

    fuel_f = filter_by_date(company_fuel_df, "date")
    download_button(fuel_f, "company_fuel_filtered.csv")
    st.subheader("🗂 Company Fuel Table")
    st.dataframe(fuel_f)

    diesel = fuel_f.groupby("date")["diesel_usd_amount"].sum().reset_index()
    petrol = fuel_f.groupby("date")["petrol_usd_amount"].sum().reset_index()

    st.subheader("📈 Diesel USD Sales Over Time")
    st.altair_chart(
        alt.Chart(diesel).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("diesel_usd_amount:Q", title="USD Amount"),
            tooltip=["date:T", "diesel_usd_amount:Q"]
        ).interactive(),
        use_container_width=True
    )

    st.subheader("📈 Petrol USD Sales Over Time")
    st.altair_chart(
        alt.Chart(petrol).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("petrol_usd_amount:Q", title="USD Amount"),
            tooltip=["date:T", "petrol_usd_amount:Q"]
        ).interactive(),
        use_container_width=True
    )

# ════════════════════════════════════════════
#          PAGE 6: COMPANY LITRES REPORT
# ════════════════════════════════════════════
elif page == "Company Litres Report":
    st.title("💸 Company Litres Report")
    st.markdown("---")

    litres_f = filter_by_date(companies_daily_litres_sales_df, "MONTH")
    download_button(litres_f, "company_litres_filtered.csv")
    st.dataframe(litres_f)

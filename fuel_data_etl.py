import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# --- DATE RANGE ---
end_dt = datetime.today()
start_dt = end_dt - timedelta(days=300)

# --- CONNECTIONS ---
mariadb_conn_str = "mysql+pymysql://reports:PcbPkHvrQDUJZG53@41.72.151.66:3306/trek_prod"
postgres_conn_str = (
    "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets"
    "?options=-csearch_path=public"
)

mariadb_engine = create_engine(mariadb_conn_str)
postgres_engine = create_engine(postgres_conn_str)

# --- QUERIES ---

coupon_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    service_station_id,
    service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount,
    AVG(unit_price)/100 AS avg_price
FROM trek_prod.coupon_transaction
WHERE deleted = 0
  AND response_description LIKE '%%Success%%'
  AND created_at BETWEEN DATE_SUB(CURDATE(), INTERVAL 300 DAY) AND CURDATE()
GROUP BY sale_date, service_station_id, service_station_name, product;
"""

coupon_redeemed_query = """
SELECT
    DATE(created_at) AS sale_date,
    service_station_id,
    service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount,
    AVG(unit_price)/100 AS avg_price
FROM trek_prod.coupon_transaction
WHERE deleted = 0
  AND response_description LIKE '%%Success%%'
  AND created_at BETWEEN DATE_SUB(CURDATE(), INTERVAL 300 DAY) AND CURDATE()
GROUP BY sale_date, service_station_id, service_station_name, product;
"""

card_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    service_station_id,
    service_station AS service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount,
    AVG(unit_price)/100 AS avg_price
FROM trek_prod.transaction
WHERE deleted = 0
  AND debit_txn = 1
  AND created_at BETWEEN DATE_SUB(CURDATE(), INTERVAL 300 DAY) AND CURDATE()
GROUP BY sale_date, service_station_id, service_station, product;
"""

cash_sales_query = """
SELECT
    DATE(transacted_at) AS sale_date,
    service_stationid AS site_id,
    service_station AS site_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount,
    ROUND((SUM(amount)/NULLIF(SUM(litres),0))::numeric,2) AS avg_price
FROM public.cash_sale
WHERE transacted_at >= CURRENT_DATE - INTERVAL '300 days'
GROUP BY DATE(transacted_at), service_stationid, service_station, product
ORDER BY sale_date;
"""

stock_query = f"""
SELECT
    date,
    service_station,
    product,
    SUM(amount) AS closing_stock_litres
FROM public.site_stock
WHERE date BETWEEN '{start_dt.date()}' AND '{end_dt.date()}'
GROUP BY date, service_station, product
ORDER BY date DESC;
"""

swipe_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    site AS site_id,
    site AS site_name,
    product,
    SUM(litres)/100.0 AS total_litres,
    SUM(amount)/100.0 AS total_amount,
    ROUND(((SUM(amount)/100.0)/NULLIF((SUM(litres)/100.0),0))::numeric,2) AS avg_price
FROM public.transactions
WHERE type LIKE '%%SWIPE%%'
  AND created_at >= CURRENT_DATE - INTERVAL '300 days'
GROUP BY DATE(created_at), site, product
ORDER BY sale_date;
"""

price_query = """
SELECT
    date,
    site,
    product,
    AVG(competitor_price) AS price
FROM public.price_comparisons
WHERE date BETWEEN CURRENT_DATE - INTERVAL '300 days' AND CURRENT_DATE
GROUP BY date, site, product
ORDER BY date DESC;
"""

discounted_transaction_query = """
SELECT
    t.created_at AS created_at,
    co.name,
    c.first_name,
    t.description,
    t.amount/100 AS amount,
    t.discount/100 AS discount,
    t.discount_type,
    t.litres AS litres,
    t.pan,
    t.unit_price/100 AS unit_price,
    t.discount_litre/100 AS discount_litre
FROM `transaction` t
LEFT JOIN company co ON t.company_id = co.id
LEFT JOIN customer c ON t.customer_id = c.id
WHERE
    t.created_at BETWEEN CURRENT_DATE - INTERVAL 300 DAY AND CURRENT_DATE
    AND t.tid IS NOT NULL
    AND t.discount_litre NOT LIKE '%%0.00%%'
ORDER BY t.created_at DESC;
"""

exp_coupons_query = """
SELECT
    c.barcode,
    co.booklet_number,
    c.activation_date,
    c1.name AS company_name,
    c.status AS coupon_status
FROM coupon c
JOIN coupon_booklet co ON c.coupon_booklet_id = co.id
JOIN company c1 ON co.company_id = c1.id
WHERE c.activation_date IS NOT NULL
  AND c.activation_date < DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
  AND c.status LIKE '%%ACTIVE%%';
"""

# ✅ Lubricants Queries (fixed)
Lubs_cash_query = """
SELECT
    created_at,
    product,
    amount
FROM public.cash_sale
WHERE product NOT LIKE '%PETROL%'
  AND product NOT LIKE '%DIESEL%'
  AND product NOT LIKE '%BATT%'
  AND created_at BETWEEN CURRENT_DATE - INTERVAL '5 years' AND CURRENT_DATE;
"""

Lubs_card_query = """
SELECT
    created_at,
    description,
    product,
    amount/100 AS amount
FROM trek_prod.transaction
WHERE tid IS NOT NULL
  AND description LIKE '%%SALE%%'
  AND product NOT LIKE '%%PETROL%%'
  AND product NOT LIKE '%%DIESEL%%'
  AND created_at BETWEEN DATE_SUB(CURDATE(), INTERVAL 5 YEAR) AND CURDATE();
"""

company_fuel_query = f"""
SELECT DATE(t.created_at) AS date,
       c.name AS company_name,
       SUM(CASE WHEN t.product LIKE '%USD DIESEL%' OR t.product = 'CRIPPS DIESEL USD' OR t.product = 'GRANITESIDE DIESEL USD' THEN t.amount ELSE 0 END) / 100 AS diesel_usd_amount,
       SUM(CASE WHEN t.product LIKE '%USD DIESEL%' OR t.product = 'CRIPPS DIESEL USD' OR t.product = 'GRANITESIDE DIESEL USD' THEN t.litres ELSE 0 END) AS diesel_usd_litres,
       SUM(CASE WHEN t.product LIKE '%DIESEL LITRES%' THEN t.amount ELSE 0 END) / 100 AS diesel_litres_amount,
       SUM(CASE WHEN t.product LIKE '%DIESEL LITRES%' THEN t.litres ELSE 0 END) AS diesel_litres_litres,
       SUM(CASE WHEN t.product LIKE '%USD PETROL%' THEN t.amount ELSE 0 END) / 100 AS petrol_usd_amount,
       SUM(CASE WHEN t.product LIKE '%USD PETROL%' THEN t.litres ELSE 0 END) AS petrol_usd_litres,
       SUM(CASE WHEN t.product LIKE '%PETROL LITRES%' THEN t.amount ELSE 0 END) / 100 AS petrol_litres_amount,
       SUM(CASE WHEN t.product LIKE '%PETROL LITRES%' THEN t.litres ELSE 0 END) AS petrol_litres_litres
FROM company c
LEFT JOIN transaction t ON c.id = t.company_id
WHERE t.debit_txn = 1
  AND t.transaction_type = 'SALE'
  AND t.created_at BETWEEN '{start_dt.date()}' AND '{end_dt.date()}'
GROUP BY DATE(t.created_at), c.name
ORDER BY date, company_name;
"""

# --- LOAD DATA FUNCTION ---
def load_data():
    print("Loading coupon sales...")
    coupon_df = pd.read_sql(coupon_sales_query, mariadb_engine)

    print("Loading card sales...")
    card_df = pd.read_sql(card_sales_query, mariadb_engine)

    print("Loading stock data...")
    stock_df = pd.read_sql(text(stock_query), postgres_engine)

    print("Loading price data...")
    price_df = pd.read_sql(text(price_query), postgres_engine)

    print("Loading swipe sales...")
    swipe_df = pd.read_sql(text(swipe_sales_query), postgres_engine)

    print("Loading cash sales...")
    cash_df = pd.read_sql(text(cash_sales_query), postgres_engine)

    print("Loading discounted transactions...")
    discounts_df = pd.read_sql(discounted_transaction_query, mariadb_engine)

    print("Loading expired coupons report...")
    exp_coupons_df = pd.read_sql(exp_coupons_query, mariadb_engine)

    print("Loading cash lubricants...")
    lubricants_cash_df = pd.read_sql(text(Lubs_cash_query), postgres_engine)

    print("Loading card lubricants...")
    lubricants_card_df = pd.read_sql(Lubs_card_query, mariadb_engine)

    print("Loading company fuel sales (diesel/petrol)...")
    company_fuel_df = pd.read_sql(text(company_fuel_query), mariadb_engine)

    return (
        coupon_df, card_df, stock_df, price_df, swipe_df, cash_df,
        discounts_df, exp_coupons_df, lubricants_cash_df, lubricants_card_df,
        company_fuel_df
    )

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    (
        coupon_df, card_df, stock_df, price_df, swipe_df, cash_df,
        discounts_df, exp_coupons_df, lubricants_cash_df, lubricants_card_df,
        company_fuel_df
    ) = load_data()

    # Export to CSVs
    coupon_df.to_csv("coupon_sales.csv", index=False)
    card_df.to_csv("card_sales.csv", index=False)
    stock_df.to_csv("site_stock.csv", index=False)
    price_df.to_csv("price_history.csv", index=False)
    swipe_df.to_csv("swipe_sales.csv", index=False)
    cash_df.to_csv("cash_sales.csv", index=False)
    discounts_df.to_csv("discounted_transactions.csv", index=False)
    exp_coupons_df.to_csv("expired_coupons_report.csv", index=False)
    lubricants_card_df.to_csv("lubricants_card_report.csv", index=False)
    lubricants_cash_df.to_csv("lubricants_cash_report.csv", index=False)
    company_fuel_df.to_csv("company_fuel_report.csv", index=False)

    print("✅ Data exported to CSV. Ready for analysis or dashboard.")

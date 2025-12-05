#!/usr/bin/env python3
"""
etl_api.py

Fetches data from MariaDB API (/transactions) and Postgres, writes CSVs,
and updates last_updated.txt. Can run once (--once) or repeatedly (--interval).
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from sqlalchemy import create_engine

from queries import (
    coupon_sales_query,
    card_sales_query,
    cash_sales_query,
    swipe_sales_query,
    stock_query,
    price_query,
    discounted_transaction_query,
    exp_coupons_query,
    Lubs_cash_query,
    Lubs_card_query,
    company_fuel_query,
    companies_daily_litres_sales_query,
)

# -------------------------
# Logging
# -------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("fuel_etl_api")

# -------------------------
# Default date range: last 300 days
# -------------------------
end_dt = datetime.today()
start_dt = end_dt - timedelta(days=300)

# -------------------------
# Postgres connection
# -------------------------
postgres_conn_str = "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets?options=-csearch_path=public"
postgres_engine = create_engine(postgres_conn_str, pool_pre_ping=True)

# -------------------------
# MariaDB API endpoint
# -------------------------
MARIADB_API = "http://41.72.151.66:5000/get/transactions"  # example

# -------------------------
# Helpers
# -------------------------
def fetch_mariadb_api(params):
    """Fetch data from MariaDB API using GET"""
    try:
        resp = requests.get(MARIADB_API, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error("MariaDB API error: %s", e)
        return pd.DataFrame()


def fetch_postgres(sql):
    """Fetch from Postgres"""
    return pd.read_sql(sql, postgres_engine)


def save_csv(df, filename, output_folder="."):
    path = os.path.join(output_folder, filename)
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", filename, len(df))


def write_last_updated(output_folder="."):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(output_folder, "last_updated.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ts + "\n")
    return ts


# -------------------------
# Main ETL logic
# -------------------------
def etl_run(output_folder="."):
    logger.info("Starting ETL run...")

    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    # ----------------- MariaDB queries via API -----------------
    logger.info("Fetching coupon sales from MariaDB API")
    coupon_df = fetch_mariadb_api({"query": coupon_sales_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching card sales from MariaDB API")
    card_df = fetch_mariadb_api({"query": card_sales_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching discounted transactions from MariaDB API")
    discounts_df = fetch_mariadb_api({"query": discounted_transaction_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching expired coupons from MariaDB API")
    exp_coupons_df = fetch_mariadb_api({"query": exp_coupons_query})

    logger.info("Fetching company fuel from MariaDB API")
    company_fuel_df = fetch_mariadb_api({"query": company_fuel_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching companies daily litres sales from MariaDB API")
    companies_litres_df = fetch_mariadb_api({"query": companies_daily_litres_sales_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching lubricants cash from MariaDB API")
    lub_cash_df = fetch_mariadb_api({"query": Lubs_cash_query.format(start_date=start_date_str, end_date=end_date_str)})

    logger.info("Fetching lubricants card from MariaDB API")
    lub_card_df = fetch_mariadb_api({"query": Lubs_card_query.format(start_date=start_date_str, end_date=end_date_str)})

    # ----------------- Postgres queries -----------------
    logger.info("Fetching cash sales from Postgres")
    cash_df = fetch_postgres(cash_sales_query.format(start_date=start_date_str, end_date=end_date_str))

    logger.info("Fetching swipe sales from Postgres")
    swipe_df = fetch_postgres(swipe_sales_query.format(start_date=start_date_str, end_date=end_date_str))

    logger.info("Fetching stock from Postgres")
    stock_df = fetch_postgres(stock_query.format(start_date=start_date_str, end_date=end_date_str))

    logger.info("Fetching price history from Postgres")
    price_df = fetch_postgres(price_query.format(start_date=start_date_str, end_date=end_date_str))

    # ----------------- Save CSVs -----------------
    save_csv(coupon_df, "coupon_sales.csv", output_folder)
    save_csv(card_df, "card_sales.csv", output_folder)
    save_csv(cash_df, "cash_sales.csv", output_folder)
    save_csv(swipe_df, "swipe_sales.csv", output_folder)
    save_csv(stock_df, "site_stock.csv", output_folder)
    save_csv(price_df, "price_history.csv", output_folder)
    save_csv(discounts_df, "discounted_transactions.csv", output_folder)
    save_csv(exp_coupons_df, "expired_coupons_report.csv", output_folder)
    save_csv(lub_cash_df, "lubricants_cash_report.csv", output_folder)
    save_csv(lub_card_df, "lubricants_card_report.csv", output_folder)
    save_csv(company_fuel_df, "company_fuel_report.csv", output_folder)
    save_csv(companies_litres_df, "companies_daily_litres_sales.csv", output_folder)

    ts = write_last_updated(output_folder)
    logger.info("ETL completed at %s", ts)


# -------------------------
# Main loop
# -------------------------
def main(args):
    output_folder = args.output or "."
    interval = args.interval
    once = args.once

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if once or (interval == 0):
        etl_run(output_folder)
        return

    backoff_seconds = 5
    try:
        while True:
            start_time = time.time()
            try:
                etl_run(output_folder)
                backoff_seconds = 5
            except Exception as e:
                logger.exception("ETL run error: %s", e)
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 300)
            elapsed = time.time() - start_time
            sleep_for = max(0, interval - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        logger.info("ETL stopped by user")
    finally:
        try:
            postgres_engine.dispose()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuel ETL via MariaDB API + Postgres")
    parser.add_argument("--interval", type=int, default=60, help="Run interval in seconds (default 60)")
    parser.add_argument("--once", action="store_true", help="Run ETL once and exit")
    parser.add_argument("--output", type=str, default=".", help="CSV output folder")
    args = parser.parse_args()

    main(args)

import os
from sqlalchemy import create_engine, text

# -------------------------------
# DEFINE CONNECTION STRINGS
# -------------------------------

MARIADB_CONN_STR = os.getenv(
    "MARIADB_CONN_STR",
    "mysql+pymysql://reports:PcbPkHvrQDUJZG53@41.72.151.66:3306/trek_prod"
)

POSTGRES_CONN_STR = os.getenv(
    "POSTGRES_CONN_STR",
    "postgresql+psycopg2://reports:5vELF2V7OpRPOT@41.72.151.66:5432/site_sheets?options=-csearch_path=public"
)

# -------------------------------
# TEST MYSQL
# -------------------------------
try:
    mysql_engine = create_engine(MARIADB_CONN_STR)
    with mysql_engine.connect() as conn:
        result = conn.execute(text("SELECT NOW()"))
        print("MYSQL OK:", result.scalar())
except Exception as e:
    print("MYSQL ERROR:", e)

# -------------------------------
# TEST POSTGRES
# -------------------------------
try:
    pg_engine = create_engine(POSTGRES_CONN_STR)
    with pg_engine.connect() as conn:
        result = conn.execute(text("SELECT NOW()"))
        print("POSTGRES OK:", result.scalar())
except Exception as e:
    print("POSTGRES ERROR:", e)

from sqlalchemy import inspect

from fuel_data_etl import postgres_engine, discounts_df

insp = inspect(postgres_engine)
print(insp.get_columns("transactions", schema="public"))
print(discounts_df.columns.tolist())


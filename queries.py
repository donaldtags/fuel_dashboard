# queries.py

coupon_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    service_station_id,
    service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM trek_prod.coupon_transaction
WHERE deleted = 0
  AND response_description LIKE '%%Success%%'
GROUP BY sale_date, service_station_id, service_station_name, product;
"""

card_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    service_station_id,
    service_station AS service_station_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM trek_prod.transaction
WHERE deleted = 0 AND debit_txn = 1
GROUP BY sale_date, service_station_id, service_station, product;
"""

cash_sales_query = """
SELECT
    DATE(transacted_at) AS sale_date,
    service_stationid AS site_id,
    service_station AS site_name,
    product,
    SUM(litres) AS total_litres,
    SUM(amount)/100 AS total_amount
FROM public.cash_sale
GROUP BY DATE(transacted_at), service_stationid, service_station, product;
"""

swipe_sales_query = """
SELECT
    DATE(created_at) AS sale_date,
    site AS site_id,
    site AS site_name,
    product,
    SUM(litres)/100.0 AS total_litres,
    SUM(amount)/100.0 AS total_amount
FROM public.transactions
WHERE type LIKE '%%SWIPE%%'
GROUP BY DATE(created_at), site, product;
"""

stock_query = """
SELECT
    date,
    service_station,
    product,
    SUM(amount) AS closing_stock_litres
FROM public.site_stock
GROUP BY date, service_station, product
ORDER BY date DESC;
"""

price_query = """
SELECT
    date,
    site,
    product,
    AVG(competitor_price) AS price
FROM public.price_comparisons
GROUP BY date, site, product
ORDER BY date DESC;
"""

discounted_transaction_query = """
SELECT
    t.created_at AS created_at,
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
FROM `transaction` t
LEFT JOIN company co ON t.company_id = co.id
LEFT JOIN customer c ON t.customer_id = c.id
WHERE t.discount_litre NOT LIKE '%%0.00%%';
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

Lubs_card_query ="""sELECT
    created_at,
    service_station,
    amount / 100 AS amount,
    litres,
    product,
    description
FROM transaction t
WHERE tid IS NOT NULL
  AND (
    product NOT LIKE '%diesel%' AND
    product NOT LIKE '%petrol%' AND
    product NOT LIKE '%blend%' AND
    description NOT LIKE '%diesel%' AND
    description NOT LIKE '%petrol%' AND
    description NOT LIKE '%blend%'
      and description not like '%MUNC%'
      and description not like '%M&M%'
      
    )

"""

Lubs_cash_query ="""SELECT
    created_at,
    product,
    amount / 100 AS amount,
    litres AS quantity
FROM cash_sale
WHERE product NOT like '%PETROL%'
  AND product NOT like '%DIESEL%'
  AND product NOT like '%BLEND%'

"""

daily_fuel_sales = """
SELECT DATE(t.created_at) as date,
       c.name AS company_name,
       SUM(CASE WHEN t.product LIKE '%USD DIESEL%' OR t.product = 'CRIPPS DIESEL USD' OR t.product = 'GRANITESIDE DIESEL USD'THEN t.amount ELSE 0 END) / 100 AS diesel_usd_amount,
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
GROUP BY DATE(t.created_at), c.name
ORDER BY date, name
"""
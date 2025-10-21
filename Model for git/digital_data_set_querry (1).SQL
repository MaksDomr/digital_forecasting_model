-- DAILY REPORT 2025: costs by source + vertical newbies (delivery) from SQUIRREL
WITH
-- 1) Классификация кампаний по источнику
digital_cost_raw AS (
  SELECT
    DATE(install_date) AS day,
    country_id,
    campaign_name,
    cost,
    cost_modelled,
    CASE
      WHEN LOWER(campaign_name) LIKE '%courier%'   THEN 'courier'
      WHEN LOWER(campaign_name) LIKE '%rh_exec%'   THEN 'rh_driver'
      WHEN LOWER(campaign_name) LIKE '%rh_client%' THEN 'rh_passenger'
      ELSE NULL
    END AS src
  FROM `indriver-digital.automation.tbl_event_horizon_full`
  WHERE DATE(install_date) BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
),

digital_cost AS (
  SELECT
    day, country_id, src,
    SUM(cost)          AS cost,
    SUM(cost_modelled) AS cost_modelled
  FROM digital_cost_raw
  WHERE src IS NOT NULL
  GROUP BY day, country_id, src
),

-- 2) Косты по источникам в отдельные колонки
cost_by_src AS (
  SELECT
    day,
    country_id,
    SUM(CASE WHEN src = 'courier'      THEN cost          ELSE 0 END) AS cost_courier,
    SUM(CASE WHEN src = 'rh_driver'    THEN cost          ELSE 0 END) AS cost_rh_driver,
    SUM(CASE WHEN src = 'rh_passenger' THEN cost          ELSE 0 END) AS cost_rh_passenger,

    SUM(CASE WHEN src = 'courier'      THEN cost_modelled ELSE 0 END) AS cost_modelled_courier,
    SUM(CASE WHEN src = 'rh_driver'    THEN cost_modelled ELSE 0 END) AS cost_modelled_rh_driver,
    SUM(CASE WHEN src = 'rh_passenger' THEN cost_modelled ELSE 0 END) AS cost_modelled_rh_passenger
  FROM digital_cost
  GROUP BY day, country_id
),

-- 3a) PAID vertical newbies (delivery) по новой логике (SQUIRREL)
vertical_newbies_delivery_paid AS (
  SELECT
    DATE(event_date) AS day,
    country_id,
    SUM(new_pass_with_ride) AS vertical_newbies_delivery_paid
  FROM `indriver-digital.automation.tbl_squirrel_attributed_events_common_v1_1`
  WHERE
    media_source IS NOT NULL
    AND LOWER(media_source) NOT LIKE '%organic%'
    AND vertical = 'Delivery'
    AND EXTRACT(DAY FROM event_date - install_date) <= 550
    AND DATE(event_date) BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
  GROUP BY day, country_id
),

-- 3b) ORGANIC vertical newbies (delivery) по новой логике (SQUIRREL)
vertical_newbies_delivery_organic AS (
  SELECT
    DATE(event_date) AS day,
    country_id,
    SUM(new_pass_with_ride) AS vertical_newbies_delivery_organic
  FROM `indriver-digital.automation.tbl_squirrel_attributed_events_common_v1_1`
  WHERE
    (LOWER(media_source) LIKE '%organic%' OR media_source IS NULL)
    AND vertical = 'Delivery'
    AND EXTRACT(DAY FROM event_date - install_date) <= 550
    AND DATE(event_date) BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
  GROUP BY day, country_id
),

-- 4) Справочник стран
country_map AS (
  SELECT DISTINCT country_id, country_name
  FROM `indriver-bi.heap.vw_macroregion_mapping`
),

-- 5) Ключи day×country из всех источников
keys AS (
  SELECT day, country_id FROM cost_by_src
  UNION DISTINCT
  SELECT day, country_id FROM vertical_newbies_delivery_paid
  UNION DISTINCT
  SELECT day, country_id FROM vertical_newbies_delivery_organic
)

-- 6) Финал
SELECT
  k.day                                                AS day,
  DATE_TRUNC(k.day, MONTH)                             AS month,
  cm.country_name,
  k.country_id,

  -- Косты
  COALESCE(c.cost_courier, 0)                 AS cost_courier,
  COALESCE(c.cost_rh_driver, 0)               AS cost_rh_driver,
  COALESCE(c.cost_rh_passenger, 0)            AS cost_rh_passenger,

  COALESCE(c.cost_modelled_courier, 0)        AS cost_modelled_courier,
  COALESCE(c.cost_modelled_rh_driver, 0)      AS cost_modelled_rh_driver,
  COALESCE(c.cost_modelled_rh_passenger, 0)   AS cost_modelled_rh_passenger,

  -- Новички (delivery) из SQUIRREL по новой логике
  COALESCE(p.vertical_newbies_delivery_paid, 0)     AS vertical_newbies_delivery_paid,
  COALESCE(o.vertical_newbies_delivery_organic, 0)  AS vertical_newbies_delivery_organic

FROM keys k
LEFT JOIN cost_by_src                        c  ON c.day = k.day AND c.country_id = k.country_id
LEFT JOIN vertical_newbies_delivery_paid     p  ON p.day = k.day AND p.country_id = k.country_id
LEFT JOIN vertical_newbies_delivery_organic  o  ON o.day = k.day AND o.country_id = k.country_id
LEFT JOIN country_map                        cm ON cm.country_id = k.country_id
WHERE k.day BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
ORDER BY day, country_name;

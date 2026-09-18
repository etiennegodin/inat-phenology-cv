CREATE SCHEMA IF NOT EXISTS staged;

CREATE OR REPLACE TABLE staged.inat_request_photos AS
WITH unpacked AS (
SELECT
    o.id as observation_id,
    UNNEST(o.photos, RECURSIVE := true)

FROM staged.inat_requests o
)
SELECT
    observation_id,
    id AS photo_id,
FROM unpacked

-- photos --
CREATE OR REPLACE TABLE staged.photos AS

WITH unpacked AS (
SELECT
    o.id AS observation_id,
    UNNEST(o.photos, RECURSIVE := true)

FROM staged.observations o
)

SELECT
    observation_id,
    id AS photo_id,
    url

FROM unpacked
WHERE id in (SELECT raw_id::INT FROM raw.ingested_photos)

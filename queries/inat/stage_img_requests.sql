INSERT OR REPLACE INTO staged.img_requests
(photo_id,observation_id)

WITH unpacked AS (
SELECT
    o.id as observation_id,
    UNNEST(o.photos, RECURSIVE := true)

FROM staged.obs_requests o
)
SELECT
    id AS photo_id,
    observation_id,
FROM unpacked

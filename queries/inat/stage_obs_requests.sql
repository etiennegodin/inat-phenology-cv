INSERT OR REPLACE INTO staged.obs_requests
(id,uuid,photos)

WITH unpacked AS (
    SELECT obs.*
    FROM (
        SELECT from_json(
            raw_json,
            '{
                "uuid":"VARCHAR",
                "id":"INT",
                "photos":[{
                    "id":"INT",
                }],
            }'
            ) AS obs
            FROM raw.obs_requests
        )
)
SELECT
id,
"uuid",
photos
FROM unpacked

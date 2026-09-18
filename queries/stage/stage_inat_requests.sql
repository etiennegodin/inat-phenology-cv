

INSERT INTO staged.inat_requests
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
            FROM raw.inat_api
        )
)
SELECT
id,
"uuid",
photos
FROM unpacked
ON CONFLICT DO NOTHING;

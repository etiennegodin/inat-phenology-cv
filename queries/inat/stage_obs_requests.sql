INSERT OR REPLACE INTO serving.observations
(observation_id,uuid,photos, taxon, ancestor_ids)

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
                "taxon" : { "id" : "INT",
                    "ancestor_ids" : ["INT"]}
            }'
            ) AS obs
            FROM raw.obs_requests
        )
)
SELECT
id as observation_id,
"uuid",
photos,
taxon.id,
taxon.ancestor_ids
FROM unpacked;


INSERT OR REPLACE INTO serving.photos
(photo_id,observation_id)

WITH unpacked AS (
SELECT
    o.observation_id,
    UNNEST(o.photos, RECURSIVE := true)

FROM serving.observations o
)
SELECT
    id AS photo_id,
    observation_id,
FROM unpacked

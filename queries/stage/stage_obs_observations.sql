CREATE SCHEMA IF NOT EXISTS staged;

CREATE OR REPLACE TABLE staged.observations AS

WITH unpacked AS (
    SELECT obs.*
    FROM (
        SELECT from_json(
            raw_json,
            '{
                "uuid":"VARCHAR",
                "id":"UBIGINT",
                "reviewed_by":["UBIGINT"],
                "owners_identification_from_vision":"NULL",
                "identifications_count":"UBIGINT",
                "user":{
                    "id":"UBIGINT",
                    "created_at":"VARCHAR",
                    "orcid":"NULL"
                },
                "description":"NULL",
                "tags":["NULL"],
                "observation_photos":[{"id":"UBIGINT"}],
                "comments_count":"UBIGINT",
                "faves_count":"UBIGINT",
                "outlinks":["NULL"],
                "community_taxon_id":"NULL",
                "taxon_geoprivacy":"NULL",
                "place_ids":["UBIGINT"],
                "identifications":[{
                    "id":"UBIGINT",
                    "uuid":"VARCHAR",
                    "created_at":"VARCHAR",
                    "user":{
                        "id":"UBIGINT",
                        "login":"VARCHAR",
                        "observations_count":"UBIGINT",
                        "identifications_count":"UBIGINT",
                        "species_count":"UBIGINT"
                    },
                    "body":"VARCHAR",
                    "category":"VARCHAR",
                    "current":"BOOLEAN",
                    "own_observation":"BOOLEAN",
                    "vision":"BOOLEAN",
                    "disagreement":"NULL",
                    "previous_observation_taxon_id":"NULL",
                    "taxon_id":"UBIGINT"
                }]
            }'
            ) AS obs
            FROM raw.inat_api
        )
)
--sql-fluff:off
SELECT DISTINCT u.* EXCLUDE(u.'description',     -- noqa
                            u.tags,              -- noqa
                            u.taxon_geoprivacy), -- noqa
u.user.id AS user_id,
d.observed_on,
COALESCE(
        try_strptime(d.observed_on_string, '%Y-%m-%d %H:%M:%S %z'), -- Handles -0400
        try_strptime(d.observed_on_string, '%Y-%m-%d %H:%M:%S %Z'), -- Handles UTC
        try_cast(d.observed_on_string AS TIMESTAMPTZ)               -- Fallback to default
    ) AS observed_on_string,
COALESCE(
        try_strptime(d.created_at, '%Y-%m-%d %H:%M:%S %z'), -- Handles -0400
        try_strptime(d.created_at, '%Y-%m-%d %H:%M:%S %Z'), -- Handles UTC
        try_cast(d.created_at AS TIMESTAMPTZ)               -- Fallback to default
    ) AS created_at,
d.quality_grade,
d.tag_list,
d.description,
d.license,
d.captive_cultivated,
d.oauth_application_id,
d.latitude,
d.longitude,
d.positional_accuracy,
d.geoprivacy,
d.place_guess,
d.taxon_geoprivacy,
d.coordinates_obscured,
d.positioning_method,
d.taxon_id,
t.phylum,
t.class,
t.order,
t.family,
t.genus,
t.species,
t.rank,
t.rank_level
--s.sampling_pool

FROM unpacked u
JOIN raw.downloads d ON u.id = d.id
JOIN staged.taxa t ON d.taxon_id = t.taxon_id
--LEFT JOIN raw.obs_sample s ON u.uuid = s.uuid
;


-- Manual cleanup


-- inactive taxons:


CREATE OR REPLACE TABLE staged.observations AS

SELECT *


FROM staged.observations
WHERE taxon_id != 241449
AND taxon_id != 157557

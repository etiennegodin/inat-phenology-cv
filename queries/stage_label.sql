CREATE OR REPLACE TABLE staged.label AS


SELECT
    a.observation_id,
    [list_contains(a.attrs.phenology, 13), list_contains(a.attrs.phenology, 14), list_contains(a.attrs.phenology, 15)]::INT[3] AS label

FROM staged.observations o
JOIN staged.annotations a ON o.id = a.observation_id
WHERE a.attrs.phenology IS NOT NULL

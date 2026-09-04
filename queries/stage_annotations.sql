CREATE OR REPLACE TABLE staged.annotations AS

WITH unpacked AS(
SELECT
    o.id AS observation_id,
    UNNEST(o.annotations, RECURSIVE := true)
FROM staged.observations o

)

SELECT
    observation_id,
    {
        'phenology': list(controlled_value_id) FILTER (WHERE controlled_attribute_id = 12),
        'sex': list(controlled_value_id) FILTER (WHERE controlled_attribute_id = 9),
        'leaves': list(controlled_value_id) FILTER (WHERE controlled_attribute_id = 36)
    } AS attrs
FROM unpacked
GROUP BY observation_id;

CREATE OR REPLACE TABLE staged.label_corrected AS
SELECT
    t1.observation_id,
    list_transform(
        t1.label,
        (x, idx) -> COALESCE(t2.correction[idx], x)
    ) AS label
FROM staged.label t1
LEFT JOIN staged.corrections t2 ON t1.observation_id = t2.observation_id;

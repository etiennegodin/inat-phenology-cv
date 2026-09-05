-- Label re-derivation fix
CREATE OR REPLACE TABLE eda.label_update AS
SELECT
    t1.observation_id,
    t1.label AS original_label,
    t2.label AS updated_label
FROM main.cv_photos2 t1
JOIN staged.label t2 ON t1.observation_id = t2.observation_id
WHERE t1.label IS DISTINCT FROM t2.label;

-- Label corrections
CREATE OR REPLACE TABLE eda.label_corrections AS
SELECT
    t1.observation_id,
    t1.label AS original_label,
    t2.label AS updated_label
FROM staged.label t1
JOIN staged.label_corrected t2 ON t1.observation_id = t2.observation_id
WHERE t1.label IS DISTINCT FROM t2.label;

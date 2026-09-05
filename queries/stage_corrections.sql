CREATE OR REPLACE TABLE staged.corrections AS
SELECT
    COALESCE(A.observation_id, B.observation_id, C.observation_id) AS observation_id,
    [A.correction, B.correction, C.correction] AS correction

FROM corrections.flowering A
FULL OUTER JOIN corrections.fruiting B
    ON A.observation_id = B.observation_id
FULL OUTER JOIN corrections.flower_budding C
    ON COALESCE(A.observation_id, B.observation_id) = C.observation_id; -- Crucial: Joins C to whichever ID exists

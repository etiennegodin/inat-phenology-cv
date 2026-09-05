-- Import csvs
CREATE OR REPLACE TABLE corrections.flowering AS SELECT * FROM "/home/etienne/projects/inat-phenology-cv/data/label_corrections/val_label_correction - flowering.csv";
CREATE OR REPLACE TABLE corrections.fruiting AS SELECT * FROM "/home/etienne/projects/inat-phenology-cv/data/label_corrections/val_label_correction - fruiting.csv";
CREATE OR REPLACE TABLE corrections.flower_budding AS SELECT * FROM "/home/etienne/projects/inat-phenology-cv/data/label_corrections/val_label_correction - budding.csv";

-- Stage in one table
CREATE OR REPLACE TABLE staged.corrections AS
SELECT
    COALESCE(A.observation_id, B.observation_id, C.observation_id) AS observation_id,
    [A.correction, B.correction, C.correction] AS correction

FROM corrections.flowering A
FULL OUTER JOIN corrections.fruiting B
    ON A.observation_id = B.observation_id
FULL OUTER JOIN corrections.flower_budding C
    ON COALESCE(A.observation_id, B.observation_id) = C.observation_id; -- Crucial: Joins C to whichever ID exists

-- Create a corrected label table from the corrections
CREATE OR REPLACE TABLE staged.label_corrected AS
SELECT
    t1.observation_id,
    list_transform(
        t1.label,
        (x, idx) -> COALESCE(t2.correction[idx], x)
    ) AS label
FROM staged.label t1
LEFT JOIN staged.corrections t2 ON t1.observation_id = t2.observation_id;

-- Re-export to main_cv_photos

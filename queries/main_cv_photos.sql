-- Create the table for training
CREATE OR REPLACE TABLE main.cv_photos4 AS
SELECT p.observation_id,
    p.photo_id,
    l.label
FROM staged.photos p
JOIN staged.label_corrected l ON l.observation_id = p.observation_id

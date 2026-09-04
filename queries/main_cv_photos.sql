-- Create the table for training
CREATE OR REPLACE TABLE main.cv_photos3 AS

SELECT p.observation_id,
p.photo_id,
l.label
FROM staged.photos p
JOIN staged.label l ON l.observation_id = p.observation_id

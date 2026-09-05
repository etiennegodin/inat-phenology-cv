CREATE OR REPLACE VIEW eda.photo_counts_hist AS
WITH counts AS(
SELECT observation_id,
    COUNT(photo_id) as photo_count
FROM cv_photos
GROUP BY observation_id
)
SELECT * FROM histogram(counts, photo_count, bin_count := 10, technique := 'auto');

--
CREATE OR REPLACE VIEW eda.obs_dist AS
WITH classified AS(
SELECT
    CASE controlled_value_id
        WHEN 13 THEN 'Flowering'
        WHEN 14 THEN 'Fruiting'
        WHEN 15 THEN 'Flower Budding'
        WHEN 21 THEN 'No Evidence of Flowering'
    END AS classes,
    COUNT(DISTINCT(observation_id)) as obs_count,
    COUNT(photo_id) as photo_count

FROM cv_photos
GROUP BY controlled_value_id
)
SELECT *,
    ROUND(photo_count / obs_count,2) as mean_photo_per_obs
FROM classified;

--
CREATE OR REPLACE VIEW eda.photo_counts AS
WITH counts AS(
SELECT observation_id,
    COUNT(photo_id) as photo_count
FROM cv_photos
GROUP BY observation_id
)
SELECT
    AVG(photo_count) as avg_photos,
    STDDEV(photo_count) as std_photos,
    MEDIAN(photo_count) as median_photos,
    MIN(photo_count) as min_photos,
    MAX(photo_count) as max_photos
FROM counts

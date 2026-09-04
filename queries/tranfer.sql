ATTACH '/home/etienne/projects/inatML/data/inat_raw.duckdb' AS source_db (READ_ONLY);

CREATE OR REPLACE TABLE raw.ingested_photos AS
SELECT *
FROM source_db.raw.ingested_photos;

CREATE OR REPLACE TABLE staged.observations AS
SELECT *
FROM source_db.staged.observations
WHERE annotations IS NOT NULL; --skips without annotations

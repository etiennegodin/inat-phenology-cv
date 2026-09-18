CREATE SCHEMA IF NOT EXISTS "raw" ;

CREATE TABLE IF NOT EXISTS {table_name}
(
raw_id VARCHAR PRIMARY KEY,
raw_json JSON,
scraped_at VARCHAR,
scrapper_version VARCHAR,
);


CREATE TABLE IF NOT EXISTS staged.inat_requests (
    id INT PRIMARY KEY,
    uuid VARCHAR,
    photos STRUCT(id INT)[]
    )

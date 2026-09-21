CREATE SCHEMA IF NOT EXISTS "raw";
CREATE SCHEMA IF NOT EXISTS staged;


CREATE TABLE IF NOT EXISTS raw.obs_requests
(
raw_id VARCHAR PRIMARY KEY,
raw_json JSON,
scraped_at VARCHAR,
scrapper_version VARCHAR,
);

CREATE TABLE IF NOT EXISTS raw.img_requests
(
raw_id VARCHAR PRIMARY KEY,
raw_json JSON,
scraped_at VARCHAR,
scrapper_version VARCHAR,
);


CREATE TABLE IF NOT EXISTS staged.obs_requests (
    id INT PRIMARY KEY,
    uuid VARCHAR,
    photos STRUCT(id INT)[]
    );

CREATE TABLE IF NOT EXISTS staged.img_requests (
    photo_id INT PRIMARY KEY,
    observation_id INT,
    downloaded BOOLEAN
    );

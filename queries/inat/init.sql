CREATE SCHEMA IF NOT EXISTS "raw";
CREATE SCHEMA IF NOT EXISTS serving;


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


CREATE TABLE IF NOT EXISTS serving.observations (
    observation_id INT PRIMARY KEY,
    uuid VARCHAR,
    photos STRUCT(id INT)[],
    taxon INT,
    ancestor_ids INT[]
    );

CREATE TABLE IF NOT EXISTS serving.photos (
    photo_id INT PRIMARY KEY,
    observation_id INT,
    downloaded BOOLEAN
    );

CREATE TABLE IF NOT EXISTS serving.predictions (
    prediction_id VARCHAR PRIMARY KEY,
    observation_id INT,
    model_id INT,
    raw_preds FLOAT[],
    bin_preds INT[]
    );

CREATE TABLE IF NOT EXISTS serving.prediction_attention_weights (
    prediction_id VARCHAR,
    class_name VARCHAR,
    weights FLOAT[],
    );

CREATE TABLE IF NOT EXISTS serving.models (
    model_id INT PRIMARY KEY,
    class_name VARCHAR,
    model_name VARCHAR,
    model_version INT,
    model_uri VARCHAR,
    thresholds FLOAT[]
    );

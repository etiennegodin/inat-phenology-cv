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
    observation_id BIGINT PRIMARY KEY,
    uuid VARCHAR,
    photos STRUCT(id INT)[],
    taxon BIGINT,
    ancestor_ids INT[]
    );

CREATE TABLE IF NOT EXISTS serving.photos (
    photo_id BIGINT PRIMARY KEY,
    observation_id BIGINT,
    downloaded BOOLEAN
    );

CREATE TABLE IF NOT EXISTS serving.predictions (
    observation_id BIGINT NOT NULL,
    model_id INT NOT NULL,
    raw_preds FLOAT[],
    bin_preds INT[],
    predicted_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (observation_id, model_id)

    );

CREATE TABLE IF NOT EXISTS serving.prediction_attention_weights (
    observation_id BIGINT NOT NULL,
    model_id INTEGER NOT NULL,
    class_name VARCHAR NOT NULL,
    weights FLOAT[],
    predicted_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (observation_id, model_id, class_name)

    );

CREATE TABLE IF NOT EXISTS serving.models (
    model_id INT PRIMARY KEY,
    class_name VARCHAR,
    model_name VARCHAR,
    model_version INT,
    model_uri VARCHAR,
    thresholds FLOAT[]
    );

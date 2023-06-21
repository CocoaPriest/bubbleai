CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents
(
    id bigserial PRIMARY KEY,
    embedding vector(1536) NOT NULL,
    machine_id text COLLATE pg_catalog."default" NOT NULL,
    full_path text COLLATE pg_catalog."default" NOT NULL,
    text text COLLATE pg_catalog."default" NOT NULL
)
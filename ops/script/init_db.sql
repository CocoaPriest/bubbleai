CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (id bigserial PRIMARY KEY, embedding vector(1536));
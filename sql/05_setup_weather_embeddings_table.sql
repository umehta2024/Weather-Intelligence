-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Replace {{EMBEDDING_DIM}} with your model's dimension (e.g., 384 for all-MiniLM-L6-v2)

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the weather embeddings table
-- IMPORTANT: Replace {{EMBEDDING_DIM}} below with the correct dimension for your model:
--   - sentence-transformers/all-MiniLM-L6-v2: 384
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768
--   - BAAI/bge-large-en-v1.5: 1024
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Foreign key to weather_documents
    CONSTRAINT fk_weather_document
        FOREIGN KEY (document_id)
        REFERENCES weather_documents(id)
        ON DELETE CASCADE
);

-- Create HNSW index for fast cosine similarity search
-- HNSW is more accurate than IVFFlat and recommended for most use cases
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Create index on document_id for JOIN queries
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
ON weather_embeddings (document_id);

-- Create index on chunk_index for ordering within a document
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_chunk_index
ON weather_embeddings (document_id, chunk_index);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;

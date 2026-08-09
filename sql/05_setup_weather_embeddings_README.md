# Weather Embeddings Table Setup

Before running the `ingest_weather_embeddings.py` script, you must create the `weather_embeddings` table in your Lakebase Postgres database.

## Prerequisites

- Lakebase Postgres database is running
- pgvector extension is enabled
- weather_documents table exists

## Setup Instructions

1. Open `05_setup_weather_embeddings_table.sql`
2. Replace `{{EMBEDDING_DIM}}` with `384` (for all-MiniLM-L6-v2)
3. Run the SQL script in your Lakebase database

## What Gets Created

**Table**: `weather_embeddings`
- `id` (TEXT, PRIMARY KEY)
- `document_id` (TEXT, FK to weather_documents.id)
- `chunk_index` (INTEGER)
- `chunk_text` (TEXT)
- `embedding` (VECTOR(384))
- `model_name` (TEXT)
- `created_at` (TIMESTAMPTZ)

**Indexes**:
- HNSW index on embedding for vector search
- Index on document_id for JOINs
- Composite index on (document_id, chunk_index)

## Model Dimensions

- sentence-transformers/all-MiniLM-L6-v2: **384**
- sentence-transformers/all-mpnet-base-v2: 768
- BAAI/bge-small-en-v1.5: 384

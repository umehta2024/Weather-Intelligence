# Weather Intelligence - Embeddings Pipeline Guide

This guide explains how to set up and run the weather embeddings pipeline.

## Overview

The embeddings pipeline transforms weather documents (alerts and forecasts) into searchable vector embeddings for semantic search and RAG applications.

**Pipeline Flow:**
```
weather_documents (raw text)
    ↓ (chunk if needed)
    ↓ (embed with sentence-transformers)
    ↓
weather_embeddings (384-dim vectors)
```

## Prerequisites

1. ✅ Weather documents synced into `weather_documents` table
2. ✅ Lakebase database connection configured
3. ⏳ `weather_embeddings` table created (see Setup below)

## Setup (One-Time)

### Step 1: Create the Embeddings Table

Open `sql/05_setup_weather_embeddings_table.sql` and replace `{{EMBEDDING_DIM}}` with `384`:

```sql
-- Change this line:
embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,

-- To this:
embedding VECTOR(384) NOT NULL,
```

Then run the SQL script in your Lakebase database to create:
- `weather_embeddings` table with pgvector column
- HNSW index for fast cosine similarity search
- Indexes for JOIN performance

### Step 2: Verify Table Creation

```sql
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
```

You should see the `embedding` column with type `USER-DEFINED` and `udt_name` = `vector`.

## Running the Pipeline

### Option 1: As a Python Script

```bash
python notebooks/ingest_weather_embeddings.py
```

### Option 2: As a Databricks Notebook

1. Open the notebook in Databricks: `/Users/YOUR_EMAIL/Weather-Intelligence/notebooks/ingest_weather_embeddings.py`
2. Attach to a cluster with serverless compute
3. Run all cells

### What the Pipeline Does

1. **Loads unembedded documents** - Queries `weather_documents` for rows not yet in `weather_embeddings`
2. **Chunks long narratives** - Splits text > 800 chars into overlapping chunks (800 char chunks, 100 char overlap)
3. **Computes embeddings** - Uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dim vectors
4. **Writes to database** - Upserts via psycopg2 with `execute_values` for performance

### Expected Output

```
============================================================
Weather Intelligence - Embedding Pipeline
============================================================
Source table: weather_documents
Destination table: weather_embeddings
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
Chunking: size=800, overlap=100
============================================================
Connecting to Lakebase: xxx.cloud.databricks.com:5432/databricks_postgres

✅ Connection successful

Loading unembedded documents from weather_documents...
Found 42 unembedded documents

Chunking documents...
Created 68 chunks from 42 documents
  - Avg 1.6 chunks per document
  - Single-chunk documents: 35
  - Multi-chunk documents: 7

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
✅ Model loaded

Computing embeddings...
  Processed 68/68 chunks
✅ Computed 68 embeddings

Ensuring weather_embeddings table exists...
✅ Table weather_embeddings ready

Upserting 68 embeddings into weather_embeddings...
✅ Successfully upserted 68 embeddings
   Documents processed: 42
   Total chunks: 68

============================================================
✅ Weather embeddings pipeline completed successfully!
============================================================
```

## Querying Embeddings

### Find Similar Weather Events

```sql
-- Find top 5 weather documents similar to a query
WITH query_embedding AS (
    SELECT embedding 
    FROM weather_embeddings 
    WHERE id = 'some_document_id_0'
)
SELECT 
    we.document_id,
    wd.location,
    wd.event,
    wd.headline,
    we.chunk_text,
    1 - (we.embedding <=> qe.embedding) AS cosine_similarity
FROM weather_embeddings we
JOIN weather_documents wd ON we.document_id = wd.id
CROSS JOIN query_embedding qe
WHERE we.document_id != 'some_document_id'
ORDER BY we.embedding <=> qe.embedding
LIMIT 5;
```

### Search by Location

```sql
SELECT 
    we.document_id,
    wd.location,
    wd.event,
    we.chunk_text,
    wd.issued_at
FROM weather_embeddings we
JOIN weather_documents wd ON we.document_id = wd.id
WHERE wd.location ILIKE '%Chicago%'
ORDER BY wd.issued_at DESC
LIMIT 10;
```

## Scheduling

To keep embeddings up-to-date, schedule the pipeline to run after weather data sync:

1. Create a Databricks Job
2. Add task: Run `notebooks/ingest_weather_embeddings.py`
3. Set schedule: Run 30 minutes after `/weather/sync` completes
4. Or use: Triggered by table update on `weather_documents`

## Configuration

Edit these constants in `ingest_weather_embeddings.py` to customize:

```python
CHUNK_SIZE = 800          # Max characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384       # Model output dimension
```

**Note:** If you change the model or dimension, you must:
1. Update `EMBEDDING_DIM` in the script
2. Drop and recreate `weather_embeddings` table with new dimension
3. Re-run the pipeline to recompute all embeddings

## Troubleshooting

### "Table weather_embeddings does not exist"
→ Run the SQL setup script first (see Setup Step 1)

### "Connection refused" or "Connection timeout"
→ Check your Lakebase database secret is configured:
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
print("Secret exists:", secret is not None)
```

### "No new documents to embed"
→ All documents are already embedded. Add new weather data via `/weather/sync` first.

### "Model download failed"
→ The cluster needs internet access to download the model from HuggingFace Hub

## Architecture Notes

- **No Spark JDBC**: Uses psycopg2 directly (Spark JDBC doesn't work reliably with Lakebase)
- **Chunking Strategy**: Sliding window prevents cutting sentences mid-thought
- **Vector Index**: HNSW (Hierarchical Navigable Small World) for accuracy + speed
- **Idempotent**: Safe to re-run - uses `ON CONFLICT DO UPDATE` for deduplication
- **FK Constraint**: Embeddings cascade-delete when source documents are removed

## Performance

- **Small datasets** (<1000 docs): Runs in ~2-3 minutes
- **Medium datasets** (1K-10K docs): Runs in ~10-20 minutes  
- **Large datasets** (>10K docs): Consider batching or parallel processing

Bottlenecks:
1. Model loading (~30s first time, cached after)
2. Embedding computation (32 docs/batch, ~1 sec/batch)
3. Database insert (100 rows/batch, fast with execute_values)

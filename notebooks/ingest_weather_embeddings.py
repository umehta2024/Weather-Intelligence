# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Weather Intelligence** project.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads unembedded weather documents from the `weather_documents` table in Lakebase
# MAGIC 2. Chunks long weather narratives into overlapping text segments (CHUNK_SIZE=800, CHUNK_OVERLAP=100)
# MAGIC 3. Computes sentence embeddings for each chunk using sentence-transformers/all-MiniLM-L6-v2 (384-dim)
# MAGIC 4. Writes embeddings into the `weather_embeddings` table using the `pgvector` Postgres extension
# MAGIC    for fast cosine similarity search
# MAGIC
# MAGIC This enables semantic search over weather alerts and forecasts - e.g., "severe storms in the Midwest"
# MAGIC can retrieve relevant weather documents even when the exact phrasing differs.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that the Weather Intelligence Flask app uses.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text("weather_table_name", "weather_documents", "Source table (weather docs)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Narrative text chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Narrative text chunk overlap (chars)")

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Map embedding models to their output dimensions
if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
elif EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L12-v2":
    EMBEDDING_DIM = 384
elif EMBEDDING_MODEL_NAME == "sentence-transformers/all-mpnet-base-v2":
    EMBEDDING_DIM = 768
else:
    EMBEDDING_DIM = 384  # Default for most sentence-transformers models

print(f"\n{'='*60}")
print(f"Weather Intelligence - Embedding Pipeline")
print(f"{'='*60}")
print(f"Source table: {WEATHER_TABLE_NAME}")
print(f"Destination table: {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}-dim)")
print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg3
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# DBTITLE 1,Test Psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"User: {db_user}\n")

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10
    )
    cursor = conn.cursor()
    
    # Check weather_documents table
    cursor.execute(f"SELECT COUNT(*) FROM {WEATHER_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} documents in {WEATHER_TABLE_NAME}")
    
    # Show sample
    cursor.execute(f"SELECT id, location, source_type, event FROM {WEATHER_TABLE_NAME} LIMIT 3")
    rows = cursor.fetchall()
    print(f"\nSample weather documents:")
    for row in rows:
        print(f"  {row[0][:30]}... | {row[1]} | {row[2]} | {row[3]}")
    
    cursor.close()
    conn.close()
    print("\n✅ Lakebase connection working!")
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    traceback.print_exc()

# COMMAND ----------

# DBTITLE 1,Database Setup
# MAGIC %md
# MAGIC ## Database Setup
# MAGIC
# MAGIC This notebook expects:
# MAGIC * `weather_documents` table - created by app setup, contains weather alerts/forecasts
# MAGIC * `weather_embeddings` table - created below if needed, stores vector embeddings
# MAGIC
# MAGIC The notebook automatically creates the embeddings table with pgvector support if it doesn't exist.
# MAGIC All operations use psycopg2 with standard authentication.

# COMMAND ----------

# DBTITLE 1,Load unembedded weather documents
# MAGIC %md
# MAGIC ## Load unembedded weather documents
# MAGIC
# MAGIC Reads weather documents from the `weather_documents` table that don't yet have
# MAGIC embeddings. The query filters for documents with non-empty `narrative_text` and
# MAGIC excludes any that already have corresponding rows in `weather_embeddings`.
# MAGIC
# MAGIC This allows the pipeline to be run incrementally - only new weather documents
# MAGIC will be processed on each run.

# COMMAND ----------

# DBTITLE 1,Fetch news and sync using Lakebase SDK
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

print(f"Loading unembedded documents from {WEATHER_TABLE_NAME}...")

query = f"""
SELECT 
    id,
    location,
    source_type,
    headline,
    event,
    narrative_text,
    issued_at,
    effective_at,
    synced_at
FROM {WEATHER_TABLE_NAME}
WHERE narrative_text IS NOT NULL 
  AND narrative_text != ''
  AND id NOT IN (
    SELECT DISTINCT document_id 
    FROM {EMBEDDINGS_TABLE_NAME}
  )
ORDER BY synced_at DESC
"""

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require',
    cursor_factory=RealDictCursor,
)

try:
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    weather_df = pd.DataFrame(rows)
finally:
    cursor.close()
    conn.close()

if weather_df.empty:
    print("✅ No new documents to embed. All documents are up-to-date.")
    print("\nTo test the pipeline, sync some weather data via the app first!")
else:
    print(f"Found {len(weather_df)} unembedded documents\n")
    print("Sample:")
    display(weather_df[['id', 'location', 'source_type', 'event']].head(3))

# COMMAND ----------

# DBTITLE 1,Insert collected news articles using psycopg2
# Skip if no documents to process
if not weather_df.empty:
    print("Chunking weather narratives...")

    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # Stop if we've reached the end
            if end >= len(text):
                break
        
        return chunks

    # Build chunked documents DataFrame
    chunk_records = []
    
    for _, row in weather_df.iterrows():
        document_id = row["id"]
        narrative_text = row["narrative_text"]
        
        # Chunk the narrative text
        chunks = chunk_text(narrative_text)
        
        for chunk_index, text_chunk in enumerate(chunks):
            chunk_records.append({
                "document_id": document_id,
                "chunk_index": chunk_index,
                "chunk_text": text_chunk,
                "location": row["location"],
                "source_type": row["source_type"],
                "event": row["event"],
            })
    
    chunks_df = pd.DataFrame(chunk_records)
    print(f"Created {len(chunks_df)} chunks from {len(weather_df)} documents")
    print(f"  - Avg {len(chunks_df) / len(weather_df):.1f} chunks per document")
    print(f"  - Single-chunk: {(chunks_df.groupby('document_id').size() == 1).sum()}")
    print(f"  - Multi-chunk: {(chunks_df.groupby('document_id').size() > 1).sum()}\n")
else:
    print("Skipping chunking - no documents to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model and computes embeddings for each chunk.
# MAGIC Embeddings are computed in batches for memory efficiency.

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
# Skip if no chunks to process
if not weather_df.empty:
    import os
    from datetime import datetime
    from sentence_transformers import SentenceTransformer

    # Set up HuggingFace cache
    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
    print("✅ Model loaded\n")

    print("Computing embeddings...")
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(chunks_df), batch_size):
        batch = chunks_df.iloc[i : i + batch_size]
        vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())
        
        if (i + batch_size) % 128 == 0 or i + batch_size >= len(chunks_df):
            print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

    # Add embeddings to DataFrame
    chunks_df["embedding"] = all_embeddings
    chunks_df["model_name"] = EMBEDDING_MODEL_NAME
    chunks_df["created_at"] = datetime.now()

    print(f"\n✅ Computed {len(chunks_df)} embeddings\n")
else:
    print("Skipping embeddings - no documents to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure weather_embeddings table exists
# MAGIC
# MAGIC The table should already exist (created in the earlier notebook cells).
# MAGIC This cell creates it if needed, with pgvector support and HNSW indexes.

# COMMAND ----------

# DBTITLE 1,Ensure embeddings table exists
if not weather_df.empty:
    print(f"Ensuring {EMBEDDINGS_TABLE_NAME} table exists...")

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
        model_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        
        CONSTRAINT fk_weather_document
            FOREIGN KEY (document_id)
            REFERENCES {WEATHER_TABLE_NAME}(id)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_embedding
    ON {EMBEDDINGS_TABLE_NAME}
    USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_document_id
    ON {EMBEDDINGS_TABLE_NAME} (document_id);

    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_chunk_index
    ON {EMBEDDINGS_TABLE_NAME} (document_id, chunk_index);
    """

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
    )

    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"✅ Table {EMBEDDINGS_TABLE_NAME} ready\n")
    finally:
        cursor.close()
        conn.close()
else:
    print("Skipping table setup - no documents to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Uses psycopg2's `execute_values` for batch insert with ON CONFLICT.
# MAGIC Embeddings are cast to Postgres `vector` type via `::vector`.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
if not weather_df.empty:
    from psycopg2.extras import execute_values
    
    print(f"Upserting {len(chunks_df)} embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    # Prepare data for insertion
    # Generate ID as document_id_chunk_index
    chunks_df["id"] = chunks_df["document_id"] + "_" + chunks_df["chunk_index"].astype(str)
    
    insert_data = [
        (
            row["id"],
            row["document_id"],
            int(row["chunk_index"]),
            row["chunk_text"],
            row["embedding"],  # Pass as list - psycopg2 will handle conversion
            row["model_name"],
            row["created_at"],
        )
        for _, row in chunks_df.iterrows()
    ]
    
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
    )
    
    try:
        cursor = conn.cursor()
        
        # Use execute_values for batch insert with ON CONFLICT
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE
                SET chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    created_at = EXCLUDED.created_at
        """
        
        # Template with vector cast
        template = "(%s, %s, %s, %s, %s::vector, %s, %s)"
        
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        
        print(f"✅ Successfully upserted {inserted_count} embeddings")
        print(f"   Documents processed: {chunks_df['document_id'].nunique()}")
        print(f"   Total chunks: {len(chunks_df)}")
        
    finally:
        cursor.close()
        conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Weather embeddings pipeline completed successfully!")
    print("=" * 60)
else:
    print("No embeddings to write.")
"""
Ingest Weather Documents -> Vector Embeddings (Lakebase)

This script:
1. Reads unembedded rows from weather_documents table in Lakebase
2. Chunks narrative_text for long documents (sliding window: CHUNK_SIZE=800, CHUNK_OVERLAP=100)
3. Embeds each chunk using sentence-transformers/all-MiniLM-L6-v2 (384-dim)
4. Writes embeddings into weather_embeddings table via psycopg2

Run this as a Databricks notebook or standalone Python script.
"""

# MARK: Setup and Configuration
import base64
import json as _json
import os
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor, execute_values
from sentence_transformers import SentenceTransformer

# Configuration
WEATHER_TABLE_NAME = "weather_documents"
EMBEDDINGS_TABLE_NAME = "weather_embeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Lakebase connection config
LAKEBASE_SECRET_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
LAKEBASE_SECRET_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

print("=" * 60)
print("Weather Intelligence - Embedding Pipeline")
print("=" * 60)
print(f"Source table: {WEATHER_TABLE_NAME}")
print(f"Destination table: {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}-dim)")
print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
print("=" * 60)

# MARK: Resolve Lakebase Connection
w = WorkspaceClient()


def get_lakebase_url() -> str:
    """Fetch and decode the Lakebase connection URL from Databricks secrets."""
    secret = w.secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

print(f"Connecting to Lakebase: {db_host}:{db_port}/{db_name}")
print(f"User: {db_user}\n")

# MARK: Test Connection
print("Testing psycopg2 connection...")
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode="require",
        connect_timeout=10,
    )
    conn.close()
    print("✅ Connection successful\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# MARK: Load Unembedded Weather Documents
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
    sslmode="require",
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
    print("\nDone.")
    exit(0)

print(f"Found {len(weather_df)} unembedded documents\n")

# MARK: Chunk Documents
print("Chunking documents...")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
    
    for chunk_index, chunk_text in enumerate(chunks):
        chunk_records.append({
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "location": row["location"],
            "source_type": row["source_type"],
            "event": row["event"],
        })

chunks_df = pd.DataFrame(chunk_records)
print(f"Created {len(chunks_df)} chunks from {len(weather_df)} documents")
print(f"  - Avg {len(chunks_df) / len(weather_df):.1f} chunks per document")
print(f"  - Single-chunk documents: {(chunks_df.groupby('document_id').size() == 1).sum()}")
print(f"  - Multi-chunk documents: {(chunks_df.groupby('document_id').size() > 1).sum()}\n")

# MARK: Compute Embeddings
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

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

print(f"✅ Computed {len(chunks_df)} embeddings\n")

# MARK: Ensure Table Exists
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
    sslmode="require",
)

try:
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    conn.commit()
    print(f"✅ Table {EMBEDDINGS_TABLE_NAME} ready\n")
finally:
    cursor.close()
    conn.close()

# MARK: Upsert Embeddings
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
    sslmode="require",
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

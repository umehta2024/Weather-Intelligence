# Weather Intelligence - Semantic Search System

A Databricks-powered weather intelligence application that enables semantic search over weather alerts and forecasts using vector embeddings and pgvector.

## Architecture Overview

This system provides:
- **Data ingestion** from the National Weather Service API
- **Vector embeddings** for semantic search over weather narratives
- **REST API** for syncing data and querying via natural language
- **Lakebase (Postgres)** storage with pgvector for fast similarity search

## Data Source: National Weather Service (NWS) API

### Why NWS?

1. **Free & No Authentication** - The NWS API is completely free with no API keys required
2. **Comprehensive Coverage** - Provides both active alerts and detailed forecasts for all US locations
3. **Rich Narrative Text** - Weather descriptions are detailed and perfect for embedding-based search
4. **Reliable & Authoritative** - Official government weather data, updated continuously
5. **Well-Documented** - Clear API documentation at https://www.weather.gov/documentation/services-web-api

### Data Types Collected

**Weather Alerts:**
- Source: `/alerts/active` endpoint filtered by state
- Contains: event type, headline, description, instructions, severity, urgency
- Rich narrative text combining description + instructions

**Weather Forecasts:**
- Source: `/points/{lat},{lon}/forecast` endpoint
- Contains: 7-day detailed forecasts with temperature, wind, conditions
- Each period has both short and detailed forecast text

## Schema Decisions

### weather_documents Table

```sql
CREATE TABLE weather_documents (
    id TEXT PRIMARY KEY,              -- Alert ID or hash of location+time for forecasts
    location TEXT NOT NULL,           -- Human-readable location (e.g., "Chicago, IL")
    source_type TEXT NOT NULL,        -- "alert" or "forecast"
    headline TEXT,                    -- Brief summary or period name
    event TEXT,                       -- Alert event type or "Forecast"
    narrative_text TEXT NOT NULL,     -- Full text to embed (description + instructions)
    issued_at TIMESTAMPTZ,            -- When the alert/forecast was issued
    effective_at TIMESTAMPTZ,         -- When it takes effect
    payload JSONB NOT NULL,           -- Full raw API response for reference
    synced_at TIMESTAMPTZ DEFAULT now()
);
```

**Design Rationale:**
- `id`: Stable identifier - alert IDs from NWS, SHA256 hash for forecasts
- `narrative_text`: Combined description + instructions provide rich semantic content
- `payload`: Preserve full API response as JSONB for future analysis without re-fetching
- Separate `headline` and `event` fields enable filtering before semantic search
- Timestamps allow temporal filtering (e.g., "only active alerts")

### weather_embeddings Table

```sql
CREATE TABLE weather_embeddings (
    id TEXT PRIMARY KEY,                    -- {document_id}_{chunk_index}
    document_id TEXT NOT NULL,              -- FK to weather_documents
    chunk_index INTEGER NOT NULL,           -- Position of this chunk (0, 1, 2...)
    chunk_text TEXT NOT NULL,               -- The actual text chunk
    embedding VECTOR(384) NOT NULL,         -- 384-dim vector from all-MiniLM-L6-v2
    model_name TEXT NOT NULL,               -- Embedding model used
    created_at TIMESTAMPTZ DEFAULT now(),
    FOREIGN KEY (document_id) REFERENCES weather_documents(id) ON DELETE CASCADE
);

CREATE INDEX ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
```

**Design Rationale:**
- **Chunking**: Long weather narratives (especially alerts) are split into 800-char chunks with 100-char overlap
  - Ensures no single passage exceeds model context limits
  - Overlap prevents semantic breaks at chunk boundaries
  - Most forecasts are single-chunk; severe weather alerts often span 2-3 chunks
- **384-dim embeddings**: `sentence-transformers/all-MiniLM-L6-v2` chosen for:
  - Fast inference (can run on CPU)
  - Good semantic quality for short-to-medium text
  - Compact vector size (vs 768 or 1536 dims) = faster search
- **HNSW index**: Approximate nearest neighbor search scales to millions of vectors
- **Cosine similarity**: Standard for sentence embeddings, measures semantic similarity

## Pipeline: End-to-End

### 1. Sync Weather Data

**POST /weather/sync**

```bash
curl -X POST https://your-app.cloud.databricks.com/weather/sync \
  -H "Content-Type: application/json" \
  -d '{
    "locations": ["Chicago, IL", "Austin, TX", "Seattle, WA"],
    "limit": 50
  }'
```

**What it does:**
1. Geocodes each location using Nominatim (OpenStreetMap)
2. Extracts state code and fetches active alerts for that state
3. Fetches 7-day point forecast for the lat/lon
4. Normalizes both alerts and forecasts to the `weather_documents` schema
5. Upserts to Lakebase (ON CONFLICT updates existing records)

**Response:**
```json
{
  "synced": 42,
  "locations": ["Chicago, IL", "Austin, TX", "Seattle, WA"],
  "errors": []
}
```

### 2. Compute Embeddings

**Run the notebook:** `notebooks/ingest_weather_embeddings`

```python
# Option A: Run from Databricks UI
# Open the notebook and click "Run All"

# Option B: Schedule as a Databricks Job (recommended for production)
# The notebook is idempotent - it only processes new/unembedded documents
```

**What it does:**
1. Queries `weather_documents` for records not yet in `weather_embeddings`
2. Chunks `narrative_text` into 800-char segments with 100-char overlap
3. Loads `sentence-transformers/all-MiniLM-L6-v2` model
4. Encodes each chunk to a 384-dim vector
5. Creates `weather_embeddings` table if needed (with pgvector extension)
6. Upserts embeddings with HNSW index for fast cosine similarity search

**Configuration (via notebook widgets):**
- `weather_table_name`: Source table (default: `weather_documents`)
- `embeddings_table_name`: Destination table (default: `weather_embeddings`)
- `embedding_model`: Model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `chunk_size`: Max chars per chunk (default: 800)
- `chunk_overlap`: Overlap between chunks (default: 100)

### 3. Semantic Search

**POST /weather/search**

```bash
curl -X POST https://your-app.cloud.databricks.com/weather/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "risk of flooding near rivers",
    "top_k": 5
  }'
```

**What it does:**
1. Validates query string and top_k parameter (clamped to 1-20)
2. Embeds the query using the same `all-MiniLM-L6-v2` model (loaded at app startup)
3. Runs pgvector cosine similarity search against weather_embeddings
4. Returns matching documents with similarity scores (0-1, higher = more similar)

## Known Limitations & Future Improvements

### Current Limitations

1. **US-Only Coverage** - NWS API only covers United States and territories
2. **State-Level Alerts** - Alerts fetched by state, not precise lat/lon
3. **No Real-Time Updates** - Data pulled on-demand or via scheduled jobs
4. **Single Embedding Model** - Only supports all-MiniLM-L6-v2 (384-dim)
5. **No Hybrid Search** - Pure vector search only, no keyword/BM25 combination
6. **Fixed Chunking** - Character-based 800-char chunks may split sentences

### Potential Improvements

1. **Hybrid Search** - Combine vector similarity with keyword search and metadata filtering
2. **Better Chunking** - Sentence-aware chunking, token-based overlap
3. **Multi-Model Support** - Larger models (768/1536-dim) for better quality
4. **Real-Time Ingestion** - Webhooks or streaming for new alerts
5. **Geographic Precision** - Use NWS zone/county codes for precise filtering
6. **Query Understanding** - Handle temporal queries, expand with synonyms
7. **Caching** - Cache frequent queries and their embeddings
8. **Monitoring** - Track search latency, accuracy, embedding freshness
9. **Multi-Source Data** - Integrate OpenWeatherMap, global coverage
10. **Advanced Features** - Q&A, multi-turn search, alert summarization

## Tech Stack

* **Data Source:** National Weather Service API (free, no auth)
* **Database:** Databricks Lakebase (managed Postgres) with pgvector extension
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
* **Vector Search:** pgvector with HNSW index (cosine similarity)
* **API:** Flask (hosted as Databricks App)
* **ETL:** Databricks notebook with psycopg2
* **Geocoding:** Nominatim (OpenStreetMap, free)

## Files

* `app.py` - Flask API with `/weather/sync` and `/weather/search` endpoints
* `weather_client.py` - NWS API client (alerts, forecasts, normalization)
* `lakebase.py` - Lakebase connection helper (psycopg2 + SQLAlchemy)
* `notebooks/ingest_weather_embeddings` - Embedding ingestion pipeline
* `requirements.txt` - Python dependencies (includes sentence-transformers)
* `app.yaml` - Databricks App deployment config

"""
Databricks Weather Intelligence App:
- Serves a Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls weather data from the NWS API via weather_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json as _json
import logging
import os

from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from weather_client import WeatherClient, geocode_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)
_w = WorkspaceClient()

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")


def ensure_weather_table():
    """
    Create the weather documents table in Lakebase if it doesn't exist yet.
    This is the RAW document store that will later be used to compute
    vector embeddings.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL,
            headline TEXT,
            event TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI for the weather intelligence app."""
    return render_template("index.html")


@app.route("/weather/documents")
def list_weather_documents():
    """Read weather documents already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    location = request.args.get("location")
    source_type = request.args.get("source_type")
    
    where_clauses = []
    params = []
    
    if location:
        where_clauses.append("location ILIKE %s")
        params.append(f"%{location}%")
    
    if source_type:
        where_clauses.append("source_type = %s")
        params.append(source_type)
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    
    rows = lakebase.run_query(
        f"""
        SELECT id, location, source_type, headline, event, 
               narrative_text, issued_at, effective_at, synced_at 
        FROM {WEATHER_TABLE_NAME} 
        {where_sql}
        ORDER BY issued_at DESC NULLS LAST, synced_at DESC 
        LIMIT %s
        """,
        tuple(params),
    )
    return jsonify(rows)


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Pull weather alerts and forecasts from the NWS API and upsert them
    into the weather_documents table in Lakebase.

    Body (JSON): {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    
    - locations: List of human-readable location strings
    - limit: Max number of alerts to fetch per location (default: 50)
    
    Returns: {"synced": <count>, "locations": [...], "errors": [...]}
    """
    ensure_weather_table()
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations", [])
    limit = int(body.get("limit", 50))

    if not locations:
        return jsonify({"error": "No locations provided"}), 400

    total = 0
    errors = []
    processed_locations = []

    for location in locations:
        if not isinstance(location, str) or not location.strip():
            continue
            
        location = location.strip()
        logger.info(f"Syncing weather data for: {location}")
        
        try:
            # Geocode the location
            coords = geocode_location(location)
            if not coords:
                error_msg = f"Failed to geocode location: {location}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
            
            lat, lon = coords
            logger.info(f"Geocoded {location} to ({lat}, {lon})")
            
            # Fetch alerts (state-level, using first 2 chars of location if it looks like a state)
            alerts = []
            location_parts = location.split(",")
            if len(location_parts) >= 2:
                state_code = location_parts[-1].strip().upper()
                if len(state_code) == 2:
                    alerts = client.get_active_alerts(area=state_code, limit=limit)
                    logger.info(f"Fetched {len(alerts)} alerts for {state_code}")
            
            # Fetch forecast
            forecast_periods = client.get_forecast(lat, lon)
            logger.info(f"Fetched {len(forecast_periods)} forecast periods for {location}")
            
            # Normalize and upsert alerts
            alert_docs = [
                client.normalize_alert(alert, location) 
                for alert in alerts
            ]
            if alert_docs:
                total += _upsert_weather_batch(alert_docs)
            
            # Normalize and upsert forecast periods
            forecast_docs = [
                client.normalize_forecast_period(period, lat, lon, location)
                for period in forecast_periods
            ]
            if forecast_docs:
                total += _upsert_weather_batch(forecast_docs)
            
            processed_locations.append(location)
            
        except Exception as e:
            error_msg = f"Error processing {location}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
    
    response = {
        "synced": total,
        "locations": processed_locations,
    }
    if errors:
        response["errors"] = errors
    
    return jsonify(response)


def _upsert_weather_batch(documents: list[dict]) -> int:
    """
    Upsert a batch of weather documents into Lakebase.
    
    Args:
        documents: List of normalized weather document dicts
    
    Returns:
        Number of documents upserted
    """
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, source_type, headline, event,
                        narrative_text, issued_at, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            event = EXCLUDED.event,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc.get("id"),
                        doc.get("location"),
                        doc.get("source_type"),
                        doc.get("headline"),
                        doc.get("event"),
                        doc.get("narrative_text"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        _json.dumps(doc.get("payload", {})),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

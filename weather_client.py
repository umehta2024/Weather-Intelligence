"""
Client for the National Weather Service API (api.weather.gov).

The NWS API is free and requires no authentication. This client fetches:
- Active weather alerts
- Point forecasts (7-day detailed forecasts)

API documentation: https://www.weather.gov/documentation/services-web-api
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional, List, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30

# NWS API requires a User-Agent header identifying your application
_USER_AGENT = "WeatherIntelligenceApp/1.0 (Databricks)"


class WeatherClient:
    """Client for the National Weather Service API."""

    def __init__(self, base_url: str = _BASE_URL, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get_active_alerts(self, area: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Fetch active weather alerts.

        Args:
            area: Optional state code (e.g., "IL" for Illinois) or zone (e.g., "ILZ014")
            limit: Maximum number of alerts to return

        Returns:
            List of alert feature dicts, each with properties: id, event, headline,
            description, instruction, severity, urgency, areas, effective, expires, etc.
        """
        params: dict[str, Any] = {"limit": limit}
        if area:
            params["area"] = area

        try:
            resp = self._session.get(
                f"{self.base_url}/alerts/active",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("features", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch alerts: {e}")
            return []

    def get_point_metadata(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get grid metadata for a lat/lon point.

        Returns the forecast office, grid X/Y coordinates, and forecast/observation URLs.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/points/{lat:.4f},{lon:.4f}",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("properties", {})
        except requests.RequestException as e:
            logger.error(f"Failed to fetch point metadata for {lat},{lon}: {e}")
            return None

    def get_forecast(self, lat: float, lon: float) -> List[Dict]:
        """
        Get the 7-day forecast for a lat/lon point.

        Returns a list of forecast periods, each with: number, name, startTime,
        endTime, temperature, temperatureUnit, windSpeed, windDirection,
        shortForecast, detailedForecast.
        """
        # First, get the grid coordinates
        metadata = self.get_point_metadata(lat, lon)
        if not metadata:
            return []

        forecast_url = metadata.get("forecast")
        if not forecast_url:
            logger.error(f"No forecast URL in metadata for {lat},{lon}")
            return []

        try:
            resp = self._session.get(forecast_url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            properties = data.get("properties", {})
            return properties.get("periods", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch forecast from {forecast_url}: {e}")
            return []

    def normalize_alert(self, alert_feature: dict, location: str) -> dict:
        """
        Normalize an alert feature into a document record.

        Args:
            alert_feature: A feature dict from the /alerts endpoint
            location: Human-readable location string (e.g., "Chicago, IL")

        Returns:
            Normalized document with: id, location, source_type, headline,
            event, narrative_text, issued_at, effective_at, payload, synced_at
        """
        props = alert_feature.get("properties", {})
        alert_id = props.get("id") or alert_feature.get("id", "")

        # Combine description + instruction for the narrative text to embed
        description = props.get("description", "")
        instruction = props.get("instruction", "")
        narrative_parts = [p for p in [description, instruction] if p]
        narrative_text = "\n\n".join(narrative_parts)

        # Parse timestamps
        issued_at = props.get("sent")  # ISO 8601 string
        effective_at = props.get("effective")  # ISO 8601 string

        return {
            "id": alert_id,
            "location": location,
            "source_type": "alert",
            "headline": props.get("headline", ""),
            "event": props.get("event", ""),
            "narrative_text": narrative_text,
            "issued_at": issued_at,
            "effective_at": effective_at,
            "payload": alert_feature,
            "synced_at": datetime.utcnow().isoformat() + "Z",
        }

    def normalize_forecast_period(
        self, period: dict, lat: float, lon: float, location: str
    ) -> dict:
        """
        Normalize a forecast period into a document record.

        Args:
            period: A period dict from the forecast endpoint
            lat: Latitude
            lon: Longitude
            location: Human-readable location string

        Returns:
            Normalized document with: id, location, source_type, headline,
            event, narrative_text, issued_at, effective_at, payload, synced_at
        """
        # Generate a stable ID from location + startTime
        start_time = period.get("startTime", "")
        id_input = f"{location}:{start_time}"
        doc_id = hashlib.sha256(id_input.encode()).hexdigest()[:16]

        # Use the detailed forecast as the narrative text
        narrative_text = period.get("detailedForecast", "")

        # Headline: combine name + short forecast (e.g., "Tonight: Partly Cloudy")
        period_name = period.get("name", "")
        short_forecast = period.get("shortForecast", "")
        headline = f"{period_name}: {short_forecast}" if period_name else short_forecast

        return {
            "id": doc_id,
            "location": location,
            "source_type": "forecast",
            "headline": headline,
            "event": "Forecast",
            "narrative_text": narrative_text,
            "issued_at": start_time,
            "effective_at": start_time,
            "payload": period,
            "synced_at": datetime.utcnow().isoformat() + "Z",
        }


def geocode_location(location: str) -> Optional[Tuple[float, float]]:
    """
    Simple geocoding using Nominatim (OpenStreetMap) - free, no API key required.
    
    Args:
        location: Location string (e.g., "Chicago, IL" or "Austin, TX")
    
    Returns:
        (latitude, longitude) tuple, or None if geocoding fails
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": _USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        logger.error(f"Failed to geocode location '{location}': {e}")
    return None

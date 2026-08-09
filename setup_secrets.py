"""
One-time script to store the Lakebase connection URL in Databricks secrets.

Run this from a Databricks notebook to securely store your Lakebase credentials:
    
    %sh python setup_secrets.py

Or from a notebook terminal (if enabled on your cluster):

    python setup_secrets.py

The script prompts for your Lakebase connection URL and stores it as a base64-encoded
secret in the `database` scope with key `lakebase-url`. The weather app and embedding
notebook both read from this secret.
"""

import base64
import getpass

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

SCOPE = "database"
KEY = "lakebase-url"


def ensure_scope(scope: str):
    """Create the secret scope if it doesn't exist."""
    try:
        w.secrets.create_scope(scope=scope)
        print(f"✅ Created secret scope: {scope}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✅ Secret scope already exists: {scope}")
        else:
            raise


def store_secret(scope: str, key: str, value: str):
    """Store a secret value (base64 encoded)."""
    encoded_value = base64.b64encode(value.encode()).decode()
    w.secrets.put_secret(scope=scope, key=key, string_value=encoded_value)
    print(f"✅ Stored secret: {scope}/{key}")


def main():
    print("=" * 70)
    print("Weather Intelligence - Lakebase Secret Setup")
    print("=" * 70)
    print()
    print("This script stores your Lakebase connection URL as a Databricks secret.")
    print("The URL should look like:")
    print()
    print("  postgresql://role:password@host.cloud.databricks.com:5432/databricks_postgres?sslmode=require")
    print()
    print("You can find this URL in your Lakebase instance under 'Roles & Databases'")
    print("after creating a native password role.")
    print()
    
    # Ensure the scope exists
    ensure_scope(SCOPE)
    print()
    
    # Prompt for Lakebase URL
    print("Please enter your Lakebase connection URL:")
    print("(Input is hidden for security)")
    lakebase_url = getpass.getpass("Lakebase URL: ").strip()
    
    if not lakebase_url:
        print("❌ No URL provided. Exiting.")
        return
    
    if not lakebase_url.startswith("postgresql://"):
        print("⚠️  Warning: URL should start with 'postgresql://'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != "y":
            print("❌ Cancelled.")
            return
    
    # Store the secret
    store_secret(SCOPE, KEY, lakebase_url)
    
    print()
    print("=" * 70)
    print("✅ Setup complete!")
    print("=" * 70)
    print()
    print("Your Lakebase credentials are now stored securely.")
    print()
    print("Next steps:")
    print("  1. Deploy the Weather Intelligence app (it will use this secret)")
    print("  2. Sync weather data: POST /weather/sync")
    print("  3. Run the embedding notebook: notebooks/ingest_weather_embeddings")
    print("  4. Search weather: POST /weather/search")
    print()


if __name__ == "__main__":
    main()

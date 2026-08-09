"""
One-time setup script: creates the Databricks secret scope and stores the
Lakebase database URL. The weather API (api.weather.gov) requires no authentication.

Run this locally (with the Databricks CLI configured) or from a notebook.
Never commit the resulting secret value anywhere.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# Create the database scope (uncomment if it doesn't exist yet)
# w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: ")
)


w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)


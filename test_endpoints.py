#!/usr/bin/env python3
"""
Test script for Weather Intelligence API endpoints.
Usage:
    python test_endpoints.py <app_url>
    
Example:
    python test_endpoints.py https://your-app.cloud.databricks.com
"""

import json
import sys

import requests


def test_sync(base_url):
    """Test POST /weather/sync endpoint."""
    print("\n" + "="*60)
    print("TEST 1: POST /weather/sync")
    print("="*60)
    
    url = f"{base_url}/weather/sync"
    payload = {
        "locations": ["Chicago, IL", "Austin, TX"],
        "limit": 10
    }
    
    print(f"Sending POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        
        print(f"\n✅ Status: {resp.status_code}")
        print(f"Response:")
        print(json.dumps(result, indent=2))
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")
        return False


def test_search(base_url):
    """Test POST /weather/search endpoint."""
    print("\n" + "="*60)
    print("TEST 2: POST /weather/search")
    print("="*60)
    
    url = f"{base_url}/weather/search"
    payload = {
        "query": "tornado warnings",
        "top_k": 3
    }
    
    print(f"Sending POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        
        print(f"\n✅ Status: {resp.status_code}")
        print(f"Response:")
        print(json.dumps(result, indent=2))
        
        if result.get("results"):
            print(f"\n📊 Found {len(result['results'])} matches:")
            for i, item in enumerate(result["results"][:3], 1):
                print(f"  {i}. {item.get('location')}: {item.get('headline')} (sim: {item.get('similarity', 0):.3f})")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")
        return False


def test_list_documents(base_url):
    """Test GET /weather/documents endpoint."""
    print("\n" + "="*60)
    print("TEST 3: GET /weather/documents")
    print("="*60)
    
    url = f"{base_url}/weather/documents?limit=5"
    
    print(f"Sending GET {url}")
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        
        print(f"\n✅ Status: {resp.status_code}")
        print(f"Found {len(result)} documents")
        
        if result:
            print(f"\nSample document:")
            print(json.dumps(result[0], indent=2))
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_endpoints.py <app_url>")
        print("Example: python test_endpoints.py https://your-app.cloud.databricks.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip("/")
    
    print("="*60)
    print(f"Testing Weather Intelligence API: {base_url}")
    print("="*60)
    
    # Test health endpoint
    print("\nChecking health endpoint...")
    try:
        resp = requests.get(f"{base_url}/healthz", timeout=10)
        resp.raise_for_status()
        print(f"✅ App is healthy: {resp.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the app is deployed and running!")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("sync", test_sync(base_url)))
    
    print("\n⏳ Waiting 5 seconds for sync to complete...")
    import time
    time.sleep(5)
    
    results.append(("documents", test_list_documents(base_url)))
    results.append(("search", test_search(base_url)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed"))
    
    if not all_passed:
        print("\nNote: If search failed, make sure you ran the embeddings notebook first:")
        print("  notebooks/ingest_weather_embeddings")


if __name__ == "__main__":
    main()

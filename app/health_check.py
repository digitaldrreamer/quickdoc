#!/usr/bin/env python3
"""
Simple health check script for the document processor.
Can be used as an alternative to curl-based health checks.
"""

import sys
import requests
import time

def check_health(max_retries=3, delay=1):
    """Check if the service is healthy."""
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8002/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("Health check passed")
                    return True
        except Exception as e:
            print(f"Health check attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    
    print("Health check failed")
    return False

if __name__ == "__main__":
    success = check_health()
    sys.exit(0 if success else 1) 
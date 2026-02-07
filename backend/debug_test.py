import asyncio
import httpx
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

async def test_endpoint():
    # Test the specific endpoint that's causing issues
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "http://localhost:8000/api/chat/test_user/chat/history",
                headers={"X-User-ID": "test_user"},
                timeout=10.0
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error making request: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoint())
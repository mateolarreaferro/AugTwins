#!/usr/bin/env python3
"""
Quick test script to verify ElevenLabs API credentials work with both REST and WebSocket APIs.
"""
import asyncio
import json
import ssl
import requests
import websockets
from config import ELEVEN_API_KEY

# Test parameters
VOICE_ID = "5epn2vbuws8S5MRzxJH8"  # Same voice ID from logs
TEST_TEXT = "Hello, this is a test."

async def test_websocket_api():
    """Test WebSocket streaming API"""
    print("=== Testing WebSocket API ===")
    
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?output_format=pcm_22050"
    headers = {"xi-api-key": ELEVEN_API_KEY}
    
    try:
        print(f"Connecting to: {url}")
        
        # Create SSL context for macOS compatibility
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        websocket = await websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=10, ssl=ssl_context)
        print("✅ WebSocket connected successfully")
        
        # Send initialization without waiting for response
        config = {
            "text": " ",
            "voice_settings": {
                "speed": 1,
                "stability": 0.55,
                "similarity_boost": 0.8
            },
            "xi_api_key": ELEVEN_API_KEY
        }
        
        print("Sending config...")
        await websocket.send(json.dumps(config))
        
        # Send text
        text_message = {
            "text": TEST_TEXT,
            "try_trigger_generation": True
        }
        print(f"Sending text: '{TEST_TEXT}'")
        await websocket.send(json.dumps(text_message))
        
        # Send end signal
        await websocket.send(json.dumps({"text": ""}))
        print("Sent end-of-input signal")
        
        # Read responses
        chunk_count = 0
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                if isinstance(response, str):
                    data = json.loads(response)
                    if "audio" in data and data["audio"]:
                        chunk_count += 1
                        print(f"✅ Received audio chunk {chunk_count}")
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        break
                    if "error" in data:
                        print(f"❌ Error: {data['error']}")
                        return False
            except asyncio.TimeoutError:
                print("❌ Timeout - no more responses")
                break
            except websockets.exceptions.ConnectionClosed as e:
                print(f"WebSocket closed: {e}")
                break
        
        await websocket.close()
        print(f"WebSocket test completed - received {chunk_count} audio chunks")
        return chunk_count > 0
        
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_rest_api():
    """Test REST API"""
    print("\n=== Testing REST API ===")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    body = {
        "text": TEST_TEXT,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.8}
    }
    
    try:
        print(f"Making REST request to: {url}")
        response = requests.post(url, json=body, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ REST API success - received {len(response.content)} bytes of audio")
            return True
        else:
            print(f"❌ REST API failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ REST API test failed: {e}")
        return False

async def main():
    """Run both tests"""
    print(f"Testing with API key: {ELEVEN_API_KEY[:10]}...{ELEVEN_API_KEY[-10:]}")
    print(f"Testing with voice ID: {VOICE_ID}")
    print(f"Test text: '{TEST_TEXT}'")
    
    rest_success = test_rest_api()
    websocket_success = await test_websocket_api()
    
    print("\n=== Summary ===")
    print(f"REST API: {'✅ Working' if rest_success else '❌ Failed'}")
    print(f"WebSocket API: {'✅ Working' if websocket_success else '❌ Failed'}")
    
    if rest_success and not websocket_success:
        print("\n🔍 Diagnosis: API key and voice ID are valid, but WebSocket implementation has issues")
    elif not rest_success:
        print("\n🔍 Diagnosis: API key or voice ID are invalid")
    else:
        print("\n🔍 Both APIs working - issue is in integration layer")

if __name__ == "__main__":
    asyncio.run(main())
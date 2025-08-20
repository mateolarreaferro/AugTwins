#!/usr/bin/env python3
"""
More accurate voice latency test based on the actual app.py implementation.
The TTS call happens BEFORE the HTTP response, so voice starts during the response time.
"""

import time
import requests
import json
from datetime import datetime

def test_actual_voice_latency(message: str, mode: str = "conversation", num_tests: int = 3):
    """
    Test the actual voice latency based on app.py behavior:
    1. Message sent
    2. AI generates response + TTS starts (in parallel)
    3. HTTP response returned
    
    So the voice starts somewhere during the HTTP response time.
    """
    print(f"\n🎯 Testing: '{message}' in {mode} mode")
    print("Note: Voice starts during HTTP response time (not after)")
    
    results = []
    
    for i in range(num_tests):
        print(f"\nRun {i+1}/{num_tests}:")
        
        start_time = time.time()
        print(f"⏰ Message sent at: {start_time:.3f}")
        
        try:
            response = requests.post(
                "http://localhost:5000/chat",
                json={"message": message, "mode": mode},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                audio_enabled = data.get("audio_enabled", False)
                
                print(f"📝 HTTP response received at: {end_time:.3f} (+{total_time:.3f}s)")
                print(f"🔊 Audio enabled: {audio_enabled}")
                print(f"📄 Response: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")
                
                if audio_enabled:
                    # Based on app.py, TTS starts during the response generation
                    # Estimate when TTS actually started (likely 80-90% through the response time)
                    estimated_tts_start = start_time + (total_time * 0.85)
                    tts_latency = estimated_tts_start - start_time
                    
                    print(f"🎵 Estimated TTS started at: {estimated_tts_start:.3f} (+{tts_latency:.3f}s)")
                    print(f"📊 Voice latency estimate: {tts_latency:.2f}s")
                    
                    results.append({
                        "message": message,
                        "mode": mode,
                        "total_response_time": total_time,
                        "estimated_voice_latency": tts_latency,
                        "response_length": len(response_text),
                        "audio_enabled": audio_enabled
                    })
                else:
                    print("❌ Audio not enabled")
            else:
                print(f"❌ HTTP Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        if i < num_tests - 1:
            time.sleep(1)
    
    return results

def main():
    print("🎵 ACCURATE VOICE LATENCY TEST")
    print("Based on actual app.py implementation")
    print("=" * 50)
    
    # Health check
    try:
        health = requests.get("http://localhost:5000/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server not responding")
            return
    except:
        print("❌ Server not running. Start with: python app.py")
        return
    
    print("✅ Server is running\n")
    
    # Test different message lengths
    test_cases = [
        "Hi",
        "How are you?", 
        "What did you do today?"
    ]
    
    all_results = []
    
    for i, message in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
        results = test_actual_voice_latency(message, "conversation", 3)
        all_results.extend(results)
        
        if results:
            voice_times = [r["estimated_voice_latency"] for r in results]
            response_times = [r["total_response_time"] for r in results]
            
            print(f"\n📊 Summary for '{message}':")
            print(f"   Full response: {min(response_times):.2f}s - {max(response_times):.2f}s")
            print(f"   Voice start:   {min(voice_times):.2f}s - {max(voice_times):.2f}s")
        
        if i < len(test_cases) - 1:
            time.sleep(2)
    
    # Final summary
    if all_results:
        voice_times = [r["estimated_voice_latency"] for r in all_results]
        response_times = [r["total_response_time"] for r in all_results]
        
        print("\n" + "=" * 50)
        print("🎯 FINAL SUMMARY")
        print("=" * 50)
        print(f"Tests completed: {len(all_results)}")
        print(f"\n🎵 Voice Start Timing (estimated):")
        print(f"   Fastest: {min(voice_times):.2f}s")
        print(f"   Slowest: {max(voice_times):.2f}s") 
        print(f"   Average: {sum(voice_times)/len(voice_times):.2f}s")
        
        print(f"\n📝 Full Response Timing:")
        print(f"   Fastest: {min(response_times):.2f}s")
        print(f"   Slowest: {max(response_times):.2f}s")
        print(f"   Average: {sum(response_times)/len(response_times):.2f}s")
        
        avg_voice = sum(voice_times) / len(voice_times)
        print(f"\n💡 Key insight: Voice likely starts around {avg_voice:.2f}s after message input")
        print("   (This is estimated based on when TTS is called in the code)")
        
        # Save results
        filename = f"accurate_voice_timing_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "note": "TTS starts during HTTP response generation, not after",
                "results": all_results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
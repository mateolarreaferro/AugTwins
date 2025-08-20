#!/usr/bin/env python3
"""
Mode-specific performance test to compare conversation vs storytelling mode timing.
"""

import time
import requests
import json
import statistics
from datetime import datetime

def test_mode_performance(mode: str, message: str, num_tests: int = 3):
    """Test performance for a specific mode."""
    print(f"\n--- Testing {mode.upper()} mode ---")
    print(f"Message: '{message}'")
    
    times = []
    responses = []
    
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}...", end=" ")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:5000/chat",
                json={"message": message, "mode": mode},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                times.append(duration)
                responses.append({
                    "duration": duration,
                    "response_length": len(data.get("response", "")),
                    "response": data.get("response", "")[:100] + "..." if len(data.get("response", "")) > 100 else data.get("response", "")
                })
                print(f"{duration:.2f}s")
            else:
                print(f"FAILED ({response.status_code})")
                
        except Exception as e:
            print(f"ERROR: {e}")
        
        # Small delay between tests
        if i < num_tests - 1:
            time.sleep(1)
    
    if times:
        print(f"\nResults for {mode} mode:")
        print(f"  Tests: {len(times)}")
        print(f"  Min:   {min(times):.3f}s")
        print(f"  Max:   {max(times):.3f}s")
        print(f"  Mean:  {statistics.mean(times):.3f}s")
        print(f"  Median: {statistics.median(times):.3f}s")
        
        for i, resp in enumerate(responses):
            print(f"  Test {i+1}: {resp['duration']:.2f}s - {resp['response_length']} chars - \"{resp['response']}\"")
    
    return times

def main():
    print("AugTwins Mode Performance Comparison")
    print("Testing conversation vs storytelling mode timing")
    print("=" * 50)
    
    # Check server health
    try:
        health = requests.get("http://localhost:5000/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server not responding properly")
            return
    except:
        print("❌ Server not running. Start with: python app.py")
        return
    
    print("✅ Server is running")
    
    # Test messages
    conversation_msg = "How are you doing today?"
    storytelling_msg = "Tell me a story about your childhood"
    
    # Test conversation mode
    conv_times = test_mode_performance("conversation", conversation_msg, 3)
    
    # Wait between mode tests
    print("\nWaiting 2 seconds before next test...")
    time.sleep(2)
    
    # Test storytelling mode  
    story_times = test_mode_performance("storytelling", storytelling_msg, 3)
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    if conv_times and story_times:
        conv_avg = statistics.mean(conv_times)
        story_avg = statistics.mean(story_times)
        
        print(f"Conversation mode average: {conv_avg:.3f}s")
        print(f"Storytelling mode average: {story_avg:.3f}s")
        
        difference = abs(conv_avg - story_avg)
        if conv_avg < story_avg:
            print(f"✅ Conversation mode is {difference:.3f}s faster")
        elif story_avg < conv_avg:
            print(f"📚 Storytelling mode is {difference:.3f}s faster")
        else:
            print("⚖️  Both modes have similar performance")
        
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "conversation_mode": {
                "times": conv_times,
                "average": conv_avg,
                "message": conversation_msg
            },
            "storytelling_mode": {
                "times": story_times,
                "average": story_avg,
                "message": storytelling_msg
            }
        }
        
        filename = f"mode_comparison_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {filename}")
    
    print("\nNote: Conversation mode uses gpt-4o-mini with 150 token limit")
    print("      Storytelling mode uses gpt-5-mini with 2048 token limit")

if __name__ == "__main__":
    main()
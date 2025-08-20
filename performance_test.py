#!/usr/bin/env python3
"""
Performance test to measure the time from entering a message to playing the agent's voice.
This script measures the end-to-end latency of the AugTwins system.
"""

import time
import requests
import asyncio
import websockets
import json
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTest:
    def __init__(self, base_url: str = "http://localhost:5000", ws_url: str = "ws://localhost:5000"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.results: List[Dict[str, Any]] = []
        
    def test_health(self) -> bool:
        """Test if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def test_http_chat_latency(self, message: str = "Hello, how are you?") -> Dict[str, float]:
        """Test the HTTP chat endpoint latency."""
        logger.info(f"Testing HTTP chat with message: '{message}'")
        
        # Measure total time from request to response
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"message": message, "mode": "conversation"},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Response received: '{data.get('response', '')[:50]}...'")
                logger.info(f"HTTP Chat latency: {total_time:.3f}s")
                
                return {
                    "success": True,
                    "total_time": total_time,
                    "audio_enabled": data.get("audio_enabled", False),
                    "response_length": len(data.get("response", "")),
                    "message": message
                }
            else:
                logger.error(f"HTTP request failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"HTTP chat test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_websocket_tts_latency(self, message: str = "Hello, how are you?") -> Dict[str, float]:
        """Test WebSocket TTS streaming latency."""
        logger.info(f"Testing WebSocket TTS with message: '{message}'")
        
        try:
            uri = f"{self.ws_url.replace('ws://', 'ws://').replace('http://', 'ws://')}/socket.io/?EIO=4&transport=websocket"
            
            # Measure time from connection to first audio data
            start_time = time.time()
            first_audio_time = None
            audio_end_time = None
            
            async with websockets.connect(uri) as websocket:
                connection_time = time.time()
                
                # Send prompt
                prompt_data = json.dumps({
                    "type": "prompt",
                    "data": {
                        "text": message,
                        "id": f"test_{int(time.time())}"
                    }
                })
                await websocket.send(prompt_data)
                prompt_sent_time = time.time()
                
                # Listen for responses
                audio_chunks = 0
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        current_time = time.time()
                        
                        if isinstance(response, str):
                            data = json.loads(response)
                            event_type = data.get("type")
                            
                            if event_type == "audio_start":
                                first_audio_time = current_time
                                logger.info(f"First audio packet received at {first_audio_time - start_time:.3f}s")
                            elif event_type == "audio_data":
                                audio_chunks += 1
                            elif event_type == "audio_end":
                                audio_end_time = current_time
                                logger.info(f"Audio stream ended at {audio_end_time - start_time:.3f}s")
                                break
                            elif event_type == "error":
                                logger.error(f"WebSocket error: {data}")
                                break
                                
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket timeout waiting for response")
                        break
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        break
            
            total_time = audio_end_time - start_time if audio_end_time else None
            first_audio_latency = first_audio_time - start_time if first_audio_time else None
            
            return {
                "success": True,
                "connection_time": connection_time - start_time,
                "prompt_sent_time": prompt_sent_time - start_time,
                "first_audio_latency": first_audio_latency,
                "total_streaming_time": total_time,
                "audio_chunks": audio_chunks,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"WebSocket TTS test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_multiple_tests(self, num_tests: int = 5, test_messages: List[str] = None) -> Dict[str, Any]:
        """Run multiple performance tests and calculate statistics."""
        if test_messages is None:
            test_messages = [
                "Hello, how are you?",
                "Tell me about your day.",
                "What's your favorite hobby?",
                "Can you explain quantum physics?",
                "What's the weather like?"
            ]
        
        logger.info(f"Running {num_tests} performance tests...")
        
        http_times = []
        ws_first_audio_times = []
        ws_total_times = []
        
        for i in range(num_tests):
            message = test_messages[i % len(test_messages)]
            logger.info(f"\n--- Test {i+1}/{num_tests} ---")
            
            # Test HTTP chat
            http_result = self.test_http_chat_latency(message)
            if http_result.get("success"):
                http_times.append(http_result["total_time"])
                self.results.append({"test": i+1, "type": "http", **http_result})
            
            # Small delay between tests
            time.sleep(1)
            
            # Test WebSocket TTS (if available)
            try:
                ws_result = asyncio.run(self.test_websocket_tts_latency(message))
                if ws_result.get("success"):
                    if ws_result.get("first_audio_latency"):
                        ws_first_audio_times.append(ws_result["first_audio_latency"])
                    if ws_result.get("total_streaming_time"):
                        ws_total_times.append(ws_result["total_streaming_time"])
                    self.results.append({"test": i+1, "type": "websocket", **ws_result})
            except Exception as e:
                logger.warning(f"WebSocket test {i+1} skipped: {e}")
            
            # Delay between tests
            time.sleep(2)
        
        # Calculate statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": num_tests,
            "http_chat": self._calculate_stats(http_times, "HTTP Chat Response Time"),
            "websocket_first_audio": self._calculate_stats(ws_first_audio_times, "WebSocket First Audio Latency"),
            "websocket_total": self._calculate_stats(ws_total_times, "WebSocket Total Streaming Time"),
            "raw_results": self.results
        }
        
        return stats
    
    def _calculate_stats(self, times: List[float], label: str) -> Dict[str, float]:
        """Calculate statistics for a list of times."""
        if not times:
            return {"count": 0, "label": label}
        
        return {
            "count": len(times),
            "label": label,
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def print_results(self, stats: Dict[str, Any]):
        """Print formatted performance test results."""
        print("\n" + "="*60)
        print("AUGTWINS PERFORMANCE TEST RESULTS")
        print("="*60)
        print(f"Test completed: {stats['timestamp']}")
        print(f"Total tests run: {stats['total_tests']}")
        
        for test_type in ["http_chat", "websocket_first_audio", "websocket_total"]:
            data = stats[test_type]
            if data["count"] > 0:
                print(f"\n{data['label']}:")
                print(f"  Tests: {data['count']}")
                print(f"  Min:    {data['min']:.3f}s")
                print(f"  Max:    {data['max']:.3f}s")
                print(f"  Mean:   {data['mean']:.3f}s")
                print(f"  Median: {data['median']:.3f}s")
                if data['count'] > 1:
                    print(f"  Std Dev: {data['std_dev']:.3f}s")
        
        print("\n" + "="*60)
        
        # Key insights
        if stats["http_chat"]["count"] > 0:
            avg_response = stats["http_chat"]["mean"]
            print(f"Average message-to-response time: {avg_response:.3f}s")
        
        if stats["websocket_first_audio"]["count"] > 0:
            avg_audio = stats["websocket_first_audio"]["mean"]
            print(f"Average message-to-voice time: {avg_audio:.3f}s")
            
            if stats["http_chat"]["count"] > 0:
                improvement = avg_response - avg_audio
                if improvement > 0:
                    print(f"Voice streaming is {improvement:.3f}s faster than HTTP response")
                else:
                    print(f"HTTP response is {abs(improvement):.3f}s faster than voice streaming")

def main():
    """Run the performance test."""
    print("AugTwins Performance Test")
    print("This test measures the latency from message input to agent voice output.")
    print("-" * 60)
    
    tester = PerformanceTest()
    
    # Check if server is running
    if not tester.test_health():
        print("❌ Server is not running. Please start the AugTwins server with:")
        print("   python app.py")
        return
    
    print("✅ Server is running")
    
    # Run tests
    try:
        num_tests = 3  # Default to 3 tests for reasonable runtime
        results = tester.run_multiple_tests(num_tests)
        tester.print_results(results)
        
        # Save results to file
        results_file = f"performance_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
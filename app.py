"""Flask web application for AugTwins - connects backend with Unreal Engine."""
import os
# Set tokenizers environment variable before any imports that might use it
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import asyncio
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sock import Sock
import json
import threading
import time
from collections import defaultdict

# Agents
from agents.Lars.lars import lars

AGENTS = {
    "lars": lars,
}

# Initialize Flask app with WebSocket support
app = Flask(__name__)
CORS(app, 
     origins=["*"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"]
)  # Enable CORS for Unreal Engine integration
sock = Sock(app)

# Global state
current_agent = lars
conversation_history = []
tts_manager = None  # Will be initialized when needed
active_connections = {}  # Track active WebSocket connections
connection_flags = defaultdict(dict)  # Per-connection cancellation flags


def load_agent(agent) -> None:
    """Load agent - memories are now handled by individual agents"""
    # Force memory loading during startup to avoid first-response latency
    if hasattr(agent, '_ensure_memories_loaded'):
        print(f"[{agent.name}] Loading memories during startup...")
        agent._ensure_memories_loaded()


def save_conversation_history(agent, conversations: list) -> None:
    """Save conversation history to agent's directory with reflection."""
    agent_dir = Path(f"agents/{agent.name.title()}")
    history_file = agent_dir / "conversation_history.json"
    
    # Load existing history
    existing_history = []
    if history_file.exists():
        try:
            existing_history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            existing_history = []
    
    # Generate reflection on the conversation
    reflection = None
    if hasattr(agent, 'reflect_on_conversation') and conversations:
        print(f"[{agent.name}] Generating reflection on conversation...")
        try:
            reflection = agent.reflect_on_conversation(conversations)
        except Exception as e:
            print(f"[{agent.name}] Reflection failed: {e}")
    
    # Add new conversations with timestamp and reflection
    session = {
        "timestamp": datetime.now().isoformat(),
        "conversations": conversations,
        "reflection": reflection
    }
    existing_history.append(session)
    
    # Save updated history
    history_file.write_text(json.dumps(existing_history, indent=2, ensure_ascii=False))
    print(f"[Conversation history saved for {agent.name}]")
    
    if reflection and isinstance(reflection, dict):
        print(f"[{agent.name}] Reflection: {reflection.get('reflection', '')[:100]}...")
        if reflection.get('new_insights'):
            print(f"[{agent.name}] New insights: {', '.join(reflection['new_insights'][:2])}...")
        if reflection.get('topics_to_explore'):
            print(f"[{agent.name}] Topics to explore: {', '.join(reflection['topics_to_explore'][:2])}...")


def save_new_memories_to_mem0(agent, conversations: list) -> None:
    """Save new memories from conversation to Mem0.

    Sends the full conversation as role-tagged messages so Mem0 can
    perform its own memory extraction in a single request.
    """
    if not conversations:
        return
    
    print(f"[{agent.name}] Saving new memories to Mem0...")
    
    try:
        # Check if agent has Mem0 integration
        if hasattr(agent, '_get_mem0_client'):
            client = agent._get_mem0_client()
            if client:
                try:
                    messages = []
                    for conv in conversations:
                        if conv.get("user"):
                            messages.append({"role": "user", "content": conv.get("user", "")})
                        if conv.get("agent"):
                            messages.append({"role": "assistant", "content": conv.get("agent", "")})

                    metadata = {
                        "category": "conversation",
                        "source": "live_chat",
                        "timestamp": conversations[-1].get('timestamp', ''),
                    }

                    client.add(messages, user_id=agent.name.lower(), metadata=metadata)
                    print(f"[{agent.name}] Saved conversation with {len(messages)} messages to Mem0")
                except Exception as e:
                    print(f"[{agent.name}] Failed to save memories: {e}")
            else:
                print(f"[{agent.name}] Mem0 client not available")
        else:
            print(f"[{agent.name}] Agent doesn't support Mem0 integration")
            
    except Exception as e:
        print(f"[{agent.name}] Error saving memories to Mem0: {e}")


# Flask Routes

@app.route('/')
def index():
    """Serve the debugging frontend."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from frontend or Unreal Engine."""
    global current_agent, conversation_history
    
    data = request.get_json()
    message = data.get('message', '').strip()
    mode = data.get('mode', 'conversation')  # Default to conversation mode
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Generate response with mode
        reply = current_agent.generate_response(message, mode=mode)
        
        # Add to conversation history
        conversation_entry = {
            "user": message,
            "agent": reply,
            "timestamp": datetime.now().isoformat()
        }
        conversation_history.append(conversation_entry)
        
        # Text-to-speech (optional)
        audio_enabled = False
        try:
            current_agent.speak(reply)
            audio_enabled = True
        except Exception as e:
            print(f"TTS failed: {e}")
        
        return jsonify({
            'response': reply,
            'agent': current_agent.name,
            'timestamp': conversation_entry['timestamp'],
            'audio_enabled': audio_enabled
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch-agent', methods=['POST'])
def switch_agent():
    """Switch to a different agent."""
    global current_agent, conversation_history
    
    data = request.get_json()
    agent_name = data.get('agent', '').lower()
    
    if agent_name not in AGENTS:
        return jsonify({'error': f'Unknown agent: {agent_name}'}), 400
    
    # Save current conversation history
    if conversation_history:
        save_conversation_history(current_agent, conversation_history)
        conversation_history = []
    
    # Clear conversation context for the current agent
    current_agent.clear_context()
    
    # Switch agent
    current_agent = AGENTS[agent_name]
    load_agent(current_agent)
    
    return jsonify({
        'current_agent': current_agent.name,
        'message': f'Switched to {current_agent.name}'
    })

@app.route('/save-conversation', methods=['POST'])
def save_conversation():
    """Save current conversation history."""
    global conversation_history
    
    if not conversation_history:
        return jsonify({'message': 'No conversation to save'})
    
    try:
        save_conversation_history(current_agent, conversation_history)
        conversation_history = []
        return jsonify({'message': 'Conversation history saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-context', methods=['POST'])
def clear_context():
    """Clear the current agent's conversation context."""
    global current_agent
    
    try:
        current_agent.clear_context()
        return jsonify({
            'message': 'Conversation context cleared',
            'agent': current_agent.name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agents', methods=['GET'])
def get_agents():
    """Get list of available agents."""
    return jsonify({
        'agents': list(AGENTS.keys()),
        'current_agent': current_agent.name
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Unreal Engine integration."""
    return jsonify({
        'status': 'healthy',
        'current_agent': current_agent.name,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/unreal/tts', methods=['POST'])
def unreal_tts():
    """Direct TTS endpoint for Unreal Engine (HTTP-based alternative)."""
    global current_agent
    
    data = request.get_json()
    text = data.get('text', '').strip()
    voice_id = data.get('voice_id') or getattr(current_agent, 'tts_voice_id', '21m00Tcm4TlvDq8ikWAM')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # For Unreal Engine, provide WebSocket endpoint URL
        return jsonify({
            'status': 'accepted',
            'text': text,
            'voice_id': voice_id,
            'websocket_url': f'ws://{request.host}/ws',
            'sample_rate': 22050,
            'encoding': 'pcm_s16le',
            'channels': 1
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket handler for real-time TTS streaming
@sock.route('/ws')
def websocket_handler(ws):
    """Handle plain WebSocket connections for Unreal Engine."""
    connection_id = id(ws)
    active_connections[connection_id] = ws
    connection_flags[connection_id]['cancelled'] = False
    
    print(f"[WebSocket] Client connected: {connection_id}")
    
    try:
        while True:
            # Receive message from client
            message = ws.receive()
            
            if message is None:
                break
                
            print(f"[WebSocket] Received message from client {connection_id}: {message}")
            try:
                data = json.loads(message)
                message_type = data.get('type')
                print(f"[WebSocket] Message type: '{message_type}' from client {connection_id}")
                
                if message_type == 'prompt':
                    handle_websocket_prompt(ws, data, connection_id)
                elif message_type == 'cancel':
                    handle_websocket_cancel(ws, data, connection_id)
                else:
                    print(f"[WebSocket] Unknown message type '{message_type}' from client {connection_id}")
                    ws.send(json.dumps({
                        'type': 'error',
                        'error': f'Unknown message type: {message_type}'
                    }))
                    
            except json.JSONDecodeError:
                ws.send(json.dumps({
                    'type': 'error',
                    'error': 'Invalid JSON message'
                }))
                
    except Exception as e:
        print(f"[WebSocket] Error for client {connection_id}: {e}")
    finally:
        print(f"[WebSocket] Client disconnected: {connection_id}")
        # Clean up connection
        if connection_id in active_connections:
            del active_connections[connection_id]
        if connection_id in connection_flags:
            del connection_flags[connection_id]

def handle_websocket_prompt(ws, data, connection_id):
    """Handle text prompt for TTS streaming."""
    global current_agent, tts_manager
    
    print(f"[WebSocket TTS] Received prompt request from client {connection_id}")
    print(f"[WebSocket TTS] Data received: {data}")
    
    text = data.get('text', '').strip()
    print(f"[WebSocket TTS] Extracted text: '{text}' (length: {len(text)})")
    
    if not text:
        print(f"[WebSocket TTS] No text provided, sending error")
        ws.send(json.dumps({
            'type': 'error',
            'error': 'No text provided'
        }))
        return
    
    # Initialize TTS manager if needed
    if tts_manager is None:
        print(f"[WebSocket TTS] Initializing TTS manager")
        from core.tts_utils import get_tts_manager
        tts_manager = get_tts_manager()
    
    # Get agent's voice ID
    voice_id = getattr(current_agent, 'tts_voice_id', '21m00Tcm4TlvDq8ikWAM')
    job_id = data.get('id', f"job_{int(time.time() * 1000)}")
    
    print(f"[WebSocket TTS] Using voice_id: {voice_id}, job_id: {job_id}")
    print(f"[WebSocket TTS] Starting TTS processing for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Send audio_start JSON response
    ws.send(json.dumps({
        'type': 'audio_start',
        'id': job_id,
        'encoding': 'pcm_s16le',
        'sample_rate': 22050,
        'channels': 1
    }))
    
    def stream_audio():
        """Stream audio in background thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _stream():
                try:
                    chunk_index = 0
                    async for packet in tts_manager.stream_tts(text, voice_id, job_id):
                        # Check cancellation flag
                        if connection_flags[connection_id].get('cancelled', False):
                            break
                            
                        if packet['type'] == 'audio_data':
                            # Send binary PCM frames (4.4KB chunks for 100ms at 22.05kHz)
                            chunk_data = packet['data']
                            chunk_size = 4410  # ~100ms at 22.05kHz mono 16-bit
                            
                            for i in range(0, len(chunk_data), chunk_size):
                                if connection_flags[connection_id].get('cancelled', False):
                                    break
                                    
                                chunk = chunk_data[i:i + chunk_size]
                                if chunk and connection_id in active_connections:
                                    try:
                                        # Send raw binary PCM data
                                        active_connections[connection_id].send(chunk)
                                        chunk_index += 1
                                    except Exception as e:
                                        print(f"[WebSocket] Error sending audio chunk: {e}")
                                        break
                    
                    # Send audio_end JSON response
                    if not connection_flags[connection_id].get('cancelled', False) and connection_id in active_connections:
                        active_connections[connection_id].send(json.dumps({
                            'type': 'audio_end',
                            'id': job_id
                        }))
                        
                except Exception as e:
                    if connection_id in active_connections:
                        active_connections[connection_id].send(json.dumps({
                            'type': 'error',
                            'error': str(e)
                        }))
            
            loop.run_until_complete(_stream())
            loop.close()
            
        except Exception as e:
            print(f"[WebSocket] Stream error: {e}")
    
    # Start streaming in background thread
    thread = threading.Thread(target=stream_audio)
    thread.daemon = True
    thread.start()

def handle_websocket_cancel(ws, data, connection_id):
    """Handle cancellation of TTS streaming."""
    job_id = data.get('id')
    
    # Set cancellation flag for this connection
    connection_flags[connection_id]['cancelled'] = True
    
    # Send confirmation
    ws.send(json.dumps({
        'type': 'cancelled',
        'id': job_id
    }))

if __name__ == "__main__":
    # Initialize the default agent
    load_agent(current_agent)
    print(f"AugTwins Flask server starting with agent: {current_agent.name}")
    print("Debug interface available at: http://localhost:5000")
    print("API endpoints available for Unreal Engine integration")
    print("Plain WebSocket support enabled at: ws://localhost:5000/ws")
    print("WebSocket protocol: plain WebSocket (not Socket.IO)")
    print("Audio format: PCM 16-bit mono at 22.05kHz, 4.4KB chunks (100ms)")
    
    # Run Flask app with plain WebSocket support
    app.run(host='0.0.0.0', port=5000, debug=True)

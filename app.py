"""Flask web application for AugTwins - connects backend with Unreal Engine."""
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Agents
from agents.Lars.lars import lars

AGENTS = {
    "lars": lars,
}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Unreal Engine integration

# Global state
current_agent = lars
conversation_history = []


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
    """Save new memories from conversation to Mem0."""
    if not conversations:
        return
    
    print(f"[{agent.name}] Saving new memories to Mem0...")
    
    try:
        # Check if agent has Mem0 integration
        if hasattr(agent, '_get_mem0_client'):
            client = agent._get_mem0_client()
            if client:
                memory_count = 0
                for conv in conversations:
                    # Create memory from each conversation exchange
                    conversation_memory = f"User: {conv['user']}\n{agent.name}: {conv['agent']}"
                    
                    try:
                        messages = [{"role": "user", "content": conversation_memory}]
                        metadata = {
                            "type": "conversation",
                            "source": "live_chat",
                            "timestamp": conv.get('timestamp', ''),
                            "category": "conversation"
                        }
                        
                        client.add(messages, user_id=agent.name.lower(), metadata=metadata)
                        memory_count += 1
                        
                    except Exception as e:
                        print(f"[{agent.name}] Failed to save memory: {e}")
                
                print(f"[{agent.name}] Saved {memory_count}/{len(conversations)} new memories to Mem0")
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
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Generate response
        reply = current_agent.generate_response(message)
        
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

if __name__ == "__main__":
    # Initialize the default agent
    load_agent(current_agent)
    print(f"AugTwins Flask server starting with agent: {current_agent.name}")
    print("Debug interface available at: http://localhost:5000")
    print("API endpoints available for Unreal Engine integration")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

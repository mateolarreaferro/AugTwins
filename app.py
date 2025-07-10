"""Terminal chat launcher for AugTwins."""
import json
from datetime import datetime
from pathlib import Path

# Agents
from agents.Lars.lars import lars

AGENTS = {
    "lars": lars,
}


def load_agent(agent) -> None:
    """Load agent - memories are now handled by individual agents"""
    # Agent handles its own memory loading (Mem0 + local fallback)
    pass


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


def main() -> None:
    current = AGENTS["lars"]
    load_agent(current)
    conversation_history = []
    print(
        "\n=== Digital Twin Chat ===\n"
        "Commands: agent <name>  save  exit\n"
    )
    try:
        while True:
            msg = input(f"You → {current.name}: ").strip()
            if not msg:
                continue
            cmd = msg.lower()
            if cmd == "exit":
                break
            if cmd.startswith("agent "):
                new = cmd.split(maxsplit=1)[1]
                if new in AGENTS:
                    # Save conversation history before switching
                    if conversation_history:
                        save_conversation_history(current, conversation_history)
                        conversation_history = []
                    current = AGENTS[new]
                    load_agent(current)
                    print(f"[Switched to {current.name}]\n")
                else:
                    print("[Unknown agent]\n")
                continue
            if cmd == "save":
                if conversation_history:
                    save_conversation_history(current, conversation_history)
                    conversation_history = []
                print("[Conversation history saved]\n")
                continue

            reply = current.generate_response(msg)
            print(f"{current.name}: {reply}\n")
            
            # Add to conversation history
            conversation_history.append({
                "user": msg,
                "agent": reply,
                "timestamp": datetime.now().isoformat()
            })
            
            current.speak(reply)
    finally:
        # Save final conversation history
        if conversation_history:
            save_conversation_history(current, conversation_history)
            
            # Ask user if they want to save new memories to Mem0
            try:
                save_memories = input("\nWould you like to save the new memories from this conversation to Mem0? (y/n): ").strip().lower()
                if save_memories in ['y', 'yes']:
                    save_new_memories_to_mem0(current, conversation_history)
                else:
                    print("New memories not saved to Mem0 (conversation history still saved locally).")
            except (EOFError, KeyboardInterrupt):
                print("\nNew memories not saved to Mem0.")
        print("Goodbye!\n")

if __name__ == "__main__":
    main()

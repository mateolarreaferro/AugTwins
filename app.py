"""Terminal chat launcher for AugTwins."""
# Agents
from seeds.alex import alex
from seeds.nina import nina

from core.memory_utils import load_memories, save_memories, summarize_recent

# Reload persisted memories
alex.memory = load_memories("alex") or alex.memory
nina.memory = load_memories("nina") or nina.memory

def main() -> None:
    current = alex
    print("\n=== Digital Twin Chat ===\nCommands: switch  summary  save  exit\n")
    try:
        while True:
            msg = input(f"You → {current.name}: ").strip()
            cmd = msg.lower()
            if cmd == "exit":
                break
            if cmd == "switch":
                save_memories(current)
                current = nina if current is alex else alex
                print(f"[Switched to {current.name}]\n")
                continue
            if cmd == "save":
                save_memories(current)
                print("[Memories saved]\n")
                continue
            if cmd == "summary":
                print(f"\n{current.name} summary:\n{summarize_recent(current)}\n")
                continue

            # normal conversation
            reply = current.generate_response(msg)
            print(f"{current.name}: {reply}\n")
            current.speak(reply)
    finally:
        save_memories(current)
        print("Goodbye – memories stored.\n")

if __name__ == "__main__":
    main()
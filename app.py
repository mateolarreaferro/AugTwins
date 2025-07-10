"""Terminal chat launcher for AugTwins."""
from typing import Dict

from core import seed_db

# Agents
from seeds.lars import lars, SEED_MEMORIES as LARS_MEM

from core.memory_utils import load_memories, save_memories, summarize_recent

SEEDS: Dict[str, list[str]] = {
    "lars": LARS_MEM
}

AGENTS = {
    "lars": lars,
}


def load_agent(agent) -> None:
    mems = load_memories(agent.name)
    if mems:
        agent.memory = mems
        agent.rebuild_graph()
    else:
        seeds = seed_db.load_seed_memories(agent.name)
        if not seeds:
            seeds = SEEDS.get(agent.name.lower(), [])
        for txt in seeds:
            agent.add_memory(txt)


def main() -> None:
    seed_db.init_db()
    current = AGENTS["lars"]
    load_agent(current)
    print(
        "\n=== Digital Twin Chat ===\n"
        "Commands: agent <name>  summary  save  exit\n"
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
                    save_memories(current)
                    current = AGENTS[new]
                    load_agent(current)
                    print(f"[Switched to {current.name}]\n")
                else:
                    print("[Unknown agent]\n")
                continue
            if cmd == "save":
                save_memories(current)
                print("[Memories saved]\n")
                continue
            if cmd == "summary":
                print(f"\n{current.name} summary:\n{summarize_recent(current)}\n")
                continue

            reply = current.generate_response(msg)
            print(f"{current.name}: {reply}\n")
            current.speak(reply)
    finally:
        save_memories(current)
        print("Goodbye – memories stored.\n")

if __name__ == "__main__":
    main()

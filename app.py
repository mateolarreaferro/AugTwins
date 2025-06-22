"""Terminal chat launcher for AugTwins."""
from typing import Dict

# Agents
from seeds.yuvraj import yuvraj, SEED_MEMORIES as YUVRAJ_MEM
from seeds.dunya import dunya, SEED_MEMORIES as DUNYA_MEM
from seeds.lars import lars, SEED_MEMORIES as LARS_MEM
from seeds.anushhka import anushhka, SEED_MEMORIES as ANUSHHKA_MEM
from seeds.mateo import mateo, SEED_MEMORIES as MATEO_MEM
from seeds.alif import alif, SEED_MEMORIES as ALIF_MEM

from core.memory_utils import load_memories, save_memories, summarize_recent

SEEDS: Dict[str, Dict[str, list[str]]] = {
    "yuvraj": YUVRAJ_MEM,
    "dünya": DUNYA_MEM,
    "lars": LARS_MEM,
    "anushhka": ANUSHHKA_MEM,
    "mateo": MATEO_MEM,
    "alif": ALIF_MEM,
}

AGENTS = {
    "yuvraj": yuvraj,
    "dünya": dunya,
    "lars": lars,
    "anushhka": anushhka,
    "mateo": mateo,
    "alif": alif,
}


def load_for_mode(agent, mode: str) -> None:
    mems = load_memories(agent.name, mode=mode)
    if mems:
        agent.memory = mems
    else:
        for txt in SEEDS[agent.name.lower()].get(mode, []):
            agent.add_memory(txt)


def main() -> None:
    current = AGENTS["yuvraj"]
    mode = "combined"
    load_for_mode(current, mode)
    print(
        "\n=== Digital Twin Chat ===\n"
        "Commands: agent <name>  mode <interview|web|combined>  summary  save  exit\n"
    )
    try:
        while True:
            msg = input(f"You → {current.name} ({mode}): ").strip()
            if not msg:
                continue
            cmd = msg.lower()
            if cmd == "exit":
                break
            if cmd.startswith("agent "):
                new = cmd.split(maxsplit=1)[1]
                if new in AGENTS:
                    save_memories(current, mode=mode)
                    current = AGENTS[new]
                    load_for_mode(current, mode)
                    print(f"[Switched to {current.name}]\n")
                else:
                    print("[Unknown agent]\n")
                continue
            if cmd.startswith("mode "):
                new_mode = cmd.split(maxsplit=1)[1]
                if new_mode in {"interview", "web", "combined"}:
                    save_memories(current, mode=mode)
                    mode = new_mode
                    load_for_mode(current, mode)
                    print(f"[Memory mode: {mode}]\n")
                else:
                    print("[Invalid mode]\n")
                continue
            if cmd == "save":
                save_memories(current, mode=mode)
                print("[Memories saved]\n")
                continue
            if cmd == "summary":
                print(f"\n{current.name} summary:\n{summarize_recent(current)}\n")
                continue

            reply = current.generate_response(msg)
            print(f"{current.name}: {reply}\n")
            current.speak(reply)
    finally:
        save_memories(current, mode=mode)
        print("Goodbye – memories stored.\n")

if __name__ == "__main__":
    main()

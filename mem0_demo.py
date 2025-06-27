"""Demo script for Mem0-backed agents with dummy data."""

from core.agent import Agent
from core import mem0_backend


def create_agents() -> list[Agent]:
    """Return six demo agents."""
    names = ["Ada", "Ben", "Cora", "Dion", "Eve", "Finn"]
    agents = []
    for i, name in enumerate(names, 1):
        agents.append(
            Agent(
                name=name,
                personality=f"Demo agent {i}",
                tts_voice_id="",
            )
        )
    return agents


def main() -> None:
    agents = create_agents()
    # Add a couple of dummy memories for each agent
    for agent in agents:
        agent.add_memory(f"{agent.name} loves using Mem0 for storage.")
        agent.add_memory(f"Testing memory system for {agent.name}.")

    # Search each agent's memories using Mem0
    for agent in agents:
        res = mem0_backend.search_memory(agent.name, "Mem0")
        print(f"Search for '{agent.name}': {[r.get('text') for r in res]}")
        all_mems = mem0_backend.get_all_memories(agent.name)
        print(f"Stored memories for {agent.name}: {len(all_mems)}")


if __name__ == "__main__":
    main()

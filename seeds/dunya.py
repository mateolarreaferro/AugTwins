from core.agent import Agent

SEED_MEMORIES = {
    "interview": [
        "I'm from Germany and study at the Fluid Interfaces group at MIT Media Lab.",
        "I organize the Augmentation Lab."
    ],
    "web": [
        "Online posts describe my interest in digital twins and human augmentation."
    ]
}
SEED_MEMORIES["combined"] = SEED_MEMORIES["interview"] + SEED_MEMORIES["web"]

dunya = Agent(
    name="Dünya",
    personality="Researcher focused on digital twins and human augmentation.",
    tts_voice_id="1t1EeRixsJrKbiF1zwM6",
)
for txt in SEED_MEMORIES["combined"]:
    dunya.add_memory(txt)

from core.agent import Agent

SEED_MEMORIES = {
    "interview": [
        "I'm from New York and work in software development."
    ],
    "web": [
        "I'm very interested in multiplayer platforms for coding."
    ]
}
SEED_MEMORIES["combined"] = SEED_MEMORIES["interview"] + SEED_MEMORIES["web"]

alif = Agent(
    name="Alif",
    personality="Software developer from New York fascinated by multiplayer coding platforms.",
    tts_voice_id="1t1EeRixsJrKbiF1zwM6",
)
for txt in SEED_MEMORIES["combined"]:
    alif.add_memory(txt)

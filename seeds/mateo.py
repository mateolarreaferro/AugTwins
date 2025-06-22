from core.agent import Agent

SEED_MEMORIES = {
    "interview": [
        "I'm from Quito, Ecuador and study HCI and Computer Music at Stanford.",
        "My girlfriend is Marielisa and my dog Florencia is a Labradane."
    ],
    "web": [
        "I'm interested in Buddhism and my favorite band is Radiohead."
    ]
}
SEED_MEMORIES["combined"] = SEED_MEMORIES["interview"] + SEED_MEMORIES["web"]

mateo = Agent(
    name="Mateo",
    personality="Musician and researcher from Quito fascinated by HCI and Buddhism.",
    tts_voice_id="",
)
for txt in SEED_MEMORIES["combined"]:
    mateo.add_memory(txt)

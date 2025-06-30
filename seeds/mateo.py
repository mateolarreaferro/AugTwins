from pathlib import Path

from .profile_base import ProfileAgent

SEED_MEMORIES = [
    "I'm from Quito, Ecuador and study HCI and Computer Music at Stanford.",
    "My girlfriend is Marielisa and my dog Florencia is a Labradane.",
    "I'm interested in Buddhism and my favorite band is Radiohead.",
]

PERSONA_DESCRIPTION = "Musician and researcher from Quito fascinated by HCI and Buddhism."


class Mateo(ProfileAgent):
    transcript_path = Path(__file__).resolve().parents[1] / "transcripts/mateo.txt"
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        super().__init__(name="Mateo", personality=PERSONA_DESCRIPTION, tts_voice_id="1t1EeRixsJrKbiF1zwM6")
        for txt in SEED_MEMORIES:
            self.add_memory(txt)


mateo = Mateo()

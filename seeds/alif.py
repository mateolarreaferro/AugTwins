from pathlib import Path

from .profile_base import ProfileAgent

SEED_MEMORIES = [
    "I'm from New York and work in software development.",
    "I'm very interested in multiplayer platforms for coding.",
]

PERSONA_DESCRIPTION = "Software developer from New York fascinated by multiplayer coding platforms."


class Alif(ProfileAgent):
    transcript_path = Path(__file__).resolve().parents[1] / "transcripts/alif.txt"
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        super().__init__(name="Alif", personality=PERSONA_DESCRIPTION, tts_voice_id="1t1EeRixsJrKbiF1zwM6")
        for txt in SEED_MEMORIES:
            self.add_memory(txt)


alif = Alif()

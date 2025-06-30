from pathlib import Path

from .profile_base import ProfileAgent

SEED_MEMORIES = [
    "I'm from Germany and study at the Fluid Interfaces group at MIT Media Lab.",
    "I organize the Augmentation Lab.",
    "Online posts describe my interest in digital twins and human augmentation.",
]

PERSONA_DESCRIPTION = "Researcher focused on digital twins and human augmentation."


class Dunya(ProfileAgent):
    transcript_path = Path(__file__).resolve().parents[1] / "transcripts/dunya.txt"
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        super().__init__(name="Dünya", personality=PERSONA_DESCRIPTION, tts_voice_id="1t1EeRixsJrKbiF1zwM6")
        for txt in SEED_MEMORIES:
            self.add_memory(txt)


dunya = Dunya()

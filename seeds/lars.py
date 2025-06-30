from pathlib import Path

from .profile_base import ProfileAgent

SEED_MEMORIES = [
    "I'm from Germany and Virginia and study immersive experiences at SCAD.",
    "My girlfriend is Anushhka and we have a dragon pet named Sara.",
    "I love music from the '60s and '70s.",
]

PERSONA_DESCRIPTION = "Immersive experiences student who loves classic music."


class Lars(ProfileAgent):
    transcript_path = Path(__file__).resolve().parents[1] / "transcripts/lars.txt"
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        super().__init__(name="Lars", personality=PERSONA_DESCRIPTION, tts_voice_id="5epn2vbuws8S5MRzxJH8")
        for txt in SEED_MEMORIES:
            self.add_memory(txt)


lars = Lars()

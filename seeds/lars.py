from pathlib import Path
import json

from .profile_base import ProfileAgent

BASE = Path(__file__).resolve().parents[1]
MEM_PATH = BASE / "interviews/lars/memories.json"
PERSONA_PATH = BASE / "interviews/lars/persona.json"
UTTERANCE_PATH = BASE / "transcripts/lars.json"

if PERSONA_PATH.exists():
    _persona_data = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
else:
    _persona_data = {"description": "", "personality_type": ""}

PERSONA_DESCRIPTION = _persona_data.get("description", "")
PERSONALITY_TYPE = _persona_data.get("personality_type", "")


class Lars(ProfileAgent):
    transcript_path = UTTERANCE_PATH
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        full_personality = f"{PERSONA_DESCRIPTION} ({PERSONALITY_TYPE})".strip()
        super().__init__(name="Lars", personality=full_personality, tts_voice_id="5epn2vbuws8S5MRzxJH8")
        if MEM_PATH.exists():
            mem_list = json.loads(MEM_PATH.read_text(encoding="utf-8"))
            for m in mem_list:
                txt = m.get("memory") if isinstance(m, dict) else None
                if txt:
                    self.add_memory(txt)


lars = Lars()

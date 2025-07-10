# Aug Lab Digital Twins

## Setup

Create a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

Running `app.py` requires the packages in `requirements.txt`.  The tests stub out
heavy dependencies so they can run without network access.

## Interviews and profiles

Raw interview transcripts should live under `interviews/<person>/transcripts/`.
You can generate these text files from audio using `interviews/transcribe.py`.

Once you have the `.txt` transcripts, run `python generate_profile.py <person>`
from the project root. This will create three artefacts:

1. `agents/<Person>/memories.json` – structured memories extracted from the
   transcripts (in first person perspective).
2. `agents/<Person>/persona.json` – a persona description and inferred
   personality type (in first person perspective).
3. `agents/<Person>/utterance.json` – an utterance style guide referenced by the
   agent profile (see `agents/Lars/lars.py`). The `agents/` directory is created
   automatically if it does not exist.

## Mem0 Integration

To upload the generated profile to Mem0 memory system, use the `--upload-mem0` flag:

```bash
python generate_profile.py lars --upload-mem0
```

Optional arguments:
- `--mem0-user-id <id>`: Specify a custom user ID for Mem0 (defaults to person name)
- `--verbose`: Enable verbose logging to see upload progress

Requirements:
- Mem0 API key in `settings.py` or environment variable `MEM0_API_KEY`
- Install mem0ai: `pip install mem0ai` (already in requirements.txt)


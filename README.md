# AugTwins - Digital Twin Chat System

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `settings.py` with your API keys:
```python
OPENAI_API_KEY = "your-openai-api-key"
ELEVEN_API_KEY = "your-elevenlabs-api-key"

# Mem0 Pro Configuration (optional)
MEM0_API_KEY = "your-mem0-api-key"
MEM0_ORG_ID = "your-mem0-org-id"
MEM0_PROJECT_ID = "your-mem0-project-id"
```

## Running the Application

Start the chat interface:
```bash
python app.py
```

Commands during chat:
- Type normally to chat with the current agent
- `agent <name>` - Switch to a different agent
- `save` - Save conversation history
- `exit` - Quit

## Creating Digital Twins from Interviews

1. **Transcribe audio interviews:**
   Place audio files in `interviews/<person>/transcripts/` and run:
   ```bash
   python interviews/transcribe.py
   ```

2. **Generate agent profile:**
   ```bash
   python generate_profile.py <person>
   ```
   This creates three JSON files in `agents/<Person>/`:
   - `memories.json` - Extracted personal memories
   - `persona.json` - Personality profile
   - `utterance.json` - Speech patterns and style

3. **Upload to Mem0 (optional):**
   ```bash
   python generate_profile.py <person> --upload-mem0
   ```


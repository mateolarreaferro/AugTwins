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

3. Configure API keys:
   Copy the example environment file and add your API keys:

    ```bash
    cp .env.example .env
    ```

    Edit `.env` and add your API keys:

    ```bash
    # OpenAI API Configuration
    OPENAI_API_KEY=your_openai_api_key_here

    # ElevenLabs API Configuration (for text-to-speech)
    ELEVEN_API_KEY=your_elevenlabs_api_key_here

    # Mem0 Pro Configuration (optional - for advanced memory features)
    MEM0_API_KEY=your_mem0_api_key_here
    MEM0_ORG_ID=your_mem0_org_id_here
    MEM0_PROJECT_ID=your_mem0_project_id_here
    ```

## Running the Application

Start the chat interface:

```bash
python app.py
```

Commands during chat:

-   Type normally to chat with the current agent
-   `agent <name>` - Switch to a different agent
-   `save` - Save conversation history
-   `exit` - Quit

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

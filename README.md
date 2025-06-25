# Aug Lab Digital Twins

## Setup

Create a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

Running `app.py` requires the packages in `requirements.txt`.  The tests stub out
heavy dependencies so they can run without network access.

## Transcripts and utterances

Each agent can mimic a specific speaking style.  The `transcripts/` directory
contains short example snippets used by `core.utterance_utils` to guide the LLM.
`Agent.generate_response` now calls this helper with a slightly higher
temperature (0.5) so replies sound more natural and less robotic.

## Mem0 integration

The app can optionally sync memories to [Mem0](https://mem0.ai).  To enable this
feature, set the environment variable `MEM0_API_KEY` to a valid API key.  If the
remote service or agent does not exist, the code falls back to local JSON files
in the `memories/` directory.  `404` errors during startup are therefore
harmless and usually mean the remote agent or mode hasn’t been created yet.

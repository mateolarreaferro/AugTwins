# Aug Lab Digital Twins

## Setup

Create a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

Running `app.py` requires the packages in `requirements.txt`.  The tests stub out
heavy dependencies so they can run without network access.

## Mem0 integration

The app can optionally sync memories to [Mem0](https://mem0.ai).  To enable this
feature, set the environment variable `MEM0_API_KEY` to a valid API key.  If the
remote service or agent does not exist, the code falls back to local JSON files
in the `memories/` directory.  `404` errors during startup are therefore
harmless and usually mean the remote agent or mode hasn’t been created yet.

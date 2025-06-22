# Aug Lab Digital Twins

This project now supports the **Mem0** memory service for storing and retrieving
agent memories. If a `MEM0_API_KEY` (or `settings.MEM0_API_KEY`) is provided,
memory operations such as saving, loading, searching and summarising will be
performed through Mem0's API. Without the key the previous file-based fallback
is used.

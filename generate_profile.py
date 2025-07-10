#!/usr/bin/env python3
"""
Generate structured memories, a detailed persona description, and an utterance style guide
from a speaker's interview transcripts.

The script walks the directory tree that looks like this (example):

AugTwins/
├── interviews/
│   ├── lars/
│   │   ├── transcripts/
│   │   │   ├── interview1.mp3.txt
│   │   │   └── interview2.wav.txt
│   │   └── ...
└── interviews/generate_profile.py  <‑‑ this file

Run from the project root:

    python interviews/generate_profile.py lars

Output:
    AugTwins/interviews/lars/profile.json

Main improvements compared with the earlier version
---------------------------------------------------
* **Security**
  * The OpenAI key *must* come from the environment or a `.env` file – no more  hard‑coding.
  * Removes silent fallback behaviour that could accidentally leak keys.
* **Reliability & Efficiency**
  * Robust exponential‑back‑off retry wrapper for *all* OpenAI calls (network, rate‑limit).
  * Token‑aware text chunker prevents overrunning the model context size.
  * Deduplication logic normalises whitespace & case.
  * Only the minimal text needed for persona/utterance prompts is sent (respecting token limit).
* **Usability**
  * CLI with `argparse` (adds `--model`, `--max‑tokens`, `--verbose`).
  * Structured logging instead of bare `print`.
  * Optional progress bars when `tqdm` is available.

Dependencies
------------
```
pip install --upgrade openai tiktoken tenacity python‑dotenv tqdm
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore

# Optional (soft) deps 
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

# third‑party hard deps 
import tiktoken  # type: ignore
from openai import OpenAI, OpenAIError, RateLimitError  # type: ignore


DEFAULT_MODEL = "gpt-4o-mini"
MAX_MODEL_TOKENS = 8_192  # soft cap we respect regardless of model
DEFAULT_CHUNK_TOKENS = 6_000  # leave headroom for instructions & response
ENCODING = tiktoken.encoding_for_model(DEFAULT_MODEL)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# System prompts
MEMORY_PROMPT_SYSTEM = (
    "You are a knowledge engineer. Extract atomic factual memories about the speaker. "
    "Each memory must be a single declarative sentence about a stable fact, preference, skill, event, or belief. "
    "Return ONLY valid JSON in the following schema: "
    "[{\"memory\": <string>, \"type\": <'biographical'|'preference'|'skill'|'belief'|'event'>, \"tags\": [<string>, ...]}]"
)

PERSONA_PROMPT = (
    "You are an expert biographer. Based on the complete interview transcripts, "
    "write an elaborate persona description capturing background, personality traits, interests, values, and communication style."
)

UTTERANCE_PROMPT = (
    "Analyze the speaker's dialogue to build an utterance guide that another AI can use to mimic their speech patterns. "
    "Include:\n- overall style summary (tone, vocabulary, formality)\n- common phrases or exclamations (5‑10)\n- guidance on prosody/pacing\n- filler words or quirks\nReturn this guide as valid JSON: "
    "{\"style_guide\": <string>, \"sample_phrases\": [<string>, ...]}"
)


def count_tokens(text: str) -> int:
    """Return the approximate token count for `text`."""
    return len(ENCODING.encode(text))


def chunk_text(text: str, max_tokens: int) -> List[str]:
    """Greedy sentence‑based splitter that respects a token budget."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    chunks: List[str] = []
    current: List[str] = []
    tokens_so_far = 0

    for sentence in sentences:
        t = count_tokens(sentence)
        if tokens_so_far + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, tokens_so_far = [sentence], t
        else:
            current.append(sentence)
            tokens_so_far += t

    if current:
        chunks.append(" ".join(current))
    return chunks


@retry(
    stop=stop_after_attempt(4),  # 1 original + 3 retries
    wait=wait_exponential(multiplier=2, min=1, max=10),
    reraise=True,
)
def chat(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    """Wrapper with robust exponential back‑off."""
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content  # type: ignore[index]
    except RateLimitError as e:
        logging.warning("Rate‑limited: %s", e)
        raise  # handled by tenacity
    except OpenAIError as e:  # pragma: no cover
        logging.error("OpenAI API error: %s", e)
        raise  # bubbled after retries


def extract_memories(client: OpenAI, model: str, transcript: str, chunk_tokens: int) -> List[Dict[str, Any]]:
    """Run the MEMORY prompt over transcript chunks and collect unique memories."""
    chunks = chunk_text(transcript, chunk_tokens)
    iterator = tqdm(chunks, desc="Memories") if tqdm else chunks
    all_memories: List[Dict[str, Any]] = []

    for chunk in iterator:
        raw = chat(
            client,
            model,
            [
                {"role": "system", "content": MEMORY_PROMPT_SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        try:
            batch = json.loads(raw)
            if isinstance(batch, list):
                all_memories.extend(batch)
            else:
                logging.warning("Unexpected memory JSON shape; skipping batch.")
        except json.JSONDecodeError:
            logging.warning("Failed to parse memories JSON; skipping batch.")

    # Deduplicate (case‑folded, stripped)
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for m in all_memories:
        text = str(m.get("memory", "")).strip()
        key = re.sub(r"\s+", " ", text).lower()
        if text and key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    """If text is too long, keep the last `max_tokens` worth of tokens."""
    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODING.decode(tokens[-max_tokens:])



def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.critical("OPENAI_API_KEY not set (env or .env)")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate a Mem0‑ready profile JSON from transcripts.")
    parser.add_argument("person", help="Subfolder name under interviews/<person>/transcripts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model, default: %(default)s")
    parser.add_argument("--max‑tokens", type=int, default=DEFAULT_CHUNK_TOKENS, help="Token budget per transcript chunk")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase log verbosity (-v or -vv)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent  # hop out of interviews/
    transcripts_dir = project_root / "interviews" / args.person / "transcripts"
    if not transcripts_dir.exists():
        logging.critical("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        logging.critical("No .txt transcripts found in %s", transcripts_dir)
        sys.exit(1)

    full_transcript = "\n".join(p.read_text(encoding="utf‑8") for p in txt_files)

    client = build_openai_client()

    # Step 1: Memories
    memories = extract_memories(client, args.model, full_transcript, args.max_tokens)

    # Step 2: Persona description 
    persona_text = chat(
        client,
        args.model,
        [
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "user", "content": trim_to_token_limit(full_transcript, MAX_MODEL_TOKENS // 2)},
        ],
    ).strip()

    # Step 3: Utterance style guide 
    utterance_raw = chat(
        client,
        args.model,
        [
            {"role": "system", "content": UTTERANCE_PROMPT},
            {"role": "user", "content": trim_to_token_limit(full_transcript, MAX_MODEL_TOKENS // 2)},
        ],
    )
    try:
        utterance = json.loads(utterance_raw)
    except json.JSONDecodeError:
        logging.warning("Utterance JSON malformed – embedding raw text as 'style_guide'.")
        utterance = {"style_guide": utterance_raw.strip(), "sample_phrases": []}

    # Assemble & write profile
    profile: Dict[str, Any] = {
        "name": args.person.capitalize(),
        "persona": persona_text,
        "memories": memories,
        "utterance": utterance,
    }

    out_path = transcripts_dir.parent / "profile.json"
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    logging.info("Profile written → %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted by user – exiting.")

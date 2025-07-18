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

# Optional mem0 dependency
try:
    from mem0 import MemoryClient  # type: ignore
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False


DEFAULT_MODEL = "gpt-4o-mini"
MAX_MODEL_TOKENS = 8_192  # soft cap we respect regardless of model
DEFAULT_CHUNK_TOKENS = 6_000  # leave headroom for instructions & response
ENCODING = tiktoken.encoding_for_model(DEFAULT_MODEL)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# System prompts
MEMORY_PROMPT_SYSTEM = (
    "You are a knowledge engineer. Extract atomic factual memories about the speaker. "
    "Each memory must be a single declarative sentence in FIRST PERSON (using 'I', 'my', 'me') about a stable fact, preference, skill, event, or belief. "
    "Return ONLY a valid JSON array. Do not include markdown formatting, code blocks, or any text before/after the JSON. "
    "Use this exact schema: "
    "[{\"memory\": <string>, \"type\": <'biographical'|'preference'|'skill'|'belief'|'event'>, \"tags\": [<string>, ...]}]"
)

PERSONA_PROMPT = (
    "You are an expert biographer and psychologist. Based on the complete interview transcripts, "
    "create a deeply insightful persona profile in FIRST PERSON (using 'I', 'my', 'me') as if the speaker is describing themselves. "
    
    "CRITICAL: Extract the speaker's ACTUAL NAME from the transcript - do not confuse it with other people mentioned. "
    "The name should be exactly as the person introduces themselves or is referred to as the main subject. "
    
    "Create a rich, nuanced profile that captures:\n"
    "- Core identity markers (exact name, age, background, current situation)\n"
    "- Deep personality analysis (not just surface traits - look for contradictions, growth areas, internal conflicts)\n"
    "- Authentic interests (be specific - not just 'music' but what genres, artists, why they matter)\n"
    "- Genuine motivations (what really drives them, their fears, aspirations)\n"
    "- Relationship patterns (how they connect with family, friends, romantic partners)\n"
    "- Worldview and values (political views, spiritual beliefs, moral framework)\n"
    "- Personal struggles and growth areas (anxiety, procrastination, life challenges)\n"
    "- Unique quirks and characteristics that make them distinctive\n"
    
    "For personality_type, analyze their actual cognitive patterns from the transcript - don't just guess MBTI. "
    "Consider introversion/extraversion, thinking/feeling preferences, etc. based on how they actually speak and think. "
    
    "The description should be 200-300 words and feel like the person authentically describing themselves - "
    "capture their voice, their self-perception, and their unique way of seeing the world. "
    
    "Return ONLY a valid JSON object. Do not include markdown formatting, code blocks, or any text before/after the JSON. "
    "Use this exact schema: "
    "{\"name\": <string>, \"age\": <number>, \"occupation\": <string>, \"personality_traits\": [<string>, ...], \"interests\": [<string>, ...], \"goals_motivations\": [<string>, ...], \"personality_type\": <string>, \"description\": <string>}"
)

UTTERANCE_PROMPT = (
    "You are a speech pattern analyst and linguistic expert. Analyze the speaker's dialogue deeply to create a comprehensive utterance guide "
    "in FIRST PERSON (using 'I', 'my', 'me') that will allow another AI to authentically replicate their unique speaking style. "
    
    "Extract and document these specific elements:\n"
    "1. SPEECH PATTERNS: How do I structure sentences? Do I speak in long, complex thoughts or short bursts? Do I trail off or finish strongly?\n"
    "2. VOCABULARY CHOICES: What specific words do I favor? Do I use formal or casual language? Technical terms? Slang? Regional expressions?\n"
    "3. EMOTIONAL TONE: How do I express excitement, frustration, uncertainty, confidence? What's my baseline emotional register?\n"
    "4. CONVERSATIONAL HABITS: Do I interrupt myself? Circle back to topics? Ask rhetorical questions? Make references?\n"
    "5. PERSONALITY MARKERS: What verbal tics reveal my personality? How do I show agreement/disagreement? Express opinions?\n"
    "6. CULTURAL/GENERATIONAL SPEECH: What references do I make? How formal/informal am I? What cultural speech patterns do I use?\n"
    
    "For sample_phrases, extract 10-15 ACTUAL phrases the speaker uses frequently - not generic ones. "
    "Include their exact filler words, transition phrases, and characteristic expressions. "
    
    "Create an extended schema with these fields:\n"
    "- style_guide: Overall communication style (150-200 words)\n"
    "- sample_phrases: 10-15 actual phrases from the transcript\n"
    "- prosody_pacing: Detailed description of rhythm, pace, pauses\n"
    "- filler_words_quirks: Specific verbal tics and speech patterns\n"
    "- sentence_structure: How I typically build and organize my thoughts\n"
    "- emotional_expression: How I convey different emotions through speech\n"
    "- conversation_style: How I engage in dialogue with others\n"
    
    "Return ONLY a valid JSON object. Do not include markdown formatting, code blocks, or any text before/after the JSON. "
    "Use this exact schema: "
    "{\"style_guide\": <string>, \"sample_phrases\": [<string>, ...], \"prosody_pacing\": <string>, \"filler_words_quirks\": <string>, \"sentence_structure\": <string>, \"emotional_expression\": <string>, \"conversation_style\": <string>}"
)


def clean_json_response(raw_response: str) -> str:
    """Clean up JSON response by removing markdown code blocks and extra text."""
    # Remove markdown code blocks
    if "```json" in raw_response:
        start = raw_response.find("```json") + 7
        end = raw_response.find("```", start)
        if end != -1:
            raw_response = raw_response[start:end].strip()
    elif "```" in raw_response:
        start = raw_response.find("```") + 3
        end = raw_response.find("```", start)
        if end != -1:
            raw_response = raw_response[start:end].strip()
    
    # Find JSON by looking for { or [ at start
    raw_response = raw_response.strip()
    if raw_response.startswith('{') or raw_response.startswith('['):
        # Find the matching closing brace/bracket
        if raw_response.startswith('{'):
            brace_count = 0
            for i, char in enumerate(raw_response):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return raw_response[:i+1]
        elif raw_response.startswith('['):
            bracket_count = 0
            for i, char in enumerate(raw_response):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        return raw_response[:i+1]
    
    return raw_response


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
            cleaned_raw = clean_json_response(raw)
            batch = json.loads(cleaned_raw)
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
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        logging.critical("OPENAI_API_KEY not set. Please add it to your .env file.")
        sys.exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)


def upload_to_mem0(memories: List[Dict[str, Any]], persona: Dict[str, Any], utterance: Dict[str, Any], user_id: str) -> bool:
    """Upload agent profile data to Mem0 memory system."""
    if not MEM0_AVAILABLE:
        logging.warning("Mem0 not available - skipping memory upload. Install with: pip install mem0ai")
        return False
    
    try:
        # Get Mem0 Pro credentials
        from config import MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID
        if not all([MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID]):
            logging.warning("Mem0 Pro credentials not found in config - skipping memory upload")
            return False
        
        # Initialize Mem0 Pro client
        m = MemoryClient(
            api_key=MEM0_API_KEY,
            org_id=MEM0_ORG_ID,
            project_id=MEM0_PROJECT_ID
        )
        
        logging.info(f"Starting Mem0 upload for user '{user_id}'...")
        memory_count = 0
        
        # Upload memories
        for i, memory in enumerate(memories):
            memory_text = memory.get("memory", "")
            memory_type = memory.get("type", "general")
            tags = memory.get("tags", [])
            
            if memory_text:
                metadata = {
                    "type": memory_type,
                    "tags": tags,
                    "source": "profile_generation",
                    "category": memory_type
                }
                
                try:
                    messages = [{"role": "user", "content": memory_text}]
                    result = m.add(messages, user_id=user_id, metadata=metadata)
                    memory_count += 1
                    if (i + 1) % 20 == 0:  # Show progress every 20 memories
                        logging.info(f"📊 Progress: {i+1}/{len(memories)} memories uploaded")
                except Exception as e:
                    logging.error(f"❌ Failed to add memory {i+1}: {e}")
        
        logging.info(f"Uploaded {memory_count}/{len(memories)} memories to Mem0")
        
        # Upload persona information
        if persona.get("description"):
            try:
                persona_metadata = {
                    "type": "persona",
                    "personality_type": persona.get("personality_type", ""),
                    "source": "profile_generation",
                    "category": "persona"
                }
                
                messages = [{"role": "user", "content": persona['description']}]
                result = m.add(messages, user_id=user_id, metadata=persona_metadata)
                logging.info("✅ Uploaded persona to Mem0")
            except Exception as e:
                logging.error(f"❌ Failed to upload persona: {e}")
        
        # Upload utterance style guide
        if utterance.get("style_guide"):
            try:
                style_metadata = {
                    "type": "communication_style",
                    "sample_phrases": utterance.get("sample_phrases", []),
                    "source": "profile_generation",
                    "category": "communication"
                }
                
                messages = [{"role": "user", "content": utterance['style_guide']}]
                result = m.add(messages, user_id=user_id, metadata=style_metadata)
                logging.info("✅ Uploaded communication style to Mem0")
            except Exception as e:
                logging.error(f"❌ Failed to upload communication style: {e}")
        
        # Verify upload by checking memory count
        try:
            all_memories = m.get_all(user_id=user_id)
            total_memories = len(all_memories) if all_memories else 0
            logging.info(f"🔍 Verification: Found {total_memories} total memories for user '{user_id}' in Mem0")
            
            if total_memories > 0:
                logging.info(f"📊 Memory breakdown for '{user_id}':")
                # Group by type
                type_counts = {}
                for mem in all_memories:
                    if isinstance(mem, dict):
                        mem_type = mem.get('metadata', {}).get('type', 'unknown')
                    else:
                        mem_type = 'unknown'
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                
                for mem_type, count in type_counts.items():
                    logging.info(f"  - {mem_type}: {count} memories")
                    
                logging.info(f"💡 Check Mem0 dashboard for user ID: '{user_id}'")
            else:
                logging.warning("⚠️  No memories found after upload - check Mem0 dashboard manually")
                
        except Exception as e:
            logging.warning(f"Could not verify upload: {e}")
        
        logging.info(f"🎉 Successfully completed Mem0 upload for user: {user_id}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to upload to Mem0: {e}")
        return False


def verify_mem0_data(user_id: str) -> None:
    """Verify and display what's stored in Mem0 for a user."""
    if not MEM0_AVAILABLE:
        logging.error("Mem0 not available - install with: pip install mem0ai")
        return
    
    try:
        # Get Mem0 Pro credentials
        from config import MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID
        if not all([MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID]):
            logging.error("Mem0 Pro credentials not found in config - cannot verify")
            return
        
        # Initialize Mem0 Pro client
        m = MemoryClient(
            api_key=MEM0_API_KEY,
            org_id=MEM0_ORG_ID,
            project_id=MEM0_PROJECT_ID
        )
        
        # Get all memories for the user
        all_memories = m.get_all(user_id=user_id)
        
        if not all_memories:
            logging.info(f"No memories found for user '{user_id}' in Mem0")
            return
        
        logging.info(f"📊 Found {len(all_memories)} memories for user '{user_id}' in Mem0:")
        
        # Group by memory type
        memory_types = {}
        for memory in all_memories:
            if isinstance(memory, dict):
                memory_type = memory.get('metadata', {}).get('type', 'unknown')
            else:
                memory_type = 'text_memory'
            if memory_type not in memory_types:
                memory_types[memory_type] = []
            memory_types[memory_type].append(memory)
        
        for mem_type, mems in memory_types.items():
            logging.info(f"  - {mem_type}: {len(mems)} memories")
        
        # Show recent memories
        logging.info(f"\n🔍 Sample memories:")
        for i, memory in enumerate(all_memories[:5]):
            if isinstance(memory, dict):
                content = memory.get('memory', memory.get('text', 'No content'))
                mem_type = memory.get('metadata', {}).get('type', 'unknown')
            elif isinstance(memory, str):
                content = memory
                mem_type = 'unknown'
            else:
                content = str(memory)
                mem_type = 'unknown'
            logging.info(f"  {i+1}. [{mem_type}] {content[:80]}...")
        
        if len(all_memories) > 5:
            logging.info(f"  ... and {len(all_memories) - 5} more")
            
    except Exception as e:
        logging.error(f"Failed to verify Mem0 data: {e}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate a Mem0‑ready profile JSON from transcripts.")
    parser.add_argument("person", help="Subfolder name under interviews/<person>/transcripts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model, default: %(default)s")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CHUNK_TOKENS, help="Token budget per transcript chunk")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase log verbosity (-v or -vv)")
    parser.add_argument("--upload-mem0", action="store_true", help="Upload generated profile to Mem0 memory system")
    parser.add_argument("--mem0-user-id", help="User ID for Mem0 upload (defaults to person name)")
    parser.add_argument("--verify-mem0", action="store_true", help="Verify what's stored in Mem0 for the user")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    # Handle verification only mode
    if args.verify_mem0:
        user_id = args.mem0_user_id or args.person.lower()
        verify_mem0_data(user_id)
        return

    project_root = Path(__file__).resolve().parent  # current directory is project root
    transcripts_dir = project_root / "interviews" / args.person / "transcripts"
    if not transcripts_dir.exists():
        logging.critical(
            "Transcripts directory not found: %s. Run transcribe.py first or "
            "place your .txt files there.",
            transcripts_dir,
        )
        sys.exit(1)

    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        logging.critical("No .txt transcripts found in %s", transcripts_dir)
        sys.exit(1)

    full_transcript = "\n".join(p.read_text(encoding="utf‑8") for p in txt_files)

    client = build_openai_client()

    # Step 1: Memories
    memories = extract_memories(client, args.model, full_transcript, args.max_tokens)

    # Step 2: Persona description + personality type
    persona_prompt_with_name = PERSONA_PROMPT + f"\n\nIMPORTANT: The person being profiled is named '{args.person.title()}'. Make sure the 'name' field matches this exactly."
    persona_raw = chat(
        client,
        args.model,
        [
            {"role": "system", "content": persona_prompt_with_name},
            {"role": "user", "content": trim_to_token_limit(full_transcript, MAX_MODEL_TOKENS // 2)},
        ],
    )
    try:
        cleaned_persona_raw = clean_json_response(persona_raw)
        persona = json.loads(cleaned_persona_raw)
    except json.JSONDecodeError:
        logging.warning("Persona JSON malformed – storing raw text as 'description'.")
        persona = {"description": persona_raw.strip(), "personality_type": ""}

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
        cleaned_utterance_raw = clean_json_response(utterance_raw)
        utterance = json.loads(cleaned_utterance_raw)
    except json.JSONDecodeError:
        logging.warning("Utterance JSON malformed – embedding raw text as 'style_guide'.")
        utterance = {"style_guide": utterance_raw.strip(), "sample_phrases": []}

    # ── Write separate artefacts to agents folder
    agent_dir = project_root / "agents" / args.person.title()
    agent_dir.mkdir(parents=True, exist_ok=True)

    mem_path = agent_dir / "memories.json"
    mem_path.write_text(json.dumps(memories, indent=2, ensure_ascii=False))
    logging.info("Memories written → %s", mem_path)

    persona_path = agent_dir / "persona.json"
    persona_path.write_text(json.dumps(persona, indent=2, ensure_ascii=False))
    logging.info("Persona written → %s", persona_path)

    utter_path = agent_dir / "utterance.json"
    utter_path.write_text(json.dumps(utterance, indent=2, ensure_ascii=False))
    logging.info("Utterance guide written → %s", utter_path)

    # Upload to Mem0 if requested
    if args.upload_mem0:
        user_id = args.mem0_user_id or args.person.lower()
        logging.info(f"Uploading profile to Mem0 for user: {user_id}")
        success = upload_to_mem0(memories, persona, utterance, user_id)
        if success:
            logging.info("✅ Profile successfully uploaded to Mem0")
        else:
            logging.warning("❌ Failed to upload profile to Mem0")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted by user – exiting.")

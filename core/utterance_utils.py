"""Utilities for crafting agent utterances with transcript-based style cues.

The transcript snippets describe how an agent speaks—not their favorite topics
or catchphrases. They guide delivery, phrasing, and rhythm only."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from . import llm_utils

_TRANS_DIR = Path("transcripts")


def _detect_fixation_patterns(relevant_memories: str, agent_name: str) -> Optional[str]:
    """Detect if agent shows fixation patterns in recent conversation."""
    if not relevant_memories:
        return None
    
    lines = relevant_memories.split('\n')
    recent_lines = lines[-8:]  # Look at last 8 memory entries
    
    # Extract agent's own utterances
    agent_utterances = []
    for line in recent_lines:
        if f"{agent_name}:" in line:
            # Extract just the agent's response part
            response = line.split(f"{agent_name}:", 1)[-1].strip().lower()
            agent_utterances.append(response)
    
    if len(agent_utterances) < 3:
        return None
    
    # Analyze agent's patterns
    patterns = {
        'question_fixation': 0,
        'phrase_repetition': {},
        'topic_drilling': 0,
        'similar_structure': 0
    }
    
    # Check for excessive questioning
    for utterance in agent_utterances:
        if '?' in utterance:
            patterns['question_fixation'] += 1
    
    # Check for repeated phrases/structures
    for i, utterance in enumerate(agent_utterances):
        words = utterance.split()
        if len(words) >= 3:
            # Check for repeated 3-word phrases
            for j in range(len(words) - 2):
                phrase = ' '.join(words[j:j+3])
                patterns['phrase_repetition'][phrase] = patterns['phrase_repetition'].get(phrase, 0) + 1
        
        # Check for similar sentence structures (questions, imperatives, etc.)
        if i > 0:
            prev_utterance = agent_utterances[i-1]
            # Similar starting patterns
            if (utterance.startswith(('what', 'how', 'why', 'can you', 'tell me')) and 
                prev_utterance.startswith(('what', 'how', 'why', 'can you', 'tell me'))):
                patterns['similar_structure'] += 1
            # Topic drilling (following up on same theme)
            if any(word in utterance and word in prev_utterance 
                   for word in utterance.split() if len(word) > 4):
                patterns['topic_drilling'] += 1
    
    # Determine fixation type
    if patterns['question_fixation'] >= 3:
        return "asking too many questions"
    
    for phrase, count in patterns['phrase_repetition'].items():
        if count >= 2 and len(phrase.split()) >= 3:
            return f"repeating phrases like '{phrase}'"
    
    if patterns['similar_structure'] >= 2:
        return "using repetitive sentence structures"
    
    if patterns['topic_drilling'] >= 3:
        return "drilling down on the same topics"
    
    return None


def load_transcript(name: str) -> str:
    """Return style text from transcripts/<name>.json or <name>.txt if available."""
    json_path = _TRANS_DIR / f"{name.lower()}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            style = data.get("style_guide", "")
            phrases = data.get("sample_phrases", [])
            phrase_block = "\n".join(f"- {p}" for p in phrases) if phrases else ""
            return f"{style}\n{phrase_block}".strip()
        except Exception:
            return json_path.read_text(encoding="utf-8").strip()

    path = _TRANS_DIR / f"{name.lower()}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def generate_utterance(
    *,
    agent_name: str,
    personality: str,
    user_msg: str,
    relevant: str,
    graph_info: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
) -> str:
    """Generate a reply in the style of *agent_name*, referencing transcripts."""
    transcript = load_transcript(agent_name)
    
    # Check for fixation patterns in recent memories
    fixation_context = _detect_fixation_patterns(relevant, agent_name)
    
    prompt = (
        f"You are {agent_name}. Personality: {personality}\n"
        "Speak naturally and conversationally. Keep responses concise and flowing. "
        "Use fragments, statements, and natural pauses. Avoid ending every response with questions. "
        "When you do ask questions, make them feel organic to the conversation. "
        "Don't fixate on single topics - let conversations evolve naturally. "
        "Respond like you would in a casual chat with a friend.\n"
    )
    
    if fixation_context:
        # Provide specific guidance based on fixation type
        if "questions" in fixation_context:
            prompt += "IMPORTANT: You've been asking many questions lately. Try making statements, sharing thoughts, or responding more directly instead of always asking follow-ups.\n"
        elif "repeating phrases" in fixation_context:
            prompt += f"IMPORTANT: You've been {fixation_context}. Vary your language and try expressing ideas differently.\n"
        elif "repetitive sentence structures" in fixation_context:
            prompt += "IMPORTANT: You've been using similar sentence patterns. Mix up your responses - use different sentence types, lengths, and styles.\n"
        elif "drilling down" in fixation_context:
            prompt += "IMPORTANT: You've been focusing intensely on the same topics. Try acknowledging what was said and naturally moving to related but different aspects.\n"
        else:
            prompt += f"IMPORTANT: You've been {fixation_context}. Try to vary your conversational approach and let the discussion flow more naturally.\n"
    if transcript:
        prompt += f"Example speech from transcript:\n{transcript}\n\n"
    prompt += (
        f"Relevant memories:\n{relevant}\n"
        f"Graph context: {graph_info}\n\n"
        f"User: {user_msg}\n{agent_name}:"
    )
    answer = llm_utils.chat(
        [{"role": "system", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    cleaned = answer.lstrip()
    prefix = f"{agent_name}:"
    if cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):].lstrip()
    return cleaned

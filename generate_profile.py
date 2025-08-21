from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
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


def process_digital_footprint_for_mem0(footprint_dir: Path) -> str:
    """Process digital footprint data into text chunks for Mem0 to process."""
    footprint_text_chunks = []
    
    # Define processing priorities
    priority_sources = [
        ("Profile/Profile.json", "profile", process_profile_data),
        ("Chrome/Settings.json", "settings", process_settings_data),
        ("Chrome/Extensions.json", "extensions", process_extensions_data),
        ("Chrome/History.json", "browsing", process_browsing_data),
        ("NotebookLM/*/", "academic", process_notebooklm_data),
        ("My Activity/Search/MyActivity.html", "search", process_search_data),
        ("My Activity/YouTube/MyActivity.html", "youtube", process_youtube_data),
        ("My Activity/Maps/MyActivity.html", "location", process_location_data),
        ("Chrome/Bookmarks.html", "bookmarks", process_bookmark_data),
        ("Saved/*.csv", "preferences", process_saved_items_data),
        ("Maps/My labeled places/Labeled places.json", "places", process_places_data),
        ("Maps (your places)/Reviews.json", "reviews", process_reviews_data),
        ("Timeline/Settings.json", "timeline", process_timeline_data),
    ]
    
    for pattern, source_type, processor in priority_sources:
        try:
            if "*/" in pattern:  # Handle directory patterns like "NotebookLM/*/"
                base_pattern = pattern.replace("*/", "")
                dirs = [d for d in (footprint_dir / base_pattern).iterdir() if d.is_dir()]
                files = dirs
            elif "*" in pattern:
                files = list(footprint_dir.glob(pattern))
            else:
                files = [footprint_dir / pattern] if (footprint_dir / pattern).exists() else []
            
            for file_path in files:
                if file_path.exists():
                    logging.info(f"Processing {source_type}: {file_path.name}")
                    processed_text = processor(file_path)
                    if processed_text:
                        footprint_text_chunks.append(f"[{source_type.upper()}]\n{processed_text}")
                    
        except Exception as e:
            logging.warning(f"Failed to process {source_type} from {pattern}: {e}")
    
    return "\n\n".join(footprint_text_chunks)


def process_profile_data(file_path: Path) -> str:
    """Process profile data into text for Mem0."""
    try:
        profile_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        # Convert profile to readable text
        profile_text = f"""
        Name: {profile_data.get('displayName', 'Unknown')}
        Email: {profile_data.get('emails', [{}])[0].get('value', 'Unknown') if profile_data.get('emails') else 'Unknown'}
        Birthday: {profile_data.get('birthday', 'Unknown')}
        Gender: {profile_data.get('gender', {}).get('type', 'Unknown')}
        """
        
        return profile_text.strip()
        
    except Exception as e:
        logging.warning(f"Failed to process profile data: {e}")
        return ""


def process_browsing_data(file_path: Path) -> str:
    """Process browsing history into text for Mem0."""
    try:
        history_data = json.loads(file_path.read_text(encoding="utf-8"))
        browser_history = history_data.get("Browser History", [])
        
        # Analyze browsing patterns
        domain_analysis = {}
        content_categories = {
            'educational': ['edu', 'coursera', 'khan', 'udemy', 'mit', 'stanford', 'scad', 'blackboard'],
            'creative': ['figma', 'adobe', 'behance', 'dribbble', 'unsplash', 'deviantart'],
            'social': ['facebook', 'instagram', 'twitter', 'reddit', 'discord', 'whatsapp'],
            'entertainment': ['youtube', 'netflix', 'spotify', 'twitch', 'tiktok'],
            'professional': ['linkedin', 'github', 'stackoverflow', 'medium'],
            'shopping': ['amazon', 'ebay', 'etsy', 'shopify'],
            'gaming': ['steam', 'epic', 'itch.io', 'roblox']
        }
        
        category_counts = {cat: 0 for cat in content_categories.keys()}
        
        for entry in browser_history[:1000]:
            url = entry.get("url", "")
            title = entry.get("title", "")
            
            if url and title:
                domain = url.split("//")[-1].split("/")[0]
                
                if domain not in domain_analysis:
                    domain_analysis[domain] = {'count': 0, 'titles': set()}
                domain_analysis[domain]['count'] += 1
                domain_analysis[domain]['titles'].add(title[:100])
                
                domain_lower = domain.lower()
                url_lower = url.lower()
                title_lower = title.lower()
                
                for category, keywords in content_categories.items():
                    if any(keyword in domain_lower or keyword in url_lower or keyword in title_lower for keyword in keywords):
                        category_counts[category] += 1
                        break
        
        browsing_insights = []
        
        top_domains = sorted(domain_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
        for domain, info in top_domains:
            sample_titles = list(info['titles'])[:3]
            browsing_insights.append(f"I frequently visit {domain} ({info['count']} times) for: {'; '.join(sample_titles)}")
        
        total_categorized = sum(category_counts.values())
        if total_categorized > 0:
            for category, count in category_counts.items():
                if count > 0:
                    percentage = (count / total_categorized) * 100
                    browsing_insights.append(f"{percentage:.1f}% of my browsing is {category}-related ({count} visits)")
        
        return "\n".join(browsing_insights)
        
    except Exception as e:
        logging.warning(f"Failed to process browsing data: {e}")
        return ""


def process_search_data(file_path: Path) -> str:
    """Extract memories from search activity with enhanced parsing."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        import re
        
        # Multiple patterns to extract different types of search data
        search_patterns = {
            'queries': [
                r'Searched for\s+([^<\n]+)',
                r'<div[^>]*search[^>]*>([^<]+)</div>',
                r'data-search="([^"]+)"',
                r'query["\']:\s*["\']([^"\']+)["\']'
            ],
            'visited_results': [
                r'Visited\s+([^<\n]+)',
                r'<a[^>]*href="https://[^"]*"[^>]*>([^<]+)</a>'
            ],
            'auto_complete': [
                r'Selected suggestion\s+([^<\n]+)',
                r'autocomplete[^>]*>([^<]+)<'
            ]
        }
        
        extracted_data = {
            'search_queries': [],
            'visited_sites': [],
            'search_topics': [],
            'search_frequency': {}
        }
        
        # Extract search queries with better filtering
        all_queries = []
        for pattern_list in search_patterns['queries']:
            matches = re.findall(pattern_list, content, re.IGNORECASE)
            all_queries.extend(matches)
        
        # Clean and categorize search queries
        for query in all_queries[:200]:  # Increase limit for better analysis
            cleaned = re.sub(r'[\\n\\r\\t]', ' ', query).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Filter out noise
            if (3 < len(cleaned) < 150 and 
                not cleaned.lower().startswith(('google', 'search', 'activity', 'my account', 'sign in', 'www.')) and
                not re.match(r'^[0-9\s\-]+$', cleaned)):  # Not just numbers/dates
                
                extracted_data['search_queries'].append(cleaned)
                
                # Track search frequency for topics
                words = cleaned.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        extracted_data['search_frequency'][word] = extracted_data['search_frequency'].get(word, 0) + 1
        
        # Categorize search interests
        category_keywords = {
            'academic': ['study', 'university', 'research', 'thesis', 'paper', 'academic', 'scholar', 'course', 'lecture'],
            'creative': ['design', 'art', 'creative', 'photoshop', 'figma', 'adobe', 'portfolio', 'drawing'],
            'technology': ['programming', 'code', 'software', 'tech', 'computer', 'app', 'development'],
            'career': ['job', 'career', 'interview', 'resume', 'internship', 'salary', 'professional'],
            'personal': ['health', 'fitness', 'recipe', 'travel', 'relationship', 'hobby'],
            'entertainment': ['movie', 'music', 'game', 'youtube', 'netflix', 'show', 'entertainment'],
            'shopping': ['buy', 'price', 'review', 'amazon', 'product', 'purchase', 'store']
        }
        
        category_counts = {cat: 0 for cat in category_keywords.keys()}
        
        for query in extracted_data['search_queries']:
            query_lower = query.lower()
            for category, keywords in category_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    category_counts[category] += 1
                    break
        
        # Build comprehensive search insights
        insights = []
        
        # Top search topics by frequency
        if extracted_data['search_frequency']:
            top_terms = sorted(extracted_data['search_frequency'].items(), key=lambda x: x[1], reverse=True)[:15]
            frequent_terms = [f"{term} ({count}x)" for term, count in top_terms if count > 1]
            if frequent_terms:
                insights.append(f"Frequently searched terms: {', '.join(frequent_terms)}")
        
        # Category analysis
        total_categorized = sum(category_counts.values())
        if total_categorized > 0:
            for category, count in category_counts.items():
                if count > 0:
                    percentage = (count / total_categorized) * 100
                    insights.append(f"{percentage:.1f}% of searches are {category}-related ({count} queries)")
        
        # Sample interesting queries
        interesting_queries = [q for q in extracted_data['search_queries'][:30] if len(q) > 10]
        if interesting_queries:
            insights.append(f"Sample search queries: {'; '.join(interesting_queries[:10])}")
        
        if insights:
            return "; ".join(insights)
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process search data: {e}")
        return ""


def process_location_data(file_path: Path) -> str:
    """Process location/maps activity into text for Mem0."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        import re
        location_pattern = r'location[^<]*'
        matches = re.findall(location_pattern, content, re.IGNORECASE)
        
        if matches:
            return "My location activity: " + "; ".join(matches[:50])
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process location data: {e}")
        return ""


def process_bookmark_data(file_path: Path) -> str:
    """Process bookmarks into text for Mem0."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        import re
        bookmark_pattern = r'<A[^>]*>([^<]+)</A>'
        titles = re.findall(bookmark_pattern, content)
        
        if titles:
            return "My bookmarked content: " + "; ".join(titles[:50])
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process bookmark data: {e}")
        return ""


def process_saved_items_data(file_path: Path) -> str:
    """Process saved items CSV files into text for Mem0."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        lines = content.split('\n')[:100]
        return f"My saved items from {file_path.stem}: " + "; ".join(lines[1:21])
        
    except Exception as e:
        logging.warning(f"Failed to process saved items data: {e}")
        return ""


def process_settings_data(file_path: Path) -> str:
    """Extract memories from Chrome settings and preferences."""
    try:
        settings_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        insights = []
        
        # Search engines reveal interests
        search_engines = settings_data.get("Search Engines", [])
        custom_engines = [eng for eng in search_engines if eng.get("prepopulate_id", 0) == 0]
        if custom_engines:
            engine_names = [eng.get("short_name", "") for eng in custom_engines[:10]]
            insights.append(f"Uses custom search engines: {', '.join(filter(None, engine_names))}")
        
        # Language preferences
        prefs = settings_data.get("Preferences", [])
        for pref in prefs:
            if pref.get("name") == "intl.accept_languages":
                languages = pref.get("value", "").strip('\"')
                insights.append(f"Language preferences: {languages}")
            elif pref.get("name") == "custom_links.list":
                # Parse custom shortcuts
                try:
                    links_data = json.loads(pref.get("value", "[]"))
                    shortcuts = [link.get("title", "") for link in links_data[:8]]
                    insights.append(f"Custom browser shortcuts: {', '.join(filter(None, shortcuts))}")
                except:
                    pass
            elif pref.get("name") == "translate_ignored_count_for_language":
                # Translation behavior indicates multilingual usage
                try:
                    lang_data = json.loads(pref.get("value", "{}"))
                    if lang_data:
                        insights.append(f"Frequently encounters languages: {', '.join(lang_data.keys())}")
                except:
                    pass
        
        # Demographics from sync data
        priority_prefs = settings_data.get("Priority Preferences", [])
        for pref_wrapper in priority_prefs:
            pref = pref_wrapper.get("preference", {})
            if pref.get("name") == "sync.demographics":
                try:
                    demo_data = json.loads(pref.get("value", "{}"))
                    if demo_data.get("birth_year"):
                        age = 2025 - demo_data["birth_year"]
                        insights.append(f"Approximately {age} years old")
                    if demo_data.get("gender") == 1:
                        insights.append("Identifies as male")
                except:
                    pass
        
        return "My browser settings and preferences: " + "; ".join(insights)
        
    except Exception as e:
        logging.warning(f"Failed to process settings data: {e}")
        return ""


def process_extensions_data(file_path: Path) -> str:
    """Extract memories from Chrome extensions showing tool preferences."""
    try:
        ext_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        extensions = ext_data.get("Extensions", [])
        enabled_extensions = [ext for ext in extensions if ext.get("enabled")]
        
        # Map extension IDs to likely purposes (common ones)
        extension_insights = {
            "cfhdojbkjhnklbpkdaibdccddilifddb": "Uses Adblock Plus (privacy-conscious)",
            "oocalimimngaihdkbihfgmpkcpnmlaoa": "Uses text highlighter tool (research/study habits)",
            "jjnipfcfcddhgepeneeedbiophaehhkb": "Had ad-related extension (gaming/monetization interest)"
        }
        
        insights = []
        for ext in enabled_extensions:
            ext_id = ext.get("id", "")
            if ext_id in extension_insights:
                insights.append(extension_insights[ext_id])
            elif ext.get("name"):
                insights.append(f"Uses browser extension: {ext['name']}")
        
        # Extension settings analysis
        ext_settings = ext_data.get("Extension Settings", [])
        for setting in ext_settings:
            if "gaming" in setting.get("value", "").lower():
                insights.append("Has gaming-related browser extensions")
                break
        
        if insights:
            return "My browser extension usage: " + "; ".join(insights)
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process extensions data: {e}")
        return ""


def process_notebooklm_data(dir_path: Path) -> str:
    """Extract memories from NotebookLM projects showing academic interests."""
    try:
        all_memories = []
        
        # Process each NotebookLM project directory
        for project_dir in dir_path.iterdir():
            if not project_dir.is_dir():
                continue
                
            project_name = project_dir.name
            insights = [f"Studied/researched: {project_name}"]
            
            # Look for metadata files
            for meta_file in project_dir.glob("*.json"):
                try:
                    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta_data.get("title"):
                        insights.append(f"Academic project title: {meta_data['title']}")
                    if meta_data.get("emoji"):
                        insights.append(f"Project theme: {meta_data['emoji']}")
                except:
                    continue
            
            # Look for source materials
            sources_dir = project_dir / "Sources"
            if sources_dir.exists():
                source_files = list(sources_dir.glob("*.pdf metadata.json"))
                for source_file in source_files[:3]:  # Limit to 3 sources per project
                    try:
                        source_meta = json.loads(source_file.read_text(encoding="utf-8"))
                        if source_meta.get("title"):
                            insights.append(f"Studied source: {source_meta['title']}")
                    except:
                        continue
            
            # Look for generated notes
            notes_dir = project_dir / "Notes"
            if notes_dir.exists():
                note_files = list(notes_dir.glob("*.html"))
                for note_file in note_files[:2]:  # Limit notes
                    note_title = note_file.stem.replace("_", " ")
                    insights.append(f"Created study notes: {note_title}")
            
            if len(insights) > 1:  # Only if we found meaningful data
                all_memories.append(f"NotebookLM academic project: {'; '.join(insights)}")
        
        return "\n\n".join(all_memories)
        
    except Exception as e:
        logging.warning(f"Failed to process NotebookLM data: {e}")
        return ""


def process_youtube_data(file_path: Path) -> str:
    """Extract memories from YouTube activity."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Extract video titles and channel information
        import re
        
        # Look for video titles in the HTML
        title_patterns = [
            r'Watched\s+([^<]+)',
            r'Searched for\s+([^<]+)',
            r'>([^<]+)</a>.*?youtube\.com'
        ]
        
        video_interests = []
        search_queries = []
        
        for pattern in title_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:20]:  # Limit to avoid overwhelming
                cleaned = re.sub(r'[\\n\\r\\t]', ' ', match).strip()
                if len(cleaned) > 3 and len(cleaned) < 100:
                    if 'searched for' in pattern.lower():
                        search_queries.append(cleaned)
                    else:
                        video_interests.append(cleaned)
        
        insights = []
        if video_interests:
            insights.append(f"YouTube viewing interests: {'; '.join(video_interests[:15])}")
        if search_queries:
            insights.append(f"YouTube search queries: {'; '.join(search_queries[:10])}")
        
        if insights:
            return "My YouTube activity: " + "; ".join(insights)
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process YouTube data: {e}")
        return ""


def process_reviews_data(file_path: Path) -> str:
    """Extract memories from Google Maps reviews."""
    try:
        reviews_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        insights = []
        if isinstance(reviews_data, list):
            for review in reviews_data[:10]:  # Limit reviews
                if isinstance(review, dict):
                    place_name = review.get("placeName", "")
                    rating = review.get("starRating", "")
                    comment = review.get("comment", "")
                    
                    if place_name:
                        review_text = f"Reviewed {place_name}"
                        if rating:
                            review_text += f" ({rating} stars)"
                        if comment and len(comment) < 100:
                            review_text += f": {comment[:80]}"
                        insights.append(review_text)
        
        if insights:
            return "My Google Maps reviews and opinions: " + "; ".join(insights)
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process reviews data: {e}")
        return ""


def process_timeline_data(file_path: Path) -> str:
    """Extract memories from timeline settings."""
    try:
        timeline_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        # Timeline settings can reveal privacy preferences and location habits
        insights = []
        if timeline_data.get("enableLocationHistory"):
            insights.append("Actively tracks location history")
        if timeline_data.get("enableWebAndAppActivity"):
            insights.append("Shares web and app activity data")
        
        if insights:
            return "My digital privacy and tracking preferences: " + "; ".join(insights)
        
        return ""
        
    except Exception as e:
        logging.warning(f"Failed to process timeline data: {e}")
        return ""


def process_places_data(file_path: Path) -> str:
    """Extract memories from labeled places."""
    try:
        places_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        places_text = "My labeled places: "
        if isinstance(places_data, list):
            place_names = [place.get("name", "") for place in places_data[:20]]
            places_text += "; ".join(filter(None, place_names))
        
        return places_text
        
    except Exception as e:
        logging.warning(f"Failed to process places data: {e}")
        return ""


# Note: Custom memory extraction has been removed in favor of Mem0's built-in memory generation


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


def upload_to_mem0(transcript: str, footprint_text: str, persona: Dict[str, Any], utterance: Dict[str, Any], user_id: str) -> bool:
    """Upload agent profile data to Mem0 memory system.

    Sends raw transcript and digital footprint chunks so Mem0 can
    perform its own memory extraction and relationship building.
    """
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

        # Upload transcript chunks
        if transcript:
            chunks = chunk_text(transcript, DEFAULT_CHUNK_TOKENS)
            for i, chunk in enumerate(chunks, 1):
                metadata = {
                    "category": "conversation",
                    "source": "interview_transcript",
                }
                try:
                    messages = [{"role": "user", "content": chunk}]
                    m.add(messages, user_id=user_id, metadata=metadata)
                    if i % 20 == 0:
                        logging.info(f"📊 Progress: {i}/{len(chunks)} transcript chunks uploaded")
                except Exception as e:
                    logging.error(f"❌ Failed to add transcript chunk {i}: {e}")
            logging.info(f"Uploaded {len(chunks)} transcript chunks to Mem0")
        
        # Upload digital footprint data
        if footprint_text:
            footprint_chunks = chunk_text(footprint_text, DEFAULT_CHUNK_TOKENS)
            for i, chunk in enumerate(footprint_chunks, 1):
                metadata = {
                    "category": "digital_footprint",
                    "source": "digital_footprint_analysis",
                }
                try:
                    messages = [{"role": "user", "content": chunk}]
                    m.add(messages, user_id=user_id, metadata=metadata)
                    if i % 10 == 0:
                        logging.info(f"📊 Progress: {i}/{len(footprint_chunks)} footprint chunks uploaded")
                except Exception as e:
                    logging.error(f"❌ Failed to add footprint chunk {i}: {e}")
            logging.info(f"Uploaded {len(footprint_chunks)} digital footprint chunks to Mem0")
        
        # Upload persona information
        if persona.get("description"):
            try:
                persona_metadata = {
                    "category": "persona",
                    "personality_type": persona.get("personality_type", ""),
                    "source": "profile_generation",
                }
                
                messages = [{"role": "user", "content": persona['description']}]
                m.add(messages, user_id=user_id, metadata=persona_metadata)
                logging.info("✅ Uploaded persona to Mem0")
            except Exception as e:
                logging.error(f"❌ Failed to upload persona: {e}")
        
        # Upload utterance style guide
        if utterance.get("style_guide"):
            try:
                style_metadata = {
                    "category": "communication",
                    "sample_phrases": utterance.get("sample_phrases", []),
                    "source": "profile_generation",
                }
                
                messages = [{"role": "user", "content": utterance['style_guide']}]
                m.add(messages, user_id=user_id, metadata=style_metadata)
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
    parser = argparse.ArgumentParser(description="Generate a Mem0‑ready profile JSON from transcripts and digital footprint.")
    parser.add_argument("person", help="Subfolder name under interviews/<person>/transcripts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model, default: %(default)s")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CHUNK_TOKENS, help="Token budget per transcript chunk")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase log verbosity (-v or -vv)")
    parser.add_argument("--upload-mem0", action="store_true", help="Upload generated profile to Mem0 memory system")
    parser.add_argument("--mem0-user-id", help="User ID for Mem0 upload (defaults to person name)")
    parser.add_argument("--verify-mem0", action="store_true", help="Verify what's stored in Mem0 for the user")
    parser.add_argument("--skip-footprint", action="store_true", help="Skip digital footprint processing")
    parser.add_argument("--save-raw", action="store_true", help="Save raw transcript and footprint samples for debugging")
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

    # Process digital footprint if available
    footprint_text = ""
    if not args.skip_footprint:
        footprint_dir = project_root / "digital footprint" / args.person.lower()
        if footprint_dir.exists():
            logging.info(f"Processing digital footprint from {footprint_dir}")
            footprint_text = process_digital_footprint_for_mem0(footprint_dir)
            if footprint_text:
                logging.info(f"Processed digital footprint data ({len(footprint_text)} characters)")
        else:
            logging.info(f"No digital footprint found at {footprint_dir}")
    else:
        logging.info("Skipping digital footprint processing (--skip-footprint flag)")

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

    # ── Write persona and utterance artefacts to agents folder
    agent_dir = project_root / "agents" / args.person.title()
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Note: We no longer save memories.json as Mem0 handles memory storage
    # Only save persona and utterance for agent configuration
    
    persona_path = agent_dir / "persona.json"
    persona_path.write_text(json.dumps(persona, indent=2, ensure_ascii=False))
    logging.info("Persona written → %s", persona_path)

    utter_path = agent_dir / "utterance.json"
    utter_path.write_text(json.dumps(utterance, indent=2, ensure_ascii=False))
    logging.info("Utterance guide written → %s", utter_path)
    
    # Optionally save raw data for backup/debugging
    if args.save_raw:
        raw_data = {
            "transcript": full_transcript[:10000],  # Save first 10k chars as sample
            "footprint_summary": footprint_text[:5000] if footprint_text else "",
            "timestamp": time.time()
        }
        raw_path = agent_dir / "raw_data.json"
        raw_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False))
        logging.info("Raw data sample saved → %s", raw_path)

    # Upload to Mem0 if requested
    if args.upload_mem0:
        user_id = args.mem0_user_id or args.person.lower()
        logging.info(f"Uploading profile to Mem0 for user: {user_id}")
        success = upload_to_mem0(full_transcript, footprint_text, persona, utterance, user_id)
        if success:
            logging.info("✅ Profile successfully uploaded to Mem0")
        else:
            logging.warning("❌ Failed to upload profile to Mem0")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted by user – exiting.")

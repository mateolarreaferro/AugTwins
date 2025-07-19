from pathlib import Path
import json
from typing import List

from agents.profile_base import ProfileAgent

# Optional Mem0 dependency
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

# File paths for Lars profile data
AGENT_DIR = Path(__file__).resolve().parent
MEM_PATH = AGENT_DIR / "memories.json"
PERSONA_PATH = AGENT_DIR / "persona.json"
UTTERANCE_PATH = AGENT_DIR / "utterance.json"

# Load persona data
if PERSONA_PATH.exists():
    _persona_data = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
else:
    _persona_data = {"description": "", "personality_type": ""}

# Load utterance data
if UTTERANCE_PATH.exists():
    _utterance_data = json.loads(UTTERANCE_PATH.read_text(encoding="utf-8"))
else:
    _utterance_data = {"style_guide": "", "sample_phrases": []}

PERSONA_DESCRIPTION = _persona_data.get("description", "")
PERSONALITY_TYPE = _persona_data.get("personality_type", "")
STYLE_GUIDE = _utterance_data.get("style_guide", "")
SAMPLE_PHRASES = _utterance_data.get("sample_phrases", [])

# Build SEED_MEMORIES for backwards compatibility
SEED_MEMORIES = []
if MEM_PATH.exists():
    mem_list = json.loads(MEM_PATH.read_text(encoding="utf-8"))
    for m in mem_list:
        if isinstance(m, dict):
            memory_text = m.get("memory", "")
            if memory_text:
                SEED_MEMORIES.append(memory_text)


class Lars(ProfileAgent):
    transcript_path = UTTERANCE_PATH  # Now points to utterance.json
    persona = PERSONA_DESCRIPTION

    def __init__(self) -> None:
        # Combine persona description with utterance style for rich personality
        full_personality = f"{PERSONA_DESCRIPTION}\n\nCommunication Style: {STYLE_GUIDE}"
        if PERSONALITY_TYPE:
            full_personality += f" (Personality Type: {PERSONALITY_TYPE})"
        
        super().__init__(name="Lars", personality=full_personality, tts_voice_id="5epn2vbuws8S5MRzxJH8")
        
        # Load memories from Mem0 with proper error handling
        print("[Lars] Lars initialized successfully")
        self._memories_loaded = False
        self._mem0_client = None

    def _load_mem0_memories(self) -> bool:
        """Load memories from Mem0 if available."""
        client = self._get_mem0_client()
        if not client:
            return False
        
        try:
            print("[Lars] Fetching memories from Mem0...")
            # Get all memories for Lars
            all_memories = client.get_all(user_id="lars")
            
            if all_memories and len(all_memories) > 0:
                print(f"[Lars] Found {len(all_memories)} memories in Mem0")
                for memory in all_memories:
                    if isinstance(memory, dict):
                        memory_text = memory.get('memory', memory.get('text', ''))
                        if memory_text:
                            # Use local add_memory to avoid recursive Mem0 calls during loading
                            self._add_memory_local_only(memory_text)
                    elif isinstance(memory, str):
                        self._add_memory_local_only(memory)
                return True
            else:
                print("[Lars] No memories found in Mem0")
                return False
                
        except Exception as e:
            print(f"[Lars] Failed to load from Mem0: {e}")
            return False

    def _add_memory_local_only(self, text: str):
        """Add memory to local storage only (no Mem0 sync during loading)."""
        from core.agent import Memory, _EMBEDDER
        import time
        
        emb = _EMBEDDER.encode(text)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        else:
            emb = list(emb)
        
        mem = Memory(
            text=text,
            timestamp=time.time(),
            embedding=emb,
            is_summary=False,
        )
        self.memory.append(mem)
        self._update_graph(text)

    def _load_local_memories(self) -> None:
        """Load memories from local memories.json file."""
        if MEM_PATH.exists():
            mem_list = json.loads(MEM_PATH.read_text(encoding="utf-8"))
            print(f"[Lars] Loading {len(mem_list)} memories from local file")
            for m in mem_list:
                if isinstance(m, dict):
                    memory_text = m.get("memory", "")
                    if memory_text:
                        self.add_memory(memory_text)

    def retrieve_memories(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve memories using Mem0 Pro client or local fallback."""
        results: List[str] = []
        
        # Try Mem0 first
        client = self._get_mem0_client()
        if client:
            try:
                # Search memories for Lars
                search_results = client.search(query, user_id="lars", limit=top_k)
                
                if search_results:
                    for result in search_results:
                        if isinstance(result, dict):
                            memory_text = result.get('memory', result.get('text', ''))
                            if memory_text:
                                results.append(memory_text)
                        elif isinstance(result, str):
                            results.append(result)
                            
            except Exception as e:
                print(f"[Lars] Mem0 search failed: {e}, using local memories")
        
        # Fallback to local retrieval if Mem0 fails or no results
        if not results:
            results = super().retrieve_memories(query, top_k)
            
        return results

    def _get_mem0_client(self):
        """Get or create Mem0 client with graph memory enabled."""
        if self._mem0_client is None and MEM0_AVAILABLE:
            try:
                from config import MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID
                if not all([MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID]):
                    print("[Lars] Mem0 credentials not found in config")
                    return None
                
                # Initialize Mem0 Pro client
                self._mem0_client = MemoryClient(
                    api_key=MEM0_API_KEY,
                    org_id=MEM0_ORG_ID,
                    project_id=MEM0_PROJECT_ID
                )
                
                # Check if graph memory is enabled
                project_info = self._mem0_client.get_project()
                graph_enabled = project_info.get('enable_graph', False)
                print(f"[Lars] Mem0 client initialized (graph memory: {'enabled' if graph_enabled else 'disabled'})")
            except ImportError:
                print("[Lars] Mem0 credentials not found")
            except Exception as e:
                print(f"[Lars] Error creating Mem0 client: {e}")
        return self._mem0_client

    def _ensure_memories_loaded(self):
        """Load memories on first conversation if not already loaded."""
        if not self._memories_loaded:
            print("[Lars] Loading memories on first use...")
            try:
                # Try Mem0 first
                if self._load_mem0_memories():
                    print(f"[Lars] Loaded {len(self.memory)} memories from Mem0")
                else:
                    self._load_local_memories()
                    print(f"[Lars] Loaded {len(self.memory)} memories from local file")
                self._memories_loaded = True
            except Exception as e:
                print(f"[Lars] Error loading memories: {e}")
                self._memories_loaded = True  # Don't keep trying

    def generate_response(self, user_msg: str, *, model: str = "gpt-4o-mini") -> str:
        """Generate response using Lars' specific style and Mem0 memories."""
        # Memories are loaded during app startup, no need for lazy loading
        
        # Retrieve relevant memories (this will use Mem0 if available)
        relevant_memories = "\n".join(self.retrieve_memories(user_msg))
        
        # Get graph context
        graph_info = ", ".join(self.graph_context(user_msg))
        
        # Enhanced prompt that incorporates utterance patterns
        enhanced_personality = f"{self.personality}\n\nSpeech Patterns: Use phrases like: {', '.join(SAMPLE_PHRASES[:5])}"
        
        # Import utterance utils for response generation
        from core import utterance_utils
        
        response = utterance_utils.generate_utterance(
            agent_name=self.name,
            personality=enhanced_personality,
            user_msg=user_msg,
            relevant=relevant_memories,
            graph_info=graph_info,
            model=model,
            temperature=0.8,
        )
        
        # Note: We don't automatically store conversations in memory anymore
        # The user will be asked at the end of the chat if they want to save to Mem0
        return response
    
    def add_memory_to_mem0(self, text: str, metadata: dict = None) -> bool:
        """Add memory to Mem0 Pro with graph relationships enabled."""
        client = self._get_mem0_client()
        if not client:
            return False
        
        try:
            # Format as message list for Mem0 Pro API
            messages = [{"role": "user", "content": text}]
            result = client.add(messages, user_id="lars", metadata=metadata)
            print(f"[Lars] Added memory to Mem0 Pro: {result}")
            return True
        except Exception as e:
            print(f"[Lars] Error adding memory to Mem0 Pro: {e}")
            return False
    
    def get_memory_graph(self) -> dict:
        """Get the memory graph for Lars from Mem0 Pro."""
        client = self._get_mem0_client()
        if not client:
            return {}
        
        try:
            # Note: Graph relationships are automatically created by Mem0 Pro
            print("[Lars] Graph relationships are managed by Mem0 Pro dashboard")
            return {"status": "Graph relationships available in Mem0 Pro dashboard"}
        except Exception as e:
            print(f"[Lars] Error getting memory graph: {e}")
            return {}

    def reflect_on_conversation(self, conversation_history: list, model: str = "gpt-4o-mini") -> str:
        """Generate reflection on conversation and extract new learnings."""
        if not conversation_history:
            return ""
        
        # Build conversation text
        conversation_text = "\n".join([
            f"User: {conv['user']}\nLars: {conv['agent']}" 
            for conv in conversation_history
        ])
        
        # Create reflection prompt
        reflection_prompt = f"""
As Lars, reflect on this conversation and identify what you learned about yourself, the user, or new insights you gained.

Conversation:
{conversation_text}

Instructions:
1. Write a brief reflection from Lars' perspective (100-150 words)
2. Note any new insights about yourself or the user
3. Identify topics you'd like to explore further
4. Use Lars' authentic voice and speaking style

Format your response as a JSON object:
{{
    "reflection": "Your reflection as Lars...",
    "new_insights": ["insight 1", "insight 2", ...],
    "topics_to_explore": ["topic 1", "topic 2", ...],
    "user_observations": "What you learned about the user..."
}}
"""

        from core import utterance_utils
        from generate_profile import clean_json_response
        import json
        
        try:
            raw_reflection = utterance_utils.generate_utterance(
                agent_name=self.name,
                personality=self.personality,
                user_msg=reflection_prompt,
                relevant="",
                graph_info="",
                model=model,
                temperature=0.6,
            )
            
            # Try to parse as JSON, fallback to text
            try:
                cleaned = clean_json_response(raw_reflection)
                reflection_data = json.loads(cleaned)
                return reflection_data
            except json.JSONDecodeError:
                return {"reflection": raw_reflection, "new_insights": [], "topics_to_explore": [], "user_observations": ""}
                
        except Exception as e:
            print(f"[Lars] Error generating reflection: {e}")
            return {"reflection": "Reflection generation failed", "new_insights": [], "topics_to_explore": [], "user_observations": ""}


lars = Lars()

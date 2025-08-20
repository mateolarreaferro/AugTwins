#!/usr/bin/env python3
"""
Sync local memories to Mem0 cloud.
Uploads any local memories that aren't already in the cloud.
Uses the same upload format as generate_profile.py for consistency.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

# Use Mem0 client library like generate_profile.py
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("❌ Mem0 library not installed. Install with: pip install mem0ai")

from config import MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID

def validate_config():
    """Check if Mem0 credentials are configured and library is available."""
    if not MEM0_AVAILABLE:
        print("❌ Mem0 library not available - install with: pip install mem0ai")
        return False
    
    if not all([MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID]):
        print("❌ Mem0 credentials not configured!")
        print("Please set these in your .env file:")
        print("  - MEM0_API_KEY")
        print("  - MEM0_ORG_ID") 
        print("  - MEM0_PROJECT_ID")
        return False
    return True

def get_mem0_client() -> Optional[MemoryClient]:
    """Initialize and return Mem0 Pro client."""
    if not MEM0_AVAILABLE:
        return None
    
    try:
        client = MemoryClient(
            api_key=MEM0_API_KEY,
            org_id=MEM0_ORG_ID,
            project_id=MEM0_PROJECT_ID
        )
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Mem0 client: {e}")
        return None

def get_cloud_memories(user_id: str = "lars") -> List[Dict]:
    """Fetch all memories from Mem0 cloud using the client library."""
    print(f"📡 Fetching memories for '{user_id}' from Mem0 cloud...")
    
    client = get_mem0_client()
    if not client:
        print("❌ Failed to initialize Mem0 client")
        return []
    
    try:
        # Use the client library's get_all method like in generate_profile.py
        all_memories = client.get_all(user_id=user_id)
        
        if not all_memories:
            print(f"  No memories found for user '{user_id}'")
            return []
        
        print(f"✅ Total cloud memories: {len(all_memories)}")
        return all_memories if all_memories else []
        
    except Exception as e:
        print(f"❌ Error fetching cloud memories: {e}")
        return []

def categorize_memory(text: str) -> Dict[str, any]:
    """Categorize a memory based on its content (similar to generate_profile.py logic)."""
    text_lower = text.lower()
    
    # Determine type and tags based on content
    if any(word in text_lower for word in ["i am", "my name", "years old", "born", "from"]):
        return {"type": "biographical", "tags": ["personal", "identity"]}
    elif any(word in text_lower for word in ["i like", "i prefer", "i enjoy", "favorite", "love"]):
        return {"type": "preference", "tags": ["preferences", "likes"]}
    elif any(word in text_lower for word in ["i can", "i know", "i learned", "experienced in"]):
        return {"type": "skill", "tags": ["skills", "abilities"]}
    elif any(word in text_lower for word in ["i believe", "i think", "my opinion", "i feel"]):
        return {"type": "belief", "tags": ["beliefs", "opinions"]}
    elif any(word in text_lower for word in ["i went", "i did", "happened", "i visited"]):
        return {"type": "event", "tags": ["events", "experiences"]}
    else:
        return {"type": "general", "tags": ["general"]}

def load_local_memories(agent_name: str = "lars") -> List[Dict]:
    """Load memories from local JSON file."""
    memory_file = Path(f"memories/{agent_name}_memories.json")
    
    if not memory_file.exists():
        print(f"❌ Local memory file not found: {memory_file}")
        return []
    
    print(f"📂 Loading local memories from {memory_file}...")
    with open(memory_file, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    print(f"✅ Loaded {len(memories)} local memories")
    return memories

def upload_memory(text: str, user_id: str = "lars", metadata: Dict = None, client: MemoryClient = None) -> bool:
    """Upload a single memory to Mem0 cloud using the same format as generate_profile.py."""
    if not client:
        client = get_mem0_client()
        if not client:
            return False
    
    # Build metadata following generate_profile.py format
    memory_metadata = {
        "source": "local_sync",  # Similar to "profile_generation" in generate_profile
        "category": "conversational"  # Default category
    }
    
    # Add type and tags if available in metadata
    if metadata:
        # Map metadata fields to match generate_profile format
        if metadata.get("type"):
            memory_metadata["type"] = metadata["type"]
            memory_metadata["category"] = metadata["type"]  # Use type as category too
        
        if metadata.get("tags"):
            memory_metadata["tags"] = metadata["tags"]
        
        # Add timestamp if available
        if metadata.get("timestamp") or metadata.get("original_timestamp"):
            memory_metadata["timestamp"] = metadata.get("timestamp") or metadata.get("original_timestamp")
        
        # Handle special cases
        if metadata.get("is_summary"):
            memory_metadata["type"] = "summary"
            memory_metadata["category"] = "summary"
    
    try:
        # Use the same message format as generate_profile.py
        messages = [{"role": "user", "content": text}]
        result = client.add(messages, user_id=user_id, metadata=memory_metadata)
        return True
    except Exception as e:
        # Check for rate limiting
        if "429" in str(e) or "rate" in str(e).lower():
            print(f"   ⏸️  Rate limited, waiting...")
            time.sleep(5)
        else:
            print(f"   ❌ Error: {e}")
        return False

def main():
    """Main sync function."""
    import sys
    
    # Configure logging similar to generate_profile.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    
    print("🔄 Mem0 Memory Sync Tool")
    print("="*60)
    
    # Validate configuration
    if not validate_config():
        return
    
    # Get agent name from command line or default
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "lars"
    print(f"🤖 Agent: {agent_name}")
    
    # Load local memories
    local_memories = load_local_memories(agent_name)
    if not local_memories:
        return
    
    # Get cloud memories
    cloud_memories = get_cloud_memories(agent_name)
    # Handle both dict and string formats from Mem0
    cloud_memory_texts = set()
    for mem in cloud_memories:
        if isinstance(mem, dict):
            # Try both 'memory' and 'text' keys
            text = mem.get("memory", mem.get("text", ""))
        elif isinstance(mem, str):
            text = mem
        else:
            text = str(mem)
        if text:
            cloud_memory_texts.add(text)
    
    # Find memories to upload (excluding summaries)
    memories_to_upload = []
    for mem in local_memories:
        text = mem.get("text", "").strip()
        
        # Skip empty memories and summaries
        if not text or text.startswith("(summary)"):
            continue
            
        # Check if already in cloud (fuzzy match for first 100 chars)
        text_preview = text[:100]
        is_duplicate = any(
            text_preview in cloud_text 
            for cloud_text in cloud_memory_texts
        )
        
        if not is_duplicate:
            memories_to_upload.append({
                "text": text,
                "timestamp": mem.get("timestamp"),
                "is_summary": mem.get("is_summary", False)
            })
    
    # Display summary
    print(f"\n📊 Memory Analysis:")
    print(f"  📁 Local memories: {len(local_memories)}")
    print(f"  ☁️  Cloud memories: {len(cloud_memories)}")
    print(f"  📤 To upload: {len(memories_to_upload)}")
    
    if not memories_to_upload:
        print("\n✅ All memories are already synced!")
        return
    
    # Show sample of memories to upload
    print(f"\n📝 Sample memories to upload:")
    for i, mem in enumerate(memories_to_upload[:3], 1):
        preview = mem["text"][:80] + "..." if len(mem["text"]) > 80 else mem["text"]
        print(f"  {i}. {preview}")
    if len(memories_to_upload) > 3:
        print(f"  ... and {len(memories_to_upload) - 3} more")
    
    # Confirm upload (auto-confirm if --yes flag is provided)
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv
    
    if not auto_confirm:
        print(f"\n⚠️  Ready to upload {len(memories_to_upload)} memories to Mem0")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Upload cancelled")
            return
    else:
        print(f"\n✅ Auto-confirming upload of {len(memories_to_upload)} memories...")
    
    # Upload memories with progress tracking
    print(f"\n📤 Starting upload...")
    success_count = 0
    fail_count = 0
    
    # Initialize client once for all uploads
    client = get_mem0_client()
    if not client:
        print("❌ Failed to initialize Mem0 client")
        return
    
    for i, memory in enumerate(memories_to_upload, 1):
        # Progress indicator
        progress = (i / len(memories_to_upload)) * 100
        print(f"  [{i}/{len(memories_to_upload)}] ({progress:.1f}%) Uploading...", end="")
        
        # Categorize the memory based on content
        categorization = categorize_memory(memory["text"])
        
        # Prepare metadata following generate_profile.py format
        metadata = {
            "type": categorization["type"],
            "tags": categorization["tags"],
            "source": "local_sync",
            "category": categorization["type"]  # Use type as category
        }
        
        # Add timestamp if available
        if memory.get("timestamp"):
            metadata["timestamp"] = memory["timestamp"]
        
        # Handle summary memories specially
        if memory.get("is_summary"):
            metadata["type"] = "summary"
            metadata["category"] = "summary"
            metadata["tags"] = ["summary", "consolidated"]
        
        # Upload using the client
        if upload_memory(memory["text"], agent_name, metadata, client):
            success_count += 1
            print(" ✅")
        else:
            fail_count += 1
            print(" ❌")
        
        # Rate limiting pause every 20 uploads (like generate_profile.py)
        if i % 20 == 0 and i < len(memories_to_upload):
            print("  ⏸️  Pausing for rate limits...")
            time.sleep(2)
    
    # Final report
    print("\n" + "="*60)
    print("🎉 Sync Complete!")
    print(f"  ✅ Successfully uploaded: {success_count}")
    if fail_count > 0:
        print(f"  ❌ Failed: {fail_count}")
    
    # Verify new count
    print("\n🔍 Verifying...")
    new_cloud_memories = get_cloud_memories(agent_name)
    print(f"  📊 New cloud total: {len(new_cloud_memories)} memories")
    
    # Show memory breakdown by type (like generate_profile.py does)
    if new_cloud_memories:
        type_counts = {}
        for mem in new_cloud_memories:
            if isinstance(mem, dict):
                mem_type = mem.get('metadata', {}).get('type', 'unknown')
            else:
                mem_type = 'unknown'
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        if type_counts:
            print("  📊 Memory breakdown by type:")
            for mem_type, count in type_counts.items():
                print(f"    - {mem_type}: {count} memories")
    
    # Save upload log
    log_file = Path(f"mem0_sync_{agent_name}_{int(time.time())}.log")
    with open(log_file, 'w') as f:
        json.dump({
            "agent": agent_name,
            "timestamp": time.time(),
            "local_count": len(local_memories),
            "cloud_before": len(cloud_memories),
            "cloud_after": len(new_cloud_memories),
            "uploaded": success_count,
            "failed": fail_count
        }, f, indent=2)
    print(f"  📝 Log saved to: {log_file}")

if __name__ == "__main__":
    main()
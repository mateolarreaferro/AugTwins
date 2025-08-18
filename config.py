"""
Configuration module - Single source of truth for all API keys and settings.
Loads from environment variables with .env file support.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Required API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()

# Optional API keys (for advanced features)
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()
MEM0_ORG_ID = os.getenv("MEM0_ORG_ID", "").strip()
MEM0_PROJECT_ID = os.getenv("MEM0_PROJECT_ID", "").strip()

# Legacy support for alternative environment variable names
if not ELEVEN_API_KEY:
    ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()

def validate_required_keys():
    """Validate that required API keys are set."""
    missing_keys = []
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if not ELEVEN_API_KEY:
        missing_keys.append("ELEVEN_API_KEY (or ELEVENLABS_API_KEY)")
    
    if missing_keys:
        print(f"❌ Error: Missing required API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables or add them to your .env file.")
        print("See .env.example for the required format.")
        sys.exit(1)

def validate_optional_keys():
    """Check optional keys and warn if missing."""
    if not all([MEM0_API_KEY, MEM0_ORG_ID, MEM0_PROJECT_ID]):
        print("⚠️  Warning: Mem0 API keys not configured. Advanced memory features will be disabled.")
        print("Set MEM0_API_KEY, MEM0_ORG_ID, and MEM0_PROJECT_ID to enable Mem0 integration.")

def validate_all_keys():
    """Validate all configuration."""
    validate_required_keys()
    validate_optional_keys()

# Optional: Auto-validate on import (uncomment if you want strict validation)
# validate_all_keys() 
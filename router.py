def pick_model(mode: str) -> dict:
    """
    Select model configuration based on session mode.
    
    Args:
        mode: Session mode ("conversation" or "storytelling")
        
    Returns:
        Dictionary containing model and max_completion_tokens configuration
    """
    if mode == "conversation":
        return {"model": "gpt-4o-mini", "max_completion_tokens": 150}  # Using GPT-4 for reliable conversation
    if mode == "storytelling":
        return {"model": "gpt-5-mini", "max_completion_tokens": 2048}  # Keep GPT-5 for storytelling with much higher limit
    return {"model": "gpt-4o-mini", "max_completion_tokens": 150}
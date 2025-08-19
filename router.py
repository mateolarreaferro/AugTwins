def pick_model(mode: str) -> dict:
    """
    Select model configuration based on session mode.
    
    Args:
        mode: Session mode ("conversation" or "storytelling")
        
    Returns:
        Dictionary containing model and max_tokens configuration
    """
    if mode == "conversation":
        return {"model": "gpt-5-nano", "max_tokens": 128}
    if mode == "storytelling":
        return {"model": "gpt-5-mini", "max_tokens": 512}
    return {"model": "gpt-5-nano", "max_tokens": 128}
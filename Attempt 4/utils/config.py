"""
Configuration management for the application.
"""

import json
from pathlib import Path

def load_config():
    """Load application configuration."""
    config_path = Path("config.json")
    if not config_path.exists():
        return get_default_config()
    
    with open(config_path, "r") as f:
        return json.load(f)

def get_default_config():
    """Return default configuration."""
    return {
        "search": {
            "max_results": 10,
            "timeout": 30
        },
        "coach": {
            "suggestions_limit": 5
        }
    } 
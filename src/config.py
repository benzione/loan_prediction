import configparser
import os
from pathlib import Path

def get_config():
    """Reads and returns the application configuration."""
    config = configparser.ConfigParser()
    
    # Find config.ini in the project root, regardless of current working directory
    current_file_path = Path(__file__).parent.absolute()
    project_root = current_file_path.parent  # Go up from src/ to project root
    config_path = project_root / 'config.ini'
    
    if not config_path.exists():
        # Try alternative paths
        alternative_paths = [
            Path('config.ini'),  # Current directory
            Path('../config.ini'),  # Parent directory
            Path('../../config.ini'),  # Grandparent directory
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                config_path = alt_path
                break
        else:
            raise FileNotFoundError(f"config.ini not found. Searched: {config_path} and alternatives")
    
    config.read(config_path)
    
    # Verify that required sections exist
    required_sections = ['snowflake', 'model', 'business']
    for section in required_sections:
        if not config.has_section(section):
            raise configparser.NoSectionError(f"Required section '{section}' not found in config.ini")
    
    print(f"✅ Configuration loaded from: {config_path}")
    return config 
"""
Environment setup script for Bounce Loan Prediction package.
Run this at the beginning of notebooks or scripts to ensure proper imports.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Set up the environment for running the bounce loan prediction package."""
    
    # Get the project root directory
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")
    
    # Check if .env file exists
    env_file = project_root / '.env'
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your Snowflake credentials.")
        print("Example:")
        print("SNOWFLAKE_USER=your_username")
        print("SNOWFLAKE_PASSWORD=your_password")
        print("SNOWFLAKE_ACCOUNT=your_account")
        print("SNOWFLAKE_WAREHOUSE=your_warehouse")
        print("SNOWFLAKE_DATABASE=your_database")
        print("SNOWFLAKE_SCHEMA=your_schema")
    
    # Check if config.ini exists and test configuration loading
    config_file = project_root / 'config.ini'
    if not config_file.exists():
        print("⚠️  Warning: config.ini file not found!")
    else:
        print("✅ Configuration file found")
        
        # Test configuration loading
        try:
            from src.config import get_config
            config = get_config()
            print(f"✅ Configuration loaded successfully with sections: {list(config.sections())}")
        except Exception as e:
            print(f"⚠️  Warning: Configuration loading failed: {e}")
    
    print("✅ Environment setup complete!")
    return project_root

if __name__ == "__main__":
    setup_environment() 
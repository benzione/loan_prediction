#!/usr/bin/env python3
"""
Script to create .env file with Snowflake credentials for the Bounce project.
Run this script to set up your environment variables automatically.
"""

import os
from pathlib import Path

def create_env_file():
    """Creates the .env file with Snowflake credentials."""
    
    env_content = """SNOWFLAKE_USER=bounce_guest
SNOWFLAKE_PASSWORD=masterofanalytics
SNOWFLAKE_ACCOUNT=fc26424.us-east-2.aws
SNOWFLAKE_WAREHOUSE=BOUNCE_GUEST
SNOWFLAKE_DATABASE=BOUNCE_ASSIGNMENT
SNOWFLAKE_SCHEMA=PUBLIC
"""
    
    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    
    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file}")
        response = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if response != 'y':
            print("❌ Operation cancelled.")
            return False
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Successfully created .env file at {env_file}")
        print("📋 Environment variables set:")
        print("   - SNOWFLAKE_USER=bounce_guest")
        print("   - SNOWFLAKE_PASSWORD=masterofanalytics") 
        print("   - SNOWFLAKE_ACCOUNT=fc26424.us-east-2.aws")
        print("   - SNOWFLAKE_WAREHOUSE=BOUNCE_GUEST")
        print("   - SNOWFLAKE_DATABASE=BOUNCE_ASSIGNMENT")
        print("   - SNOWFLAKE_SCHEMA=PUBLIC")
        print("")
        print("🚀 You can now run the notebooks!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up Snowflake environment variables...")
    success = create_env_file()
    if success:
        print("\n🎉 Setup complete! Try running the notebook again.")
    else:
        print("\n💡 Alternative: Copy .env.example to .env manually:") 
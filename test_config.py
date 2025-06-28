"""
Test script to verify configuration loading is working correctly.
Run this from the project root to test the config system.
"""

from src.config import get_config

def test_config():
    """Test configuration loading."""
    try:
        print("Testing configuration loading...")
        config = get_config()
        
        print(f"✅ Config sections found: {list(config.sections())}")
        
        # Test snowflake section
        print(f"📊 Snowflake table: {config['snowflake']['table_name']}")
        
        # Test model section
        print(f"🎯 Target column: {config['model']['target_column']}")
        print(f"📏 Test size: {config['model']['test_size']}")
        print(f"🎲 Random state: {config['model']['random_state']}")
        
        # Test business section
        print(f"📈 Confidence level: {config['business']['confidence_level']}")
        
        print("✅ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    test_config() 
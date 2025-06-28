import pytest
from sqlalchemy.engine import Engine
from src.data.snowflake_connector import get_snowflake_engine

def test_get_snowflake_engine():
    """Tests the creation of a Snowflake engine."""
    try:
        engine = get_snowflake_engine()
        assert isinstance(engine, Engine)
        # Try to connect
        connection = engine.connect()
        connection.close()
    except Exception as e:
        pytest.fail(f"Snowflake connection failed: {e}") 
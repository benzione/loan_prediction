import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
from src.utils import get_logger
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)
logger = get_logger(__name__)

def get_snowflake_engine():
    """Creates and returns a SQLAlchemy engine for Snowflake."""
    try:
        # Get environment variables
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        
        # Validate that all required environment variables are set
        required_vars = {
            'SNOWFLAKE_USER': user,
            'SNOWFLAKE_PASSWORD': password,
            'SNOWFLAKE_ACCOUNT': account,
            'SNOWFLAKE_DATABASE': database,
            'SNOWFLAKE_SCHEMA': schema,
            'SNOWFLAKE_WAREHOUSE': warehouse
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            logger.error(f"Please check your .env file at: {env_path}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info(f"Connecting to Snowflake account: {account}")
        logger.info(f"Using database: {database}, schema: {schema}, warehouse: {warehouse}")
        
        engine = create_engine(
            'snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}'.format(
                user=user,
                password=password,
                account=account,
                database=database,
                schema=schema,
                warehouse=warehouse,
            )
        )
        logger.info("Successfully created Snowflake engine.")
        return engine
    except Exception as e:
        logger.error(f"Error creating Snowflake engine: {e}")
        raise

def fetch_data(engine, table_name):
    """Fetches data from a Snowflake table."""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        logger.info(f"Successfully fetched data from {table_name}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise 
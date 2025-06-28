import pandas as pd
import numpy as np
import joblib
from src.utils import get_logger

logger = get_logger(__name__)

def load_model(filepath):
    """
    Loads a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def make_predictions(model, X):
    """
    Makes predictions using a trained model.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features for prediction
    
    Returns:
        tuple: Predicted labels and probabilities
    """
    logger.info("Making predictions...")
    
    try:
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        logger.info(f"Predictions completed for {len(X)} samples")
        
        return predictions, probabilities
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def calculate_portfolio_value(probabilities, loan_amounts, confidence_threshold=0.5):
    """
    Calculates the expected value of a loan portfolio based on repayment probabilities.
    
    Args:
        probabilities (array): Predicted repayment probabilities
        loan_amounts (array): Loan amounts
        confidence_threshold (float): Minimum probability threshold for investment
    
    Returns:
        dict: Portfolio analysis results
    """
    logger.info("Calculating portfolio value...")
    
    # Filter loans above confidence threshold
    high_confidence_mask = probabilities >= confidence_threshold
    
    # Calculate expected values
    expected_recovery = probabilities * loan_amounts
    total_expected_recovery = np.sum(expected_recovery)
    
    # High confidence loans
    high_confidence_loans = np.sum(high_confidence_mask)
    high_confidence_amount = np.sum(loan_amounts[high_confidence_mask])
    high_confidence_expected = np.sum(expected_recovery[high_confidence_mask])
    
    results = {
        'total_loans': len(probabilities),
        'total_loan_amount': np.sum(loan_amounts),
        'total_expected_recovery': total_expected_recovery,
        'high_confidence_loans': high_confidence_loans,
        'high_confidence_amount': high_confidence_amount,
        'high_confidence_expected_recovery': high_confidence_expected,
        'confidence_threshold': confidence_threshold
    }
    
    logger.info(f"Portfolio analysis completed. Total expected recovery: ${total_expected_recovery:,.2f}")
    
    return results 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from src.utils import get_logger

logger = get_logger(__name__)

def train_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    """
    Trains a machine learning model.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        model_type (str): Type of model to train ('random_forest' or 'logistic_regression')
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: Trained model, test features, test target, predictions
    """
    logger.info(f"Training {model_type} model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("Model training completed.")
    
    return model, X_test, y_test, y_pred, y_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Evaluates model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    classification_rep = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'auc_score': auc_score,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix
    }
    
    logger.info(f"AUC Score: {auc_score:.4f}")
    print("Classification Report:")
    print(classification_rep)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return metrics

def save_model(model, filepath):
    """
    Saves the trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}") 
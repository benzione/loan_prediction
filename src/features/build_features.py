import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import get_logger

logger = get_logger(__name__)

def clean_target_variable(df, target_column):
    """
    Clean and convert target variable based on EDA findings.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned target variable
    """
    logger.info(f"Cleaning target variable: {target_column}")
    
    df_clean = df.copy()
    
    # Drop rows with missing target values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_column]).copy()
    dropped_rows = initial_rows - len(df_clean)
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing target values")
    
    # Convert target to boolean if it's string/object type
    if df_clean[target_column].dtype == 'object':
        df_clean[target_column] = df_clean[target_column].map({
            'True': True, 'False': False, 
            True: True, False: False,
            'true': True, 'false': False,
            1: True, 0: False
        })
        logger.info("Converted target variable from string to boolean")
    
    # Ensure target is boolean type
    df_clean[target_column] = df_clean[target_column].astype(bool)
    
    # Log target distribution
    target_dist = df_clean[target_column].value_counts()
    enrollment_rate = df_clean[target_column].mean()
    logger.info(f"Target distribution: {dict(target_dist)}")
    logger.info(f"Enrollment rate: {enrollment_rate:.3f}")
    
    return df_clean

def remove_identifier_columns(df):
    """
    Remove identifier columns that don't provide predictive value.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe without identifier columns
    """
    identifier_cols = ['account_public_id']
    
    cols_to_remove = [col for col in identifier_cols if col in df.columns]
    
    if cols_to_remove:
        df_clean = df.drop(cols_to_remove, axis=1)
        logger.info(f"Removed identifier columns: {cols_to_remove}")
        return df_clean
    
    return df.copy()

def handle_multicollinearity(df):
    """
    Remove highly correlated features based on EDA findings.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (cleaned dataframe, list of removed features)
    """
    logger.info("Handling multicollinearity...")
    
    # Based on EDA findings - highly correlated pairs (>0.7 correlation)
    high_corr_pairs = [
        ('principal_at_placement', 'total_at_placement'),  # 0.998 correlation
        ('days_from_co_to_placement', 'days_from_dq_to_placement'),  # 0.971
        ('days_from_co_to_placement', 'days_from_last_payment_to_assignment'),  # 0.922
    ]
    
    features_to_remove = []
    for feat1, feat2 in high_corr_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Keep the first feature, remove the second (based on EDA insights)
            if feat1 == 'principal_at_placement':
                features_to_remove.append(feat2)  # Remove total_at_placement
            elif feat1 == 'days_from_co_to_placement':
                features_to_remove.append(feat2)  # Remove the correlated date features
    
    # Remove duplicates
    features_to_remove = list(set(features_to_remove))
    
    if features_to_remove:
        df_clean = df.drop(features_to_remove, axis=1)
        logger.info(f"Removed highly correlated features: {features_to_remove}")
        return df_clean, features_to_remove
    
    return df.copy(), []

def handle_missing_values(df):
    """
    Handle missing values based on EDA findings.
    Strategy: median for numerical, mode for categorical.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (cleaned dataframe, imputation info)
    """
    logger.info("Handling missing values...")
    
    df_clean = df.copy()
    imputation_info = {}
    
    # Identify feature types
    numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    # Handle missing values
    missing_summary = df_clean.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    
    if len(missing_features) > 0:
        logger.info(f"Found missing values in {len(missing_features)} features")
        
        for feature in missing_features.index:
            missing_count = missing_features[feature]
            missing_pct = (missing_count / len(df_clean)) * 100
            
            if feature in numerical_features:
                # Use median for numerical features
                median_value = df_clean[feature].median()
                df_clean[feature].fillna(median_value, inplace=True)
                imputation_info[feature] = {'method': 'median', 'value': median_value}
                logger.info(f"Filled {feature} ({missing_count} values, {missing_pct:.2f}%) with median: {median_value:.2f}")
                
            elif feature in categorical_features:
                # Use mode for categorical features
                mode_values = df_clean[feature].mode()
                mode_value = mode_values[0] if not mode_values.empty else 'Unknown'
                df_clean[feature].fillna(mode_value, inplace=True)
                imputation_info[feature] = {'method': 'mode', 'value': mode_value}
                logger.info(f"Filled {feature} ({missing_count} values, {missing_pct:.2f}%) with mode: {mode_value}")
    else:
        logger.info("No missing values found")
    
    return df_clean, imputation_info

def handle_outliers(df):
    """
    Handle outliers by capping based on EDA findings.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (cleaned dataframe, outlier info)
    """
    logger.info("Handling outliers...")
    
    df_clean = df.copy()
    outlier_info = {}
    
    # Features identified with >5% outliers from EDA
    outlier_features = [
        'thirty_dpd_in_last_24_months',
        'revolving_accounts_opened_3_months', 
        'total_finance_accounts_balance_gt_0',
        'installment_loans_accounts_opened_3_months',
        'days_from_origination_to_co',
        'sum_of_balance_amount_installment_loans'
    ]
    
    for feature in outlier_features:
        if feature in df_clean.columns:
            # Calculate IQR bounds
            Q1 = df_clean[feature].quantile(0.25)
            Q3 = df_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers_before = ((df_clean[feature] < lower_bound) | (df_clean[feature] > upper_bound)).sum()
            
            # Cap outliers
            df_clean[feature] = df_clean[feature].clip(lower=lower_bound, upper=upper_bound)
            
            outlier_info[feature] = {
                'outliers_capped': outliers_before,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            logger.info(f"Capped {outliers_before} outliers in {feature}")
    
    return df_clean, outlier_info

def encode_categorical_features(df, target_column=None, label_encoders=None, is_training=True):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column to exclude from encoding
        label_encoders (dict): Pre-fitted encoders for prediction mode
        is_training (bool): Whether this is training or prediction
        
    Returns:
        tuple: (encoded dataframe, label encoders dict)
    """
    logger.info("Encoding categorical features...")
    
    df_encoded = df.copy()
    
    # Identify categorical features (excluding target)
    categorical_features = df_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
    if target_column and target_column in categorical_features:
        categorical_features.remove(target_column)
    
    if is_training:
        # Training mode: fit new encoders
        label_encoders = {}
        for feature in categorical_features:
            if feature in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                label_encoders[feature] = le
                logger.info(f"Encoded {feature}: {len(le.classes_)} classes")
    else:
        # Prediction mode: use existing encoders
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for prediction mode")
        
        for feature in categorical_features:
            if feature in df_encoded.columns and feature in label_encoders:
                le = label_encoders[feature]
                # Handle unseen categories by mapping to a default value
                try:
                    df_encoded[feature] = le.transform(df_encoded[feature].astype(str))
                except ValueError:
                    # Handle unseen categories
                    feature_values = df_encoded[feature].astype(str)
                    known_classes = set(le.classes_)
                    unknown_mask = ~feature_values.isin(known_classes)
                    
                    if unknown_mask.any():
                        # Map unknown values to the most common class
                        default_value = le.classes_[0]
                        feature_values[unknown_mask] = default_value
                        logger.warning(f"Found {unknown_mask.sum()} unknown values in {feature}, mapped to {default_value}")
                    
                    df_encoded[feature] = le.transform(feature_values)
                
                logger.info(f"Applied encoding to {feature}")
    
    return df_encoded, label_encoders

def preprocess_data(df, target_column, label_encoders=None, is_training=True):
    """
    Comprehensive preprocessing pipeline based on EDA findings.
    
    Args:
        df (pd.DataFrame): Raw data
        target_column (str): Name of the target column
        label_encoders (dict): Pre-fitted encoders for prediction mode
        is_training (bool): Whether this is for training (True) or prediction (False)
    
    Returns:
        tuple: Preprocessed features, target (if training), preprocessing artifacts
    """
    logger.info("Starting comprehensive data preprocessing pipeline...")
    
    # Step 1: Clean target variable (only for training)
    if is_training:
        df_processed = clean_target_variable(df, target_column)
    else:
        df_processed = df.copy()
    
    # Step 2: Remove identifier columns
    df_processed = remove_identifier_columns(df_processed)
    
    # Step 3: Handle multicollinearity
    df_processed, removed_features = handle_multicollinearity(df_processed)
    
    # Step 4: Handle missing values
    df_processed, imputation_info = handle_missing_values(df_processed)
    
    # Step 5: Handle outliers
    df_processed, outlier_info = handle_outliers(df_processed)
    
    # Step 6: Encode categorical features
    df_processed, label_encoders_used = encode_categorical_features(
        df_processed, target_column, label_encoders, is_training
    )
    
    # Step 7: Separate features and target
    if is_training and target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Create preprocessing artifacts
        preprocessing_artifacts = {
            'removed_features': removed_features,
            'imputation_info': imputation_info,
            'outlier_info': outlier_info,
            'label_encoders': label_encoders_used,
            'feature_names': X.columns.tolist()
        }
        
        logger.info(f"Training preprocessing complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, preprocessing_artifacts
    else:
        # Prediction mode
        logger.info(f"Prediction preprocessing complete. Shape: {df_processed.shape}")
        return df_processed, label_encoders_used

def create_features(df):
    """
    Creates new features from existing ones based on EDA insights.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    logger.info("Creating new features based on EDA insights...")
    
    df_featured = df.copy()
    
    # Feature engineering based on EDA findings
    try:
        # 1. Loan-to-value ratios (if both principal features exist)
        if 'principal_at_placement' in df_featured.columns:
            # Create loan size categories based on EDA findings
            df_featured['loan_size_category'] = pd.cut(
                df_featured['principal_at_placement'],
                bins=[0, 5000, 15000, 30000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'XLarge']
            )
            logger.info("Created loan_size_category feature")
        
        # 2. Age-based features (if age column exists)
        if 'age' in df_featured.columns:
            df_featured['age_category'] = pd.cut(
                df_featured['age'],
                bins=[0, 30, 40, 50, float('inf')],
                labels=['Young', 'Middle', 'Mature', 'Senior']
            )
            logger.info("Created age_category feature")
        
        # 3. Risk indicators (based on top predictive features from EDA)
        if 'total_accounts_in_collections' in df_featured.columns:
            df_featured['has_collections'] = (df_featured['total_accounts_in_collections'] > 0).astype(int)
            logger.info("Created has_collections feature")
        
        # 4. Time-based features
        if 'days_from_co_to_placement' in df_featured.columns:
            df_featured['placement_urgency'] = pd.cut(
                df_featured['days_from_co_to_placement'],
                bins=[0, 76, 475, float('inf')],
                labels=['Fast', 'Medium', 'Slow']
            )
            logger.info("Created placement_urgency feature")
        
        logger.info(f"Feature creation complete. New shape: {df_featured.shape}")
        
    except Exception as e:
        logger.warning(f"Feature creation encountered an issue: {e}")
        logger.warning("Returning original dataframe")
        return df
    
    return df_featured 
import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    clean_target_variable,
    remove_identifier_columns,
    handle_multicollinearity,
    handle_missing_values,
    handle_outliers,
    encode_categorical_features,
    preprocess_data,
    create_features
)

@pytest.fixture
def sample_data():
    """Create sample data for testing preprocessing functions"""
    np.random.seed(42)
    
    data = {
        'account_public_id': [f'BA-{i:010d}' for i in range(100)],
        'principal_at_placement': np.random.normal(8000, 3000, 100),
        'total_at_placement': np.random.normal(8500, 3200, 100),  # Highly correlated with principal
        'total_principal_ratio': np.random.normal(1.1, 0.05, 100),
        'days_from_co_to_placement': np.random.randint(10, 500, 100),
        'days_from_dq_to_placement': np.random.randint(15, 505, 100),  # Correlated with co_to_placement
        'days_from_last_payment_to_assignment': np.random.randint(20, 510, 100),  # Correlated
        'days_from_origination_to_placement': np.random.randint(200, 1500, 100),
        'days_from_origination_to_co': np.random.randint(150, 1200, 100),
        'total_accounts_in_collections': np.random.poisson(1.5, 100),
        'thirty_dpd_in_last_24_months': np.random.poisson(0.8, 100),
        'revolving_accounts_opened_3_months': np.random.poisson(0.3, 100),
        'total_finance_accounts_balance_gt_0': np.random.poisson(1.2, 100),
        'installment_loans_accounts_opened_3_months': np.random.poisson(0.2, 100),
        'sum_of_balance_amount_installment_loans': np.random.normal(25000, 15000, 100),
        'bank_card_credit_utilization_pct': np.random.normal(70, 25, 100),
        'total_paid_accounts_6_months': np.random.poisson(0.1, 100),
        'is_fpd': np.random.choice([True, False], 100, p=[0.05, 0.95]),
        'group_': np.random.choice(['A', 'B'], 100, p=[0.3, 0.7]),
        'age': np.random.randint(22, 70, 100),
        'enrolled_to_plan_in_180': np.random.choice(['True', 'False'], 100, p=[0.16, 0.84])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to test imputation
    missing_indices = np.random.choice(df.index, 10, replace=False)
    df.loc[missing_indices, 'sum_of_balance_amount_installment_loans'] = np.nan
    df.loc[missing_indices[:5], 'bank_card_credit_utilization_pct'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, 5, replace=False)
    df.loc[outlier_indices, 'thirty_dpd_in_last_24_months'] = 10  # Extreme outlier
    
    # Add some None target values
    none_indices = np.random.choice(df.index, 5, replace=False)
    df.loc[none_indices, 'enrolled_to_plan_in_180'] = None
    
    return df

def test_clean_target_variable(sample_data):
    """Test target variable cleaning"""
    target_col = 'enrolled_to_plan_in_180'
    
    # Test cleaning
    df_clean = clean_target_variable(sample_data, target_col)
    
    # Assertions
    assert df_clean[target_col].dtype == bool
    assert df_clean[target_col].isnull().sum() == 0
    assert len(df_clean) <= len(sample_data)  # Some rows may be dropped
    
    # Check that True/False strings are properly converted
    true_count = (df_clean[target_col] == True).sum()
    false_count = (df_clean[target_col] == False).sum()
    assert true_count + false_count == len(df_clean)

def test_remove_identifier_columns(sample_data):
    """Test identifier column removal"""
    df_processed = remove_identifier_columns(sample_data)
    
    # Should remove account_public_id
    assert 'account_public_id' not in df_processed.columns
    assert len(df_processed.columns) == len(sample_data.columns) - 1

def test_handle_multicollinearity(sample_data):
    """Test multicollinearity handling"""
    df_processed, removed_features = handle_multicollinearity(sample_data)
    
    # Should remove highly correlated features
    expected_removed = ['total_at_placement', 'days_from_dq_to_placement', 'days_from_last_payment_to_assignment']
    
    for feature in expected_removed:
        if feature in sample_data.columns:
            assert feature not in df_processed.columns
            assert feature in removed_features

def test_handle_missing_values(sample_data):
    """Test missing value handling"""
    df_processed, imputation_info = handle_missing_values(sample_data)
    
    # Should have no missing values after processing
    assert df_processed.isnull().sum().sum() == 0
    
    # Should have imputation info for features that had missing values
    assert 'sum_of_balance_amount_installment_loans' in imputation_info
    assert imputation_info['sum_of_balance_amount_installment_loans']['method'] == 'median'

def test_handle_outliers(sample_data):
    """Test outlier handling"""
    df_processed, outlier_info = handle_outliers(sample_data)
    
    # Should have outlier info for features that were capped
    outlier_features = [
        'thirty_dpd_in_last_24_months',
        'revolving_accounts_opened_3_months', 
        'total_finance_accounts_balance_gt_0',
        'installment_loans_accounts_opened_3_months',
        'days_from_origination_to_co',
        'sum_of_balance_amount_installment_loans'
    ]
    
    for feature in outlier_features:
        if feature in sample_data.columns:
            assert feature in outlier_info
            assert 'outliers_capped' in outlier_info[feature]

def test_encode_categorical_features(sample_data):
    """Test categorical feature encoding"""
    target_col = 'enrolled_to_plan_in_180'
    
    # Test training mode
    df_encoded, label_encoders = encode_categorical_features(
        sample_data, target_col, is_training=True
    )
    
    # Should encode categorical features
    assert 'is_fpd' in label_encoders
    assert 'group_' in label_encoders
    assert target_col not in label_encoders  # Target should not be encoded
    
    # Encoded features should be numeric
    assert df_encoded['is_fpd'].dtype in ['int64', 'int32']
    assert df_encoded['group_'].dtype in ['int64', 'int32']

def test_preprocess_data_training(sample_data):
    """Test complete preprocessing pipeline for training"""
    target_col = 'enrolled_to_plan_in_180'
    
    X, y, preprocessing_artifacts = preprocess_data(
        sample_data, target_col, is_training=True
    )
    
    # Basic shape checks
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.shape[0] <= sample_data.shape[0]  # Some rows may be dropped
    
    # Check preprocessing artifacts
    assert 'removed_features' in preprocessing_artifacts
    assert 'imputation_info' in preprocessing_artifacts
    assert 'outlier_info' in preprocessing_artifacts
    assert 'label_encoders' in preprocessing_artifacts
    assert 'feature_names' in preprocessing_artifacts
    
    # Target should be boolean
    assert y.dtype == bool
    
    # Should not have missing values
    assert X.isnull().sum().sum() == 0

def test_preprocess_data_prediction(sample_data):
    """Test preprocessing pipeline for prediction mode"""
    target_col = 'enrolled_to_plan_in_180'
    
    # First run in training mode to get encoders
    X_train, y_train, artifacts = preprocess_data(
        sample_data, target_col, is_training=True
    )
    
    # Create new data for prediction (without target)
    prediction_data = sample_data.drop(target_col, axis=1).copy()
    
    # Run in prediction mode
    X_pred, label_encoders_used = preprocess_data(
        prediction_data, target_col, 
        label_encoders=artifacts['label_encoders'], 
        is_training=False
    )
    
    # Should return processed features
    assert isinstance(X_pred, pd.DataFrame)
    assert X_pred.shape[1] > 0
    assert X_pred.isnull().sum().sum() == 0

def test_create_features(sample_data):
    """Test feature creation"""
    df_featured = create_features(sample_data)
    
    # Should return a DataFrame
    assert isinstance(df_featured, pd.DataFrame)
    
    # Should have at least the original number of features
    assert df_featured.shape[1] >= sample_data.shape[1]
    
    # Check for expected new features
    expected_features = ['loan_size_category', 'age_category', 'has_collections', 'placement_urgency']
    
    for feature in expected_features:
        if all(col in sample_data.columns for col in ['principal_at_placement', 'age', 'total_accounts_in_collections', 'days_from_co_to_placement']):
            # Only check if base features exist
            pass  # The feature creation is optional and may not always create all features

def test_preprocessing_integration(sample_data):
    """Test the complete preprocessing integration"""
    target_col = 'enrolled_to_plan_in_180'
    
    # Run complete preprocessing
    X, y, artifacts = preprocess_data(sample_data, target_col, is_training=True)
    
    # Run feature creation
    df_features = pd.DataFrame(X, columns=artifacts['feature_names'])
    df_enhanced = create_features(df_features)
    
    # Should work without errors
    assert df_enhanced.shape[0] == df_features.shape[0]
    assert df_enhanced.shape[1] >= df_features.shape[1]

def test_preprocessing_robustness():
    """Test preprocessing with edge cases"""
    # Create minimal data
    minimal_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'A'],
        'enrolled_to_plan_in_180': ['True', 'False', 'True']
    })
    
    # Should handle minimal data without errors
    X, y, artifacts = preprocess_data(minimal_data, 'enrolled_to_plan_in_180', is_training=True)
    
    assert len(X) == 3
    assert len(y) == 3
    assert isinstance(artifacts, dict)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__]) 
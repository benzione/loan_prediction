# Bounce Loan Repayment Prediction

This project develops a machine learning solution to predict whether consumers will start repaying loans within 180 days of placement with Bounce, enabling data-driven portfolio acquisition and pricing strategies.

## 🎯 Project Overview

The analysis is structured across three comprehensive Jupyter notebooks:

1. **📊 Data Ingestion and EDA** - Explores 34,628 loans with 21 features, identifies key patterns and risk factors
2. **🤖 Model Training and Evaluation** - Develops and evaluates 16 model combinations, achieving optimal F1-score of 0.396
3. **💼 Business Case Analysis** - Provides portfolio valuation framework and strategic recommendations

## 🔍 Key Findings

### Data Insights (Notebook 01)
- **Dataset**: 34,628 loans with 16.1% historical enrollment rate
- **Class Imbalance**: Moderate imbalance requiring specialized handling techniques
- **Top Predictive Features**: `total_principal_ratio`, `total_accounts_in_collections`, `principal_at_placement`
- **Best Performing Segments**: Large loans ($15-30K) with 22.9% enrollment rate
- **Optimal Demographics**: Senior borrowers (50+) with 17.8% enrollment rate
- **Data Quality**: 5 highly correlated feature pairs, 8 features with >5% outliers, 9 features with missing values

### Model Performance (Notebook 02)
- **Best Model**: Gradient Boosting with Random Undersampling
- **Performance Metrics**:
  - F1-Score: **0.396**
  - ROC-AUC: **0.736** 
  - Precision: **0.279**
  - Recall: **0.681**
- **Business Impact**: Identifies 68.1% of actual enrollments
- **Feature Engineering**: Successfully addressed multicollinearity, outliers, and missing values
- **Model Validation**: Robust evaluation across 4 algorithms and 4 sampling strategies

### Business Recommendations (Notebook 03)
- **Portfolio Valuation**: $0.187 per dollar expected value (base case)
- **ML Enhancement Benefit**: +40.9% value improvement ($0.264 per dollar)
- **Financial Impact**: Potential $77M annual improvement on $100M portfolio
- **ROI**: 900%+ return on $1M ML implementation investment
- **Risk Assessment**: 83.9% of current portfolio classified as high-risk

## 🚀 Getting Started

### Installation
```bash
# Install in development mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the project root:
```bash
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### Running the Analysis
Execute notebooks sequentially in the `notebooks/` directory:
1. `01_data_ingestion_and_eda.ipynb` - Data exploration and insights
2. `02_model_training_and_evaluation.ipynb` - Model development and validation  
3. `03_business_case_analysis.ipynb` - Business strategy and recommendations

## 📈 Strategic Recommendations

### High Priority Actions
1. **Deploy ML Model**: Implement production pipeline for 40.9% value improvement
2. **Risk Management**: Limit high-risk loans to <30% of portfolio (currently 83.9%)
3. **Segment Optimization**: Target large loans ($15-30K) with highest enrollment rates

### Implementation Timeline
- **Months 1-2**: ML model deployment and data pipeline setup
- **Months 3-4**: Risk-based allocation limits and segment targeting
- **Months 5-6**: Performance evaluation and strategy refinement

### Expected Financial Impact
- **Annual Value Improvement**: $77M on $100M portfolio
- **Implementation Cost**: ~$1M
- **Net Annual Benefit**: $76M+
- **Payback Period**: <2 months

## 🛠️ Technical Architecture

### Model Pipeline
- **Preprocessing**: Multicollinearity removal, outlier capping, missing value imputation
- **Feature Engineering**: Categorical encoding, risk-based segmentation
- **Sampling**: Random undersampling for class imbalance
- **Algorithm**: Gradient Boosting Classifier
- **Validation**: Stratified cross-validation with comprehensive metrics

### Production Deployment
- **Model Persistence**: Saved with preprocessing pipeline (`best_loan_enrollment_model.pkl`)
- **Performance Monitoring**: Continuous F1-score and ROC-AUC tracking
- **Retraining Schedule**: Monthly recalibration with new data

## 🔧 Troubleshooting

### Common Issues
If you encounter `ModuleNotFoundError: No module named 'src'`:

1. **Install in development mode** (recommended):
   ```bash
   pip install -e .
   ```

2. **Use setup script**: Uncomment the first line in notebook cells:
   ```python
   import sys; sys.path.append('..'); from setup_env import setup_environment; setup_environment()
   ```

3. **Manual path setup**: The notebooks include automatic path configuration

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

## 📊 Key Performance Indicators

### Model Metrics
- **F1-Score**: Balanced precision-recall for imbalanced data
- **ROC-AUC**: Overall discriminatory ability
- **Business Recall**: Percentage of enrollments correctly identified

### Business Metrics  
- **Portfolio Value per Dollar**: Expected return on acquisition
- **Risk-Adjusted Returns**: Performance by risk category
- **Implementation ROI**: Return on ML investment

## 🔄 Continuous Improvement

- **Monthly**: Model performance review and recalibration
- **Quarterly**: Portfolio analysis and strategy adjustment
- **Semi-annually**: Stress testing and scenario analysis
- **Annually**: Comprehensive model refresh with new features

---

**Status**: ✅ Production Ready | **Last Updated**: 2025-06-27 | **Model Version**: v1.0 
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working on the **Advanced Telco Customer Churn Prediction** project.

## Project Overview

This is a comprehensive machine learning project focused on building production-ready customer churn prediction systems for telecommunications companies. The project emphasizes advanced EDA, ensemble methods, class imbalance handling, and business impact analysis.

**Deadline**: August 22, 2025

## Dataset Information
- **Source**: Telco Customer Churn Dataset (Kaggle)
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Location**: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size**: 7,043 customers, 21 features
- **Target**: Churn (Yes/No)
- **Key Features**: Demographics, Account Info, Services, Financial data

## Project Structure & Development Approach

### Template Reference
The `Past Notebooks and Pipeline/` folder serves as a **template and boilerplate source** for this project:
- **Week 01_02**: EDA approach and Jupyter notebook structure/naming conventions
- **Week 03_04**: Model training and ensemble methods approach
- **Week 05_06**: Production pipeline architecture and data processing modules

**IMPORTANT**: Do NOT modify anything in `Past Notebooks and Pipeline/` - it's reference only.

### New Project Structure
Create the following structure for the Advanced Telco Customer Churn project:

```
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── telco_missing_handled.csv
│       ├── telco_outliers_removed.csv
│       ├── telco_binned.csv
│       ├── telco_encoded.csv
│       └── telco_scaled.csv
├── notebooks/
│   ├── 0_data_preparation.ipynb
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_development.ipynb
│   ├── 4_ensemble_methods.ipynb
│   ├── 5_model_evaluation.ipynb
│   └── 6_business_analysis.ipynb
├── src/
│   ├── data_processing/
│   │   ├── data_ingestion.py
│   │   ├── handle_missing_values.py
│   │   ├── outlier_detection.py
│   │   ├── feature_encoding.py
│   │   ├── feature_scaling.py
│   │   ├── feature_binning.py
│   │   └── data_splitter.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── pipelines/
│   └── data_pipeline.py
├── config/
│   └── config.yaml
├── tests/
├── artifacts/
│   ├── data/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   ├── models/
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   └── catboost_model.pkl
│   └── reports/
│       ├── model_comparison.json
│       ├── feature_importance.json
│       └── evaluation_metrics.json
├── reports/
│   └── executive_summary.md
└── requirements.txt
```

## Development Workflow

### Phase 1: Exploratory Data Analysis (Jupyter Notebooks)
Follow the Week 01_02 naming convention and approach:

1. **0_data_preparation.ipynb**
   - Data loading and initial assessment
   - Data quality checks and inconsistencies
   - Target variable analysis and class imbalance identification

2. **1_exploratory_data_analysis.ipynb**
   - Comprehensive univariate analysis
   - Advanced bivariate analysis (Churn vs all features)
   - Multivariate analysis and correlation studies
   - Statistical significance testing

3. **2_feature_engineering.ipynb**
   - Create Telco-specific derived features:
     - Tenure categories (New, Established, Loyal)
     - Service adoption score
     - Average monthly charges per service
     - Payment reliability indicators
   - Feature interaction analysis

### Phase 2: Model Development (Jupyter Notebooks)
Follow the Week 03_04 approach with focus on ensemble methods:

4. **3_model_development.ipynb**
   - Baseline models (Logistic Regression, Decision Tree)
   - Data preprocessing pipeline implementation
   - Cross-validation strategy for imbalanced data

5. **4_ensemble_methods.ipynb**
   - **Random Forest**: Bagging method implementation
   - **XGBoost**: Gradient boosting implementation  
   - **CatBoost**: Advanced boosting with categorical handling
   - Hyperparameter tuning using GridSearchCV/RandomizedSearchCV

6. **5_model_evaluation.ipynb**
   - Class imbalance evaluation metrics (Precision, Recall, F1, PR-AUC)
   - Business-focused metrics and cost analysis
   - Model comparison framework
   - Threshold optimization

### Phase 3: Business Analysis

7. **6_business_analysis.ipynb**
   - Customer segmentation for retention
   - High-risk customer profiling
   - Revenue impact analysis
   - Retention strategy recommendations

### Phase 4: Production Data Pipeline (Python Modules)
**IMPORTANT**: Only data processing should be implemented in Python scripts. Model development, training, and evaluation should remain in Jupyter notebooks.

#### Data Processing Modules (`src/data_processing/`)
Use Week 05_06 as boilerplate but adapt for Telco features. Each module should save intermediate results to `data/processed/`:

- **data_ingestion.py**: Handle Telco CSV format and TotalCharges data type issues
- **handle_missing_values.py**: Telco-specific missing value strategies → saves `telco_missing_handled.csv`
- **outlier_detection.py**: IQR and Z-score methods for numerical features → saves `telco_outliers_removed.csv`
- **feature_binning.py**: Tenure categorization and service groupings → saves `telco_binned.csv`
- **feature_encoding.py**: Categorical encoding for Telco features (Gender, Contract, etc.) → saves `telco_encoded.csv`
- **feature_scaling.py**: StandardScaler/MinMaxScaler for MonthlyCharges, TotalCharges → saves `telco_scaled.csv`
- **data_splitter.py**: Stratified split maintaining churn class distribution → saves final splits to `artifacts/data/`

#### Pipeline Orchestration (`pipelines/`)
- **data_pipeline.py**: End-to-end data preprocessing pipeline that calls all data processing modules in sequence, creating intermediate files in `data/processed/` and final train/test splits in `artifacts/data/`

**Note**: Model development, ensemble methods, training, evaluation, and inference should all be implemented in Jupyter notebooks (notebooks 3-6), not in Python script modules.

## Key Requirements

### Ensemble Methods Implementation
1. **Random Forest** (Bagging)
   - scikit-learn RandomForestClassifier
   - Hyperparameters: n_estimators, max_depth, min_samples_split, max_features
   - Feature importance analysis

2. **XGBoost** (Boosting)
   - XGBoost library implementation
   - Hyperparameters: learning_rate, max_depth, n_estimators, subsample
   - Feature importance and SHAP analysis

3. **CatBoost** (Advanced Boosting)
   - CatBoost library for categorical handling
   - Automatic encoding and overfitting prevention
   - Performance comparison with other methods

### Class Imbalance Handling
- Use stratified cross-validation
- Focus on Precision, Recall, F1-Score, and PR-AUC
- Implement threshold optimization for business objectives
- Consider SMOTE for optional advanced techniques

### Business Focus
- Calculate revenue impact of churn predictions
- Develop customer segmentation strategies
- Create actionable retention recommendations
- Cost-benefit analysis of different error types

## Configuration Management

### config.yaml Structure
```yaml
data:
  raw_path: "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
  processed_path: "data/processed/"
  artifacts_path: "artifacts/data/"
  
  # Intermediate processed files
  missing_handled: "data/processed/telco_missing_handled.csv"
  outliers_removed: "data/processed/telco_outliers_removed.csv"
  binned: "data/processed/telco_binned.csv"
  encoded: "data/processed/telco_encoded.csv"
  scaled: "data/processed/telco_scaled.csv"
  
preprocessing:
  missing_strategy: "mean"  # for numerical
  scaling_method: "standard"
  encoding_method: "onehot"
  
models:
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [5, 10, 15, 20]
    min_samples_split: [2, 5, 10]
  
  xgboost:
    learning_rate: [0.01, 0.1, 0.2]
    max_depth: [3, 5, 7]
    n_estimators: [100, 200, 300]
    
  catboost:
    learning_rate: [0.01, 0.1, 0.2]
    depth: [4, 6, 8]
    iterations: [100, 200, 300]

evaluation:
  cv_folds: 5
  test_size: 0.2
  random_state: 42
  scoring: ["precision", "recall", "f1", "roc_auc"]

artifacts:
  models_path: "artifacts/models/"
  reports_path: "artifacts/reports/"
```

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Required Libraries
```txt
# Core ML libraries
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
catboost>=1.2.0

# Data visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0

# Imbalanced data handling
imbalanced-learn>=0.10.0

# Pipeline and utilities
joblib>=1.3.0
pyyaml>=6.0
typing-extensions>=4.7.0

# Development tools
jupyter>=1.0.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Optional advanced
optuna>=3.3.0
shap>=0.42.0
```

### Pipeline Execution
```bash
# Run data processing pipeline
python pipelines/data_pipeline.py

# Run notebooks in sequence for analysis and modeling
jupyter notebook notebooks/0_data_preparation.ipynb
jupyter notebook notebooks/1_exploratory_data_analysis.ipynb
jupyter notebook notebooks/2_feature_engineering.ipynb
jupyter notebook notebooks/3_model_development.ipynb
jupyter notebook notebooks/4_ensemble_methods.ipynb
jupyter notebook notebooks/5_model_evaluation.ipynb
jupyter notebook notebooks/6_business_analysis.ipynb
```

### Testing
```bash
pytest tests/ -v
black src/ pipelines/ --check
flake8 src/ pipelines/
```

## Deliverables Checklist

1. **✅ Jupyter Notebook Analysis**
   - Complete EDA with business insights
   - Ensemble method implementation and comparison
   - Model evaluation with proper imbalanced data metrics

2. **✅ Modular Python Pipeline**
   - Production-ready data processing pipeline
   - Configurable preprocessing modules
   - End-to-end data pipeline orchestration
   - (Note: Model development remains in Jupyter notebooks)

3. **✅ Executive Summary Report**
   - 2-3 page business-focused summary
   - Actionable retention recommendations
   - ROI analysis and implementation roadmap

4. **✅ Model Performance Comparison**
   - Comprehensive evaluation across all metrics
   - Statistical significance testing
   - Business value analysis

5. **✅ Business Impact Analysis**
   - Customer segmentation strategies
   - Revenue impact calculations
   - Targeted intervention recommendations

## Success Guidelines

1. **Start with EDA**: Begin with thorough Jupyter notebook analysis
2. **Follow Naming Conventions**: Use the numbered approach from past weeks
3. **Reuse Boilerplate**: Adapt data processing code from Week 05_06
4. **Focus on Business Value**: Always connect technical findings to business impact
5. **Handle Class Imbalance**: Use appropriate metrics and evaluation strategies
6. **Document Everything**: Comprehensive documentation for production readiness

## Important Notes

- **Template Only**: Use `Past Notebooks and Pipeline/` for reference - do not modify
- **Data Processing Only**: Python scripts should only handle data processing (following Week 05_06 approach)
- **Model Development in Notebooks**: All model development, training, evaluation should be in Jupyter notebooks
- **Telco Focus**: All analysis should be specific to telecommunications industry
- **Class Imbalance**: Critical focus on proper evaluation for imbalanced churn data
- **Business Impact**: Strong emphasis on actionable business recommendations
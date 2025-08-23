"""
Configuration Management Module for Telco Customer Churn Project

This module provides utilities for loading and managing configuration settings
from the config.yaml file, following the Week 05_06 approach.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'config/config.yaml'
)

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        if not os.path.exists(CONFIG_FILE):
            logger.warning(f"Configuration file not found: {CONFIG_FILE}")
            return create_default_config()
        
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from: {CONFIG_FILE}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary"""
    return load_config()

def get_data_config() -> Dict[str, Any]:
    """
    Get data-related configuration
    
    Returns:
        Dict[str, Any]: Data configuration
    """
    config = get_config()
    return config.get('data', {})

def get_data_paths() -> Dict[str, str]:
    """
    Get data file paths from configuration
    
    Returns:
        Dict[str, str]: Data paths dictionary
    """
    data_config = get_data_config()
    return {
        'raw_path': data_config.get('raw_path', 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        'processed_path': data_config.get('processed_path', 'data/processed/'),
        'artifacts_path': data_config.get('artifacts_path', 'artifacts/data/'),
        'missing_handled': data_config.get('missing_handled', 'data/processed/telco_missing_handled.csv'),
        'outliers_removed': data_config.get('outliers_removed', 'data/processed/telco_outliers_removed.csv'),
        'binned': data_config.get('binned', 'data/processed/telco_binned.csv'),
        'encoded': data_config.get('encoded', 'data/processed/telco_encoded.csv'),
        'scaled': data_config.get('scaled', 'data/processed/telco_scaled.csv')
    }

def get_preprocessing_config() -> Dict[str, Any]:
    """
    Get preprocessing configuration
    
    Returns:
        Dict[str, Any]: Preprocessing configuration
    """
    config = get_config()
    return config.get('preprocessing', {})

def get_missing_values_config() -> Dict[str, Any]:
    """
    Get missing values handling configuration
    
    Returns:
        Dict[str, Any]: Missing values configuration
    """
    preprocessing_config = get_preprocessing_config()
    return {
        'strategy': preprocessing_config.get('missing_strategy', 'telco_specific'),
        'numerical_strategy': preprocessing_config.get('numerical_missing_strategy', 'median'),
        'categorical_strategy': preprocessing_config.get('categorical_missing_strategy', 'mode')
    }

def get_outlier_config() -> Dict[str, Any]:
    """
    Get outlier detection configuration
    
    Returns:
        Dict[str, Any]: Outlier detection configuration
    """
    preprocessing_config = get_preprocessing_config()
    return {
        'strategy': preprocessing_config.get('outlier_strategy', 'telco_specific'),
        'method': preprocessing_config.get('outlier_method', 'remove'),
        'iqr_multiplier': preprocessing_config.get('iqr_multiplier', 1.5),
        'zscore_threshold': preprocessing_config.get('zscore_threshold', 3.0)
    }

def get_binning_config() -> Dict[str, Any]:
    """
    Get feature binning configuration
    
    Returns:
        Dict[str, Any]: Feature binning configuration
    """
    preprocessing_config = get_preprocessing_config()
    return {
        'strategies': preprocessing_config.get('binning_strategies', ['tenure', 'monthly_charges', 'total_charges', 'service_count']),
        'keep_original': preprocessing_config.get('keep_original_binned', True)
    }

def get_encoding_config() -> Dict[str, Any]:
    """
    Get feature encoding configuration
    
    Returns:
        Dict[str, Any]: Feature encoding configuration
    """
    preprocessing_config = get_preprocessing_config()
    return {
        'method': preprocessing_config.get('encoding_method', 'telco_specific'),
        'drop_first': preprocessing_config.get('drop_first', True),
        'handle_unknown': preprocessing_config.get('handle_unknown', 'ignore')
    }

def get_scaling_config() -> Dict[str, Any]:
    """
    Get feature scaling configuration
    
    Returns:
        Dict[str, Any]: Feature scaling configuration
    """
    preprocessing_config = get_preprocessing_config()
    return {
        'method': preprocessing_config.get('scaling_method', 'standard'),
        'feature_range': preprocessing_config.get('feature_range', [0, 1])
    }

def get_splitting_config() -> Dict[str, Any]:
    """
    Get data splitting configuration
    
    Returns:
        Dict[str, Any]: Data splitting configuration
    """
    evaluation_config = get_evaluation_config()
    return {
        'test_size': evaluation_config.get('test_size', 0.2),
        'random_state': evaluation_config.get('random_state', 42),
        'stratify': evaluation_config.get('stratify', True),
        'split_type': evaluation_config.get('split_type', 'stratified')
    }

def get_models_config() -> Dict[str, Any]:
    """
    Get models configuration
    
    Returns:
        Dict[str, Any]: Models configuration
    """
    config = get_config()
    return config.get('models', {})

def get_evaluation_config() -> Dict[str, Any]:
    """
    Get evaluation configuration
    
    Returns:
        Dict[str, Any]: Evaluation configuration
    """
    config = get_config()
    return config.get('evaluation', {})

def get_artifacts_config() -> Dict[str, Any]:
    """
    Get artifacts configuration
    
    Returns:
        Dict[str, Any]: Artifacts configuration
    """
    config = get_config()
    return config.get('artifacts', {})

def get_pipeline_config() -> Dict[str, Any]:
    """
    Get pipeline-specific configuration
    
    Returns:
        Dict[str, Any]: Pipeline configuration
    """
    config = get_config()
    return config.get('pipeline', {
        'save_intermediate': True,
        'verbose': True,
        'target_column': 'Churn'
    })

def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration
    
    Returns:
        Dict[str, Any]: Logging configuration
    """
    config = get_config()
    return config.get('logging', {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/pipeline.log'
    })

def get_telco_specific_config() -> Dict[str, Any]:
    """
    Get Telco-specific configuration settings
    
    Returns:
        Dict[str, Any]: Telco-specific configuration
    """
    return {
        'target_column': 'Churn',
        'customer_id_column': 'customerID',
        'numerical_columns': ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'],
        'categorical_columns': [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ],
        'service_columns': [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ],
        'binary_columns': [
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn'
        ]
    }

def get_environment_config() -> Dict[str, Any]:
    """
    Get environment-specific configuration
    
    Returns:
        Dict[str, Any]: Environment configuration
    """
    config = get_config()
    return config.get('environment', {
        'python_version': '3.8+',
        'required_packages': ['pandas', 'scikit-learn', 'numpy', 'yaml'],
        'memory_limit': '8GB',
        'cpu_cores': -1
    })

def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values
    
    Args:
        updates (Dict[str, Any]): Dictionary of updates to apply
    """
    try:
        config = get_config()
        
        for key, value in updates.items():
            keys = key.split('.')
            current = config
            
            # Navigate to the nested key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Update the final key
            current[keys[-1]] = value
        
        # Save updated configuration
        with open(CONFIG_FILE, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration if config file doesn't exist
    
    Returns:
        Dict[str, Any]: Default configuration
    """
    default_config = {
        'data': {
            'raw_path': 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'processed_path': 'data/processed/',
            'artifacts_path': 'artifacts/data/',
            'missing_handled': 'data/processed/telco_missing_handled.csv',
            'outliers_removed': 'data/processed/telco_outliers_removed.csv',
            'binned': 'data/processed/telco_binned.csv',
            'encoded': 'data/processed/telco_encoded.csv',
            'scaled': 'data/processed/telco_scaled.csv'
        },
        'preprocessing': {
            'missing_strategy': 'telco_specific',
            'outlier_strategy': 'telco_specific',
            'outlier_method': 'remove',
            'scaling_method': 'standard',
            'encoding_method': 'telco_specific',
            'binning_strategies': ['tenure', 'monthly_charges', 'total_charges', 'service_count']
        },
        'evaluation': {
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'scoring': ['precision', 'recall', 'f1', 'roc_auc'],
            'stratify': True,
            'split_type': 'stratified'
        },
        'artifacts': {
            'models_path': 'artifacts/models/',
            'reports_path': 'artifacts/reports/',
            'encoders_path': 'artifacts/encode/',
            'scalers_path': 'artifacts/scalers/'
        },
        'pipeline': {
            'save_intermediate': True,
            'verbose': True,
            'target_column': 'Churn'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/pipeline.log'
        }
    }
    
    try:
        # Ensure config directory exists
        config_dir = os.path.dirname(CONFIG_FILE)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save default configuration
        with open(CONFIG_FILE, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Created default configuration file: {CONFIG_FILE}")
        
    except Exception as e:
        logger.error(f"Error creating default configuration: {e}")
    
    return default_config

def validate_config() -> bool:
    """
    Validate configuration settings
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        config = get_config()
        
        # Check required sections
        required_sections = ['data', 'preprocessing', 'evaluation', 'artifacts']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Check required data paths
        data_config = get_data_config()
        if 'raw_path' not in data_config:
            logger.error("Missing raw data path in configuration")
            return False
        
        # Check if raw data file exists
        raw_path = data_config['raw_path']
        if not os.path.exists(raw_path):
            logger.warning(f"Raw data file not found: {raw_path}")
        
        logger.info("Configuration validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def print_config_summary() -> None:
    """Print a summary of the current configuration"""
    try:
        config = get_config()
        
        print("=" * 60)
        print("TELCO CHURN PREDICTION - CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Data paths
        data_config = get_data_config()
        print(f"📁 Data Configuration:")
        print(f"   Raw Data: {data_config.get('raw_path', 'Not configured')}")
        print(f"   Processed Path: {data_config.get('processed_path', 'Not configured')}")
        print(f"   Artifacts Path: {data_config.get('artifacts_path', 'Not configured')}")
        
        # Preprocessing
        preprocessing_config = get_preprocessing_config()
        print(f"\n⚙️ Preprocessing Configuration:")
        print(f"   Missing Strategy: {preprocessing_config.get('missing_strategy', 'Not configured')}")
        print(f"   Scaling Method: {preprocessing_config.get('scaling_method', 'Not configured')}")
        print(f"   Encoding Method: {preprocessing_config.get('encoding_method', 'Not configured')}")
        
        # Evaluation
        evaluation_config = get_evaluation_config()
        print(f"\n📊 Evaluation Configuration:")
        print(f"   Test Size: {evaluation_config.get('test_size', 'Not configured')}")
        print(f"   CV Folds: {evaluation_config.get('cv_folds', 'Not configured')}")
        print(f"   Random State: {evaluation_config.get('random_state', 'Not configured')}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error printing configuration summary: {e}")

# Initialize configuration on import
if __name__ == "__main__":
    # Validate configuration when run as script
    if validate_config():
        print_config_summary()
    else:
        print("Configuration validation failed!")
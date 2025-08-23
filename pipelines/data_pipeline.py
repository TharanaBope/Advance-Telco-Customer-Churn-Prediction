"""
Telco Customer Churn Data Pipeline

This module orchestrates the complete data processing pipeline for the Telco Customer Churn
prediction project, following the same structure as Week 05_06 data_pipeline.py.

Pipeline Steps:
1. Data Ingestion - Load raw Telco CSV data
2. Missing Values Handling - Handle missing values using Telco-specific strategies
3. Outlier Detection - Detect and handle outliers in numerical features
4. Feature Binning - Create categorical features from numerical ones
5. Feature Encoding - Encode categorical variables
6. Feature Scaling - Scale numerical features
7. Data Splitting - Create stratified train/test splits

Each step saves intermediate results and can be configured via config.yaml.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

# Import data processing modules
from data_processing.data_ingestion import ingest_telco_data
from data_processing.handle_missing_values import handle_missing_values
from data_processing.outlier_detection import detect_and_handle_outliers
from data_processing.feature_binning import apply_feature_binning
from data_processing.feature_encoding import encode_telco_features
from data_processing.feature_scaling import scale_numerical_features
from data_processing.data_splitter import split_telco_data

# Import utilities
from utils.config import (
    get_data_paths, get_preprocessing_config, get_missing_values_config,
    get_outlier_config, get_binning_config, get_encoding_config, 
    get_scaling_config, get_splitting_config, get_pipeline_config,
    get_telco_specific_config, validate_config, print_config_summary
)
from utils.logger import (
    get_pipeline_logger, get_step_logger, log_pipeline_start, 
    log_pipeline_end, log_error_with_context, get_file_size_string
)

class TelcoDataPipeline:
    """Main data pipeline class for Telco Customer Churn prediction"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize Telco Data Pipeline
        
        Args:
            config_override (Optional[Dict[str, Any]]): Configuration overrides
        """
        self.start_time = datetime.now()
        self.config_override = config_override or {}
        
        # Initialize logging
        self.logger = get_pipeline_logger('telco_data_pipeline')
        
        # Load and validate configuration
        if not validate_config():
            raise ValueError("Configuration validation failed. Please check config.yaml")
        
        # Load configurations
        self.data_paths = get_data_paths()
        self.preprocessing_config = get_preprocessing_config()
        self.pipeline_config = get_pipeline_config()
        self.telco_config = get_telco_specific_config()
        
        # Apply configuration overrides
        self._apply_config_overrides()
        
        # Log pipeline initialization
        log_pipeline_start(self.logger, "Telco Customer Churn Data Pipeline")
        print_config_summary()
        
        # Initialize step counters
        self.current_step = 0
        self.total_steps = 7
        
    def _apply_config_overrides(self):
        """Apply configuration overrides"""
        if self.config_override:
            self.logger.info(f"Applying configuration overrides: {self.config_override}")
            # Update configurations with overrides
            for key, value in self.config_override.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _next_step(self, step_name: str) -> int:
        """Increment step counter and return current step"""
        self.current_step += 1
        self.logger.info(f"\n{'='*20} STEP {self.current_step}/{self.total_steps}: {step_name.upper()} {'='*20}")
        return self.current_step
    
    def _check_existing_artifacts(self) -> bool:
        """Check if final artifacts already exist"""
        artifacts_path = self.data_paths['artifacts_path']
        required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
        
        all_exist = all(
            os.path.exists(os.path.join(artifacts_path, file)) 
            for file in required_files
        )
        
        if all_exist:
            self.logger.info(f"Found existing artifacts in {artifacts_path}")
            return True
        
        return False
    
    def step_1_data_ingestion(self, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Step 1: Data Ingestion
        Load and initially process the Telco CSV data
        """
        step_name = "Data Ingestion"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Loading raw Telco customer data")
            
            raw_path = self.data_paths['raw_path']
            
            # Check if raw data exists
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Raw data file not found: {raw_path}")
            
            # Load data using data ingestion module
            df = ingest_telco_data(
                file_path=raw_path,
                handle_totalcharges=True
            )
            
            # Log data information
            step_logger.log_data_info(df.shape, "Raw Telco dataset")
            
            # Log data types and missing values
            self.logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
            missing_summary = df.isnull().sum()
            missing_columns = missing_summary[missing_summary > 0]
            if len(missing_columns) > 0:
                self.logger.info(f"Missing values found in {len(missing_columns)} columns")
                for col, count in missing_columns.items():
                    pct = (count / len(df)) * 100
                    self.logger.info(f"   - {col}: {count} ({pct:.1f}%)")
            else:
                self.logger.info("No missing values found in raw data")
            
            # Log target variable distribution
            target_col = self.telco_config['target_column']
            if target_col in df.columns:
                target_dist = df[target_col].value_counts()
                self.logger.info(f"Target variable ({target_col}) distribution: {target_dist.to_dict()}")
            
            step_logger.step_end("Raw data loaded successfully", success=True)
            return df
            
        except Exception as e:
            step_logger.step_end(f"Data ingestion failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_2_handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Handle Missing Values
        Apply Telco-specific missing value handling strategies
        """
        step_name = "Handle Missing Values"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Handling missing values using Telco-specific strategies")
            
            initial_missing = df.isnull().sum().sum()
            step_logger.log_processing_stats(len(df), len(df), "Initial dataset size")
            
            # Get missing values configuration
            missing_config = get_missing_values_config()
            
            # Handle missing values using the missing values module
            df_processed = handle_missing_values(
                df=df,
                strategy=missing_config.get('strategy', 'telco_specific'),
                output_path=self.data_paths['missing_handled']
            )
            
            # Log processing results
            final_missing = df_processed.isnull().sum().sum()
            step_logger.log_processing_stats(initial_missing, final_missing, "Missing values handled")
            step_logger.log_data_info(df_processed.shape, "Dataset after missing value handling")
            
            # Log file saved
            if os.path.exists(self.data_paths['missing_handled']):
                file_size = get_file_size_string(self.data_paths['missing_handled'])
                step_logger.log_file_saved(self.data_paths['missing_handled'], file_size)
            
            step_logger.step_end("Missing values handled successfully", success=True)
            return df_processed
            
        except Exception as e:
            step_logger.step_end(f"Missing value handling failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_3_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Handle Outliers
        Detect and handle outliers in numerical features
        """
        step_name = "Handle Outliers"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Detecting and handling outliers in numerical features")
            
            initial_rows = len(df)
            
            # Get outlier configuration
            outlier_config = get_outlier_config()
            
            # Handle outliers using the outlier detection module
            df_processed = detect_and_handle_outliers(
                df=df,
                strategy=outlier_config.get('strategy', 'telco_specific'),
                method=outlier_config.get('method', 'remove'),
                output_path=self.data_paths['outliers_removed']
            )
            
            # Log processing results
            final_rows = len(df_processed)
            step_logger.log_processing_stats(initial_rows, final_rows, "Outlier handling")
            step_logger.log_data_info(df_processed.shape, "Dataset after outlier handling")
            
            # Log file saved
            if os.path.exists(self.data_paths['outliers_removed']):
                file_size = get_file_size_string(self.data_paths['outliers_removed'])
                step_logger.log_file_saved(self.data_paths['outliers_removed'], file_size)
            
            step_logger.step_end("Outliers handled successfully", success=True)
            return df_processed
            
        except Exception as e:
            step_logger.step_end(f"Outlier handling failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_4_feature_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Feature Binning
        Create categorical features from numerical ones
        """
        step_name = "Feature Binning"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Creating binned features for tenure, charges, and services")
            
            initial_columns = len(df.columns)
            
            # Get binning configuration
            binning_config = get_binning_config()
            
            # Apply feature binning using the binning module
            df_processed = apply_feature_binning(
                df=df,
                strategies=binning_config.get('strategies', ['tenure', 'monthly_charges', 'total_charges', 'service_count']),
                output_path=self.data_paths['binned']
            )
            
            # Log processing results
            final_columns = len(df_processed.columns)
            new_features = final_columns - initial_columns
            step_logger.log_feature_info(new_features, "binned features")
            step_logger.log_data_info(df_processed.shape, "Dataset after feature binning")
            
            # Log new columns created
            new_columns = [col for col in df_processed.columns if col not in df.columns]
            if new_columns:
                self.logger.info(f"New features created: {new_columns}")
            
            # Log file saved
            if os.path.exists(self.data_paths['binned']):
                file_size = get_file_size_string(self.data_paths['binned'])
                step_logger.log_file_saved(self.data_paths['binned'], file_size)
            
            step_logger.step_end("Feature binning completed successfully", success=True)
            return df_processed
            
        except Exception as e:
            step_logger.step_end(f"Feature binning failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_5_feature_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Feature Encoding
        Encode categorical variables for machine learning
        """
        step_name = "Feature Encoding"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Encoding categorical features using Telco-specific strategies")
            
            initial_categorical = len(df.select_dtypes(include=['object']).columns)
            initial_columns = len(df.columns)
            
            # Get encoding configuration
            encoding_config = get_encoding_config()
            
            # Apply feature encoding using the encoding module
            df_processed = encode_telco_features(
                df=df,
                encoding_type=encoding_config.get('method', 'telco_specific'),
                output_path=self.data_paths['encoded'],
                artifacts_path="artifacts/encode"
            )
            
            # Log processing results
            final_categorical = len(df_processed.select_dtypes(include=['object']).columns)
            final_columns = len(df_processed.columns)
            
            step_logger.log_processing_stats(initial_categorical, final_categorical, "Categorical columns encoded")
            step_logger.log_processing_stats(initial_columns, final_columns, "Total columns after encoding")
            step_logger.log_data_info(df_processed.shape, "Dataset after feature encoding")
            
            # Log data types after encoding
            dtype_summary = df_processed.dtypes.value_counts()
            self.logger.info(f"Data types after encoding: {dtype_summary.to_dict()}")
            
            # Log file saved
            if os.path.exists(self.data_paths['encoded']):
                file_size = get_file_size_string(self.data_paths['encoded'])
                step_logger.log_file_saved(self.data_paths['encoded'], file_size)
            
            step_logger.step_end("Feature encoding completed successfully", success=True)
            return df_processed
            
        except Exception as e:
            step_logger.step_end(f"Feature encoding failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_6_feature_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Feature Scaling
        Scale numerical features for machine learning algorithms
        """
        step_name = "Feature Scaling"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Scaling numerical features")
            
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column and ID-like columns
            target_col = self.telco_config['target_column']
            if target_col in numerical_columns:
                numerical_columns.remove(target_col)
            
            numerical_columns = [col for col in numerical_columns if 'id' not in col.lower()]
            
            self.logger.info(f"Scaling {len(numerical_columns)} numerical features: {numerical_columns}")
            
            # Get scaling configuration
            scaling_config = get_scaling_config()
            
            # Apply feature scaling using the scaling module
            df_processed = scale_numerical_features(
                df=df,
                scaling_type=scaling_config.get('method', 'standard'),
                columns_to_scale=numerical_columns,
                output_path=self.data_paths['scaled'],
                artifacts_path="artifacts/scalers"
            )
            
            # Log processing results
            step_logger.log_feature_info(len(numerical_columns), "scaled features")
            step_logger.log_data_info(df_processed.shape, "Dataset after feature scaling")
            
            # Log scaling statistics for key features
            key_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for feature in key_features:
                if feature in df_processed.columns:
                    mean_val = df_processed[feature].mean()
                    std_val = df_processed[feature].std()
                    self.logger.info(f"{feature} after scaling: mean={mean_val:.4f}, std={std_val:.4f}")
            
            # Log file saved
            if os.path.exists(self.data_paths['scaled']):
                file_size = get_file_size_string(self.data_paths['scaled'])
                step_logger.log_file_saved(self.data_paths['scaled'], file_size)
            
            step_logger.step_end("Feature scaling completed successfully", success=True)
            return df_processed
            
        except Exception as e:
            step_logger.step_end(f"Feature scaling failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def step_7_data_splitting(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Step 7: Data Splitting
        Create stratified train/test splits for machine learning
        """
        step_name = "Data Splitting"
        step_num = self._next_step(step_name)
        step_logger = get_step_logger(f"Step {step_num}: {step_name}", self.logger)
        
        try:
            step_logger.step_start("Creating stratified train/test splits")
            
            # Get splitting configuration
            splitting_config = get_splitting_config()
            target_column = self.telco_config['target_column']
            
            # Apply data splitting using the splitter module
            X_train, X_test, y_train, y_test = split_telco_data(
                df=df,
                target_column=target_column,
                split_type=splitting_config.get('split_type', 'stratified'),
                test_size=splitting_config.get('test_size', 0.2),
                output_path=self.data_paths['artifacts_path'],
                save_splits=True,
                random_state=splitting_config.get('random_state', 42)
            )
            
            # Log split results
            self.logger.info(f"Data split completed:")
            self.logger.info(f"   Total samples: {len(df):,}")
            self.logger.info(f"   Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
            self.logger.info(f"   Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
            self.logger.info(f"   Features: {X_train.shape[1]}")
            
            # Log target distribution in splits
            from collections import Counter
            train_dist = Counter(y_train)
            test_dist = Counter(y_test)
            self.logger.info(f"   Train target distribution: {dict(train_dist)}")
            self.logger.info(f"   Test target distribution: {dict(test_dist)}")
            
            # Log files saved
            artifacts_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
            for file_name in artifacts_files:
                file_path = os.path.join(self.data_paths['artifacts_path'], file_name)
                if os.path.exists(file_path):
                    file_size = get_file_size_string(file_path)
                    step_logger.log_file_saved(file_path, file_size)
            
            step_logger.step_end("Data splitting completed successfully", success=True)
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            step_logger.step_end(f"Data splitting failed: {str(e)}", success=False)
            log_error_with_context(self.logger, e, step_name)
            raise
    
    def run_pipeline(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Run the complete data pipeline
        
        Args:
            force_rebuild (bool): Force rebuild even if artifacts exist
            
        Returns:
            Dict[str, Any]: Pipeline results and metadata
        """
        try:
            # Check for existing artifacts
            if not force_rebuild and self._check_existing_artifacts():
                self.logger.info("Loading existing artifacts...")
                
                # Load existing splits
                artifacts_path = self.data_paths['artifacts_path']
                X_train = pd.read_csv(os.path.join(artifacts_path, 'X_train.csv'))
                X_test = pd.read_csv(os.path.join(artifacts_path, 'X_test.csv'))
                y_train = pd.read_csv(os.path.join(artifacts_path, 'y_train.csv'))['y_train']
                y_test = pd.read_csv(os.path.join(artifacts_path, 'y_test.csv'))['y_test']
                
                self.logger.info("Loaded existing artifacts successfully")
                
                # Log final results
                self._log_pipeline_summary(X_train, X_test, y_train, y_test)
                log_pipeline_end(self.logger, self.start_time, success=True)
                
                return {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'pipeline_duration': datetime.now() - self.start_time,
                    'artifacts_loaded': True
                }
            
            # Run pipeline steps
            self.logger.info("Running complete data pipeline...")
            
            # Step 1: Data Ingestion
            df = self.step_1_data_ingestion(force_rebuild)
            
            # Step 2: Handle Missing Values
            df = self.step_2_handle_missing_values(df)
            
            # Step 3: Handle Outliers
            df = self.step_3_handle_outliers(df)
            
            # Step 4: Feature Binning
            df = self.step_4_feature_binning(df)
            
            # Step 5: Feature Encoding
            df = self.step_5_feature_encoding(df)
            
            # Step 6: Feature Scaling
            df = self.step_6_feature_scaling(df)
            
            # Step 7: Data Splitting
            X_train, X_test, y_train, y_test = self.step_7_data_splitting(df)
            
            # Log final results
            self._log_pipeline_summary(X_train, X_test, y_train, y_test)
            log_pipeline_end(self.logger, self.start_time, success=True)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'pipeline_duration': datetime.now() - self.start_time,
                'artifacts_loaded': False
            }
            
        except Exception as e:
            log_pipeline_end(self.logger, self.start_time, success=False)
            log_error_with_context(self.logger, e, "Pipeline Execution")
            raise
    
    def _log_pipeline_summary(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        """Log final pipeline summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TELCO DATA PIPELINE SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Pipeline completed in {datetime.now() - self.start_time}")
        self.logger.info(f"Final dataset shapes:")
        self.logger.info(f"   X_train: {X_train.shape}")
        self.logger.info(f"   X_test: {X_test.shape}")
        self.logger.info(f"   y_train: {y_train.shape}")
        self.logger.info(f"   y_test: {y_test.shape}")
        self.logger.info(f"Artifacts saved to: {self.data_paths['artifacts_path']}")
        self.logger.info(f"Features ready for model training: {X_train.shape[1]}")
        self.logger.info("="*80)

def run_telco_data_pipeline(
    force_rebuild: bool = False,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function to run the Telco data pipeline
    
    Args:
        force_rebuild (bool): Force rebuild even if artifacts exist
        config_override (Optional[Dict[str, Any]]): Configuration overrides
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    pipeline = TelcoDataPipeline(config_override)
    return pipeline.run_pipeline(force_rebuild)

if __name__ == "__main__":
    """
    Script execution entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Telco Customer Churn Data Pipeline')
    parser.add_argument(
        '--force-rebuild', 
        action='store_true', 
        help='Force rebuild even if artifacts exist'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file (defaults to config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run the pipeline
        results = run_telco_data_pipeline(
            force_rebuild=args.force_rebuild,
            config_override={'config_file': args.config_file} if args.config_file else None
        )
        
        print("\n" + "="*60)
        print("TELCO DATA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training set: {results['X_train'].shape}")
        print(f"Test set: {results['X_test'].shape}")
        print(f"Duration: {results['pipeline_duration']}")
        print(f"Artifacts ready for model training!")
        print("="*60)
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)
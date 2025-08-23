"""
Telco Customer Churn Missing Values Handling Module

This module provides strategies for handling missing values in the Telco dataset,
including specific handling for TotalCharges and categorical features.

Based on Week 05_06 missing value handling pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from sklearn.impute import SimpleImputer, KNNImputer
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies"""
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to handle missing values"""
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy for dropping rows with missing values in critical columns"""
    
    def __init__(self, critical_columns: List[str] = None, threshold: float = 0.5):
        """
        Initialize drop missing values strategy
        
        Args:
            critical_columns (List[str]): Columns where missing values should cause row removal
            threshold (float): Drop columns if missing percentage > threshold
        """
        self.critical_columns = critical_columns or []
        self.threshold = threshold
        logging.info(f"Initialized DropMissingValuesStrategy with critical columns: {self.critical_columns}")
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by dropping rows/columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values dropped
        """
        try:
            initial_shape = df.shape
            df_cleaned = df.copy()
            
            # Drop rows with missing values in critical columns
            if self.critical_columns:
                df_cleaned = df_cleaned.dropna(subset=self.critical_columns)
                rows_dropped = initial_shape[0] - df_cleaned.shape[0]
                logging.info(f"Dropped {rows_dropped} rows with missing values in critical columns")
            
            # Drop columns with high missing percentage
            missing_percentages = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
            columns_to_drop = missing_percentages[missing_percentages > self.threshold * 100].index.tolist()
            
            if columns_to_drop:
                df_cleaned = df_cleaned.drop(columns=columns_to_drop)
                logging.info(f"Dropped columns with >{self.threshold*100}% missing: {columns_to_drop}")
            
            final_shape = df_cleaned.shape
            logging.info(f"Shape changed from {initial_shape} to {final_shape}")
            
            return df_cleaned
            
        except Exception as e:
            logging.error(f"Error in DropMissingValuesStrategy: {str(e)}")
            raise

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy for filling missing values using various imputation methods"""
    
    def __init__(
        self,
        numerical_strategy: str = 'median',
        categorical_strategy: str = 'mode',
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None
    ):
        """
        Initialize fill missing values strategy
        
        Args:
            numerical_strategy (str): Strategy for numerical columns ('mean', 'median', 'constant')
            categorical_strategy (str): Strategy for categorical columns ('mode', 'constant')
            fill_value (Optional[Any]): Value to use for constant strategy
            columns (Optional[List[str]]): Specific columns to handle
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.columns = columns
        
        logging.info(f"Initialized FillMissingValuesStrategy:")
        logging.info(f"  - Numerical strategy: {numerical_strategy}")
        logging.info(f"  - Categorical strategy: {categorical_strategy}")
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by filling them
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values filled
        """
        try:
            df_filled = df.copy()
            columns_to_process = self.columns or df.columns
            
            # Separate numerical and categorical columns
            numerical_cols = df_filled.select_dtypes(include=[np.number]).columns.intersection(columns_to_process)
            categorical_cols = df_filled.select_dtypes(include=['object']).columns.intersection(columns_to_process)
            
            # Handle numerical columns
            if len(numerical_cols) > 0:
                df_filled = self._fill_numerical_columns(df_filled, numerical_cols)
            
            # Handle categorical columns
            if len(categorical_cols) > 0:
                df_filled = self._fill_categorical_columns(df_filled, categorical_cols)
            
            # Log missing values summary
            missing_after = df_filled.isnull().sum().sum()
            logging.info(f"Total missing values after filling: {missing_after}")
            
            return df_filled
            
        except Exception as e:
            logging.error(f"Error in FillMissingValuesStrategy: {str(e)}")
            raise
    
    def _fill_numerical_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values in numerical columns"""
        for col in columns:
            if df[col].isnull().any():
                initial_missing = df[col].isnull().sum()
                
                if self.numerical_strategy == 'mean':
                    fill_value = df[col].mean()
                elif self.numerical_strategy == 'median':
                    fill_value = df[col].median()
                elif self.numerical_strategy == 'constant':
                    fill_value = self.fill_value if self.fill_value is not None else 0
                else:
                    raise ValueError(f"Unknown numerical strategy: {self.numerical_strategy}")
                
                df[col] = df[col].fillna(fill_value)
                logging.info(f"Filled {initial_missing} missing values in {col} with {self.numerical_strategy}: {fill_value}")
        
        return df
    
    def _fill_categorical_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values in categorical columns"""
        for col in columns:
            if df[col].isnull().any():
                initial_missing = df[col].isnull().sum()
                
                if self.categorical_strategy == 'mode':
                    mode_values = df[col].mode()
                    fill_value = mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'
                elif self.categorical_strategy == 'constant':
                    fill_value = self.fill_value if self.fill_value is not None else 'Unknown'
                else:
                    raise ValueError(f"Unknown categorical strategy: {self.categorical_strategy}")
                
                df[col] = df[col].fillna(fill_value)
                logging.info(f"Filled {initial_missing} missing values in {col} with {self.categorical_strategy}: {fill_value}")
        
        return df

class TelcoSpecificMissingValueStrategy(MissingValueHandlingStrategy):
    """Telco-specific missing value handling strategy"""
    
    def __init__(self):
        """Initialize Telco-specific missing value strategy"""
        logging.info("Initialized TelcoSpecificMissingValueStrategy")
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using Telco dataset domain knowledge
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        try:
            df_processed = df.copy()
            
            # Handle TotalCharges missing values (usually for new customers with tenure=0)
            if 'TotalCharges' in df_processed.columns:
                df_processed = self._handle_totalcharges_missing(df_processed)
            
            # Handle other Telco-specific missing values
            df_processed = self._handle_service_related_missing(df_processed)
            
            # Handle demographic missing values
            df_processed = self._handle_demographic_missing(df_processed)
            
            return df_processed
            
        except Exception as e:
            logging.error(f"Error in TelcoSpecificMissingValueStrategy: {str(e)}")
            raise
    
    def _handle_totalcharges_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle TotalCharges missing values specifically"""
        if 'TotalCharges' in df.columns:
            missing_mask = df['TotalCharges'].isnull()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                logging.info(f"Handling {missing_count} missing TotalCharges values")
                
                # For customers with tenure=0, TotalCharges should be 0 or equal to MonthlyCharges
                if 'tenure' in df.columns:
                    zero_tenure_mask = (df['tenure'] == 0) & missing_mask
                    zero_tenure_count = zero_tenure_mask.sum()
                    
                    if zero_tenure_count > 0:
                        # For zero tenure customers, set TotalCharges = MonthlyCharges
                        if 'MonthlyCharges' in df.columns:
                            df.loc[zero_tenure_mask, 'TotalCharges'] = df.loc[zero_tenure_mask, 'MonthlyCharges']
                            logging.info(f"Set TotalCharges = MonthlyCharges for {zero_tenure_count} zero-tenure customers")
                        else:
                            df.loc[zero_tenure_mask, 'TotalCharges'] = 0
                            logging.info(f"Set TotalCharges = 0 for {zero_tenure_count} zero-tenure customers")
                
                # For remaining missing values, use median imputation
                remaining_missing = df['TotalCharges'].isnull().sum()
                if remaining_missing > 0:
                    median_value = df['TotalCharges'].median()
                    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
                    logging.info(f"Filled remaining {remaining_missing} TotalCharges with median: {median_value}")
        
        return df
    
    def _handle_service_related_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in service-related columns"""
        service_columns = [
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        for col in service_columns:
            if col in df.columns and df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                
                # For service columns, missing often means "No service"
                df[col] = df[col].fillna('No')
                logging.info(f"Filled {missing_count} missing values in {col} with 'No'")
        
        return df
    
    def _handle_demographic_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in demographic columns"""
        # Handle gender missing values with mode
        if 'gender' in df.columns and df['gender'].isnull().any():
            missing_count = df['gender'].isnull().sum()
            mode_gender = df['gender'].mode().iloc[0] if len(df['gender'].mode()) > 0 else 'Unknown'
            df['gender'] = df['gender'].fillna(mode_gender)
            logging.info(f"Filled {missing_count} missing gender values with mode: {mode_gender}")
        
        # Handle Partner/Dependents with mode
        for col in ['Partner', 'Dependents']:
            if col in df.columns and df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'No'
                df[col] = df[col].fillna(mode_value)
                logging.info(f"Filled {missing_count} missing {col} values with mode: {mode_value}")
        
        return df

class KNNImputationStrategy(MissingValueHandlingStrategy):
    """KNN-based imputation strategy for complex missing patterns"""
    
    def __init__(self, n_neighbors: int = 5, columns: Optional[List[str]] = None):
        """
        Initialize KNN imputation strategy
        
        Args:
            n_neighbors (int): Number of neighbors for KNN imputation
            columns (Optional[List[str]]): Columns to apply KNN imputation
        """
        self.n_neighbors = n_neighbors
        self.columns = columns
        logging.info(f"Initialized KNNImputationStrategy with {n_neighbors} neighbors")
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using KNN imputation
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with KNN-imputed values
        """
        try:
            df_processed = df.copy()
            columns_to_process = self.columns or df.select_dtypes(include=[np.number]).columns
            
            if len(columns_to_process) == 0:
                logging.warning("No numerical columns found for KNN imputation")
                return df_processed
            
            # Apply KNN imputation only to numerical columns
            missing_before = df_processed[columns_to_process].isnull().sum().sum()
            
            if missing_before > 0:
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                df_processed[columns_to_process] = imputer.fit_transform(df_processed[columns_to_process])
                
                missing_after = df_processed[columns_to_process].isnull().sum().sum()
                logging.info(f"KNN imputation completed. Missing values: {missing_before} -> {missing_after}")
            
            return df_processed
            
        except Exception as e:
            logging.error(f"Error in KNNImputationStrategy: {str(e)}")
            raise

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'telco_specific',
    output_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Main function to handle missing values in Telco dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy to use ('drop', 'fill', 'telco_specific', 'knn')
        output_path (Optional[str]): Path to save processed data
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    try:
        logging.info(f"Handling missing values using {strategy} strategy")
        
        # Log initial missing values
        initial_missing = df.isnull().sum().sum()
        logging.info(f"Initial missing values: {initial_missing}")
        
        # Select appropriate strategy
        if strategy == 'drop':
            handler = DropMissingValuesStrategy(**kwargs)
        elif strategy == 'fill':
            handler = FillMissingValuesStrategy(**kwargs)
        elif strategy == 'telco_specific':
            handler = TelcoSpecificMissingValueStrategy()
        elif strategy == 'knn':
            handler = KNNImputationStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Handle missing values
        df_processed = handler.handle(df)
        
        # Log final missing values
        final_missing = df_processed.isnull().sum().sum()
        logging.info(f"Final missing values: {final_missing}")
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_processed.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df_processed
        
    except Exception as e:
        logging.error(f"Missing value handling failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from data ingestion)
        input_file = "data/processed/telco_ingested.csv"
        output_file = "data/processed/telco_missing_handled.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Handle missing values using Telco-specific strategy
            df_processed = handle_missing_values(
                df=df,
                strategy='telco_specific',
                output_path=output_file
            )
            
            print(f"Missing value handling completed successfully!")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"Missing values per column:")
            print(df_processed.isnull().sum().to_string())
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run data_ingestion.py first")
            
    except Exception as e:
        print(f"Missing value handling failed: {e}")
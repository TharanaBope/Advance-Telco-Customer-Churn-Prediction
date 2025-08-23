"""
Telco Customer Churn Outlier Detection Module

This module provides strategies for detecting and handling outliers in numerical features
of the Telco dataset, specifically for MonthlyCharges, TotalCharges, and tenure.

Based on Week 05_06 outlier detection pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
from scipy import stats
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    """Abstract base class for outlier detection strategies"""
    
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Abstract method to detect outliers in specified columns"""
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    """IQR (Interquartile Range) based outlier detection strategy"""
    
    def __init__(self, multiplier: float = 1.5):
        """
        Initialize IQR outlier detection
        
        Args:
            multiplier (float): IQR multiplier for outlier threshold (default: 1.5)
        """
        self.multiplier = multiplier
        logging.info(f"Initialized IQROutlierDetection with multiplier: {multiplier}")
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        try:
            outliers = pd.DataFrame(False, index=df.index, columns=columns)
            outlier_stats = {}
            
            for col in columns:
                if col not in df.columns:
                    logging.warning(f"Column {col} not found in dataframe")
                    continue
                
                # Convert to numeric and handle non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df[col].isnull().all():
                    logging.warning(f"Column {col} contains no numeric values")
                    continue
                
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Calculate outlier bounds
                lower_bound = Q1 - self.multiplier * IQR
                upper_bound = Q3 + self.multiplier * IQR
                
                # Detect outliers
                column_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = column_outliers
                
                # Store statistics
                outlier_count = column_outliers.sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_stats[col] = {
                    'IQR': IQR,
                    'Q1': Q1,
                    'Q3': Q3,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_percentage
                }
                
                logging.info(f"IQR outlier detection for {col}:")
                logging.info(f"  - IQR: {IQR:.2f}")
                logging.info(f"  - Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logging.info(f"  - Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
            
            logging.info("Outliers detected using IQR method")
            return outliers
            
        except Exception as e:
            logging.error(f"Error in IQR outlier detection: {str(e)}")
            raise

class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """Z-score based outlier detection strategy"""
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-score outlier detection
        
        Args:
            threshold (float): Z-score threshold for outlier detection (default: 3.0)
        """
        self.threshold = threshold
        logging.info(f"Initialized ZScoreOutlierDetection with threshold: {threshold}")
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using Z-score method
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        try:
            outliers = pd.DataFrame(False, index=df.index, columns=columns)
            outlier_stats = {}
            
            for col in columns:
                if col not in df.columns:
                    logging.warning(f"Column {col} not found in dataframe")
                    continue
                
                # Convert to numeric and handle non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df[col].isnull().all():
                    logging.warning(f"Column {col} contains no numeric values")
                    continue
                
                # Calculate Z-scores
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                
                # Create mask for original dataframe (including NaN positions)
                valid_indices = df[col].dropna().index
                column_outliers = pd.Series(False, index=df.index)
                column_outliers.loc[valid_indices] = z_scores > self.threshold
                
                outliers[col] = column_outliers
                
                # Store statistics
                outlier_count = column_outliers.sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'threshold': self.threshold,
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_percentage
                }
                
                logging.info(f"Z-score outlier detection for {col}:")
                logging.info(f"  - Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
                logging.info(f"  - Threshold: {self.threshold}")
                logging.info(f"  - Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
            
            logging.info("Outliers detected using Z-score method")
            return outliers
            
        except Exception as e:
            logging.error(f"Error in Z-score outlier detection: {str(e)}")
            raise

class ModifiedZScoreOutlierDetection(OutlierDetectionStrategy):
    """Modified Z-score (using median) based outlier detection strategy"""
    
    def __init__(self, threshold: float = 3.5):
        """
        Initialize Modified Z-score outlier detection
        
        Args:
            threshold (float): Modified Z-score threshold (default: 3.5)
        """
        self.threshold = threshold
        logging.info(f"Initialized ModifiedZScoreOutlierDetection with threshold: {threshold}")
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using Modified Z-score method (using median and MAD)
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        try:
            outliers = pd.DataFrame(False, index=df.index, columns=columns)
            
            for col in columns:
                if col not in df.columns:
                    logging.warning(f"Column {col} not found in dataframe")
                    continue
                
                # Convert to numeric and handle non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df[col].isnull().all():
                    logging.warning(f"Column {col} contains no numeric values")
                    continue
                
                # Calculate median and MAD
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                
                # Calculate modified Z-scores
                modified_z_scores = 0.6745 * (df[col] - median) / mad if mad != 0 else pd.Series(0, index=df.index)
                
                # Detect outliers
                column_outliers = np.abs(modified_z_scores) > self.threshold
                outliers[col] = column_outliers
                
                outlier_count = column_outliers.sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                
                logging.info(f"Modified Z-score outlier detection for {col}:")
                logging.info(f"  - Median: {median:.2f}, MAD: {mad:.2f}")
                logging.info(f"  - Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
            
            logging.info("Outliers detected using Modified Z-score method")
            return outliers
            
        except Exception as e:
            logging.error(f"Error in Modified Z-score outlier detection: {str(e)}")
            raise

class TelcoSpecificOutlierDetection(OutlierDetectionStrategy):
    """Telco domain-specific outlier detection strategy"""
    
    def __init__(self):
        """Initialize Telco-specific outlier detection"""
        logging.info("Initialized TelcoSpecificOutlierDetection")
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using Telco domain knowledge
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        try:
            outliers = pd.DataFrame(False, index=df.index, columns=columns)
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if col == 'MonthlyCharges':
                    outliers[col] = self._detect_monthly_charges_outliers(df)
                elif col == 'TotalCharges':
                    outliers[col] = self._detect_total_charges_outliers(df)
                elif col == 'tenure':
                    outliers[col] = self._detect_tenure_outliers(df)
                else:
                    # Use IQR for other numerical columns
                    iqr_detector = IQROutlierDetection()
                    col_outliers = iqr_detector.detect_outliers(df, [col])
                    outliers[col] = col_outliers[col]
            
            return outliers
            
        except Exception as e:
            logging.error(f"Error in Telco-specific outlier detection: {str(e)}")
            raise
    
    def _detect_monthly_charges_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers in MonthlyCharges using business logic"""
        if 'MonthlyCharges' not in df.columns:
            return pd.Series(False, index=df.index)
        
        # Business logic: MonthlyCharges should be reasonable for telecom services
        # Extremely low or high charges might be data entry errors
        reasonable_min = 10  # Minimum reasonable monthly charge
        reasonable_max = 200  # Maximum reasonable monthly charge
        
        outliers = (df['MonthlyCharges'] < reasonable_min) | (df['MonthlyCharges'] > reasonable_max)
        outlier_count = outliers.sum()
        
        logging.info(f"MonthlyCharges outliers (outside [{reasonable_min}, {reasonable_max}]): {outlier_count}")
        return outliers
    
    def _detect_total_charges_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers in TotalCharges using business logic"""
        if 'TotalCharges' not in df.columns:
            return pd.Series(False, index=df.index)
        
        outliers = pd.Series(False, index=df.index)
        
        # Business logic: TotalCharges should be consistent with MonthlyCharges and tenure
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            # Expected TotalCharges = MonthlyCharges * tenure (approximately)
            expected_total = df['MonthlyCharges'] * df['tenure']
            
            # Allow for some variation (e.g., price changes, promotions)
            tolerance = 0.5  # 50% tolerance
            lower_bound = expected_total * (1 - tolerance)
            upper_bound = expected_total * (1 + tolerance)
            
            # For customers with tenure > 0
            valid_mask = (df['tenure'] > 0) & (df['TotalCharges'].notna()) & (df['MonthlyCharges'].notna())
            outliers.loc[valid_mask] = (
                (df.loc[valid_mask, 'TotalCharges'] < lower_bound.loc[valid_mask]) |
                (df.loc[valid_mask, 'TotalCharges'] > upper_bound.loc[valid_mask])
            )
        
        outlier_count = outliers.sum()
        logging.info(f"TotalCharges outliers (inconsistent with MonthlyCharges*tenure): {outlier_count}")
        return outliers
    
    def _detect_tenure_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers in tenure using business logic"""
        if 'tenure' not in df.columns:
            return pd.Series(False, index=df.index)
        
        # Business logic: Tenure should be within reasonable bounds
        max_reasonable_tenure = 100  # Maximum reasonable tenure in months
        
        outliers = (df['tenure'] < 0) | (df['tenure'] > max_reasonable_tenure)
        outlier_count = outliers.sum()
        
        logging.info(f"Tenure outliers (outside [0, {max_reasonable_tenure}]): {outlier_count}")
        return outliers

class OutlierDetector:
    """Main outlier detector class that uses strategy pattern"""
    
    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initialize outlier detector with a specific strategy
        
        Args:
            strategy (OutlierDetectionStrategy): Outlier detection strategy to use
        """
        self._strategy = strategy
        logging.info(f"Initialized OutlierDetector with strategy: {type(strategy).__name__}")
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using the configured strategy
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        return self._strategy.detect_outliers(df, columns)
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        method: str = 'remove',
        min_outlier_columns: int = 2
    ) -> pd.DataFrame:
        """
        Handle outliers in the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            method (str): Method to handle outliers ('remove', 'cap', 'transform')
            min_outlier_columns (int): Minimum outlier columns to trigger row removal
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        try:
            initial_shape = df.shape
            
            # Detect outliers
            outliers = self.detect_outliers(df, columns)
            
            if method == 'remove':
                df_processed = self._remove_outliers(df, outliers, min_outlier_columns)
            elif method == 'cap':
                df_processed = self._cap_outliers(df, outliers, columns)
            elif method == 'transform':
                df_processed = self._transform_outliers(df, columns)
            else:
                raise ValueError(f"Unknown outlier handling method: {method}")
            
            final_shape = df_processed.shape
            rows_affected = initial_shape[0] - final_shape[0]
            
            logging.info(f"Outlier handling completed:")
            logging.info(f"  - Method: {method}")
            logging.info(f"  - Shape: {initial_shape} -> {final_shape}")
            logging.info(f"  - Rows affected: {rows_affected}")
            
            return df_processed
            
        except Exception as e:
            logging.error(f"Error in outlier handling: {str(e)}")
            raise
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame, 
        outliers: pd.DataFrame, 
        min_outlier_columns: int
    ) -> pd.DataFrame:
        """Remove rows with outliers in multiple columns"""
        outlier_count_per_row = outliers.sum(axis=1)
        rows_to_remove = outlier_count_per_row >= min_outlier_columns
        
        removed_count = rows_to_remove.sum()
        logging.info(f"Removing {removed_count} rows with outliers in >={min_outlier_columns} columns")
        
        return df[~rows_to_remove].copy()
    
    def _cap_outliers(self, df: pd.DataFrame, outliers: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Cap outliers at 5th and 95th percentiles"""
        df_capped = df.copy()
        
        for col in columns:
            if col in outliers.columns and col in df.columns:
                p5 = df[col].quantile(0.05)
                p95 = df[col].quantile(0.95)
                
                outlier_mask = outliers[col]
                capped_count = outlier_mask.sum()
                
                df_capped.loc[outlier_mask & (df[col] < p5), col] = p5
                df_capped.loc[outlier_mask & (df[col] > p95), col] = p95
                
                logging.info(f"Capped {capped_count} outliers in {col} to [{p5:.2f}, {p95:.2f}]")
        
        return df_capped
    
    def _transform_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to reduce outlier impact"""
        df_transformed = df.copy()
        
        for col in columns:
            if col in df.columns:
                # Apply log transformation for positive values
                if (df[col] > 0).all():
                    df_transformed[col] = np.log1p(df[col])
                    logging.info(f"Applied log transformation to {col}")
        
        return df_transformed

def detect_and_handle_outliers(
    df: pd.DataFrame,
    strategy: str = 'iqr',
    columns: Optional[List[str]] = None,
    method: str = 'remove',
    output_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Main function to detect and handle outliers in Telco dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Detection strategy ('iqr', 'zscore', 'modified_zscore', 'telco_specific')
        columns (Optional[List[str]]): Columns to check for outliers
        method (str): Method to handle outliers ('remove', 'cap', 'transform')
        output_path (Optional[str]): Path to save processed data
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    try:
        logging.info(f"Detecting and handling outliers using {strategy} strategy")
        
        # Use numerical columns if none specified
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            logging.info(f"Auto-detected numerical columns: {columns}")
        
        # Select appropriate strategy
        if strategy == 'iqr':
            detector_strategy = IQROutlierDetection(**kwargs)
        elif strategy == 'zscore':
            detector_strategy = ZScoreOutlierDetection(**kwargs)
        elif strategy == 'modified_zscore':
            detector_strategy = ModifiedZScoreOutlierDetection(**kwargs)
        elif strategy == 'telco_specific':
            detector_strategy = TelcoSpecificOutlierDetection()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create detector and handle outliers
        detector = OutlierDetector(detector_strategy)
        df_processed = detector.handle_outliers(df, columns, method, **kwargs)
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_processed.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df_processed
        
    except Exception as e:
        logging.error(f"Outlier detection and handling failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from previous steps)
        input_file = "data/processed/telco_missing_handled.csv"
        output_file = "data/processed/telco_outliers_removed.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Detect and handle outliers
            df_processed = detect_and_handle_outliers(
                df=df,
                strategy='telco_specific',
                method='remove',
                output_path=output_file,
                min_outlier_columns=2
            )
            
            print(f"Outlier detection and handling completed successfully!")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"Numerical columns processed: {df_processed.select_dtypes(include=[np.number]).columns.tolist()}")
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run previous data processing steps first")
            
    except Exception as e:
        print(f"Outlier detection and handling failed: {e}")
"""
Telco Customer Churn Data Splitting Module

This module provides strategies for splitting the Telco dataset into train/test sets,
with special focus on maintaining class distribution for the imbalanced churn target variable.

Based on Week 05_06 data splitting pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SplitType(str, Enum):
    """Enumeration for data splitting types"""
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'
    TELCO_SPECIFIC = 'telco_specific'

class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies"""
    
    @abstractmethod
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Abstract method to split data into train/test sets"""
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize simple train-test split strategy
        
        Args:
            test_size (float): Proportion of data for test set
            random_state (int): Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        logging.info(f"Initialized SimpleTrainTestSplitStrategy: test_size={test_size}")
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using simple random sampling
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            # Separate features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Perform split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
            # Log split results
            logging.info(f"Simple split completed:")
            logging.info(f"  - Train set: {X_train.shape[0]} samples")
            logging.info(f"  - Test set: {X_test.shape[0]} samples")
            logging.info(f"  - Train class distribution: {Counter(y_train)}")
            logging.info(f"  - Test class distribution: {Counter(y_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error in simple train-test split: {str(e)}")
            raise

class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """Stratified train-test split strategy to maintain class distribution"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize stratified train-test split strategy
        
        Args:
            test_size (float): Proportion of data for test set
            random_state (int): Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        logging.info(f"Initialized StratifiedTrainTestSplitStrategy: test_size={test_size}")
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using stratified sampling to maintain class distribution
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            # Separate features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Check class distribution before split
            original_distribution = Counter(y)
            logging.info(f"Original class distribution: {original_distribution}")
            
            # Perform stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                stratify=y,
                random_state=self.random_state
            )
            
            # Log split results
            train_distribution = Counter(y_train)
            test_distribution = Counter(y_test)
            
            logging.info(f"Stratified split completed:")
            logging.info(f"  - Train set: {X_train.shape[0]} samples")
            logging.info(f"  - Test set: {X_test.shape[0]} samples")
            logging.info(f"  - Train class distribution: {train_distribution}")
            logging.info(f"  - Test class distribution: {test_distribution}")
            
            # Calculate and log class proportions
            for class_label in original_distribution.keys():
                original_prop = original_distribution[class_label] / len(y)
                train_prop = train_distribution[class_label] / len(y_train)
                test_prop = test_distribution[class_label] / len(y_test)
                
                logging.info(f"  - Class {class_label} proportions - Original: {original_prop:.3f}, Train: {train_prop:.3f}, Test: {test_prop:.3f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error in stratified train-test split: {str(e)}")
            raise

class TelcoSpecificSplitStrategy(DataSplittingStrategy):
    """Telco domain-specific splitting strategy with additional validation"""
    
    def __init__(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42,
        ensure_tenure_distribution: bool = True,
        min_samples_per_class: int = 10
    ):
        """
        Initialize Telco-specific splitting strategy
        
        Args:
            test_size (float): Proportion of data for test set
            random_state (int): Random state for reproducibility
            ensure_tenure_distribution (bool): Ensure tenure distribution is maintained
            min_samples_per_class (int): Minimum samples required per class in each split
        """
        self.test_size = test_size
        self.random_state = random_state
        self.ensure_tenure_distribution = ensure_tenure_distribution
        self.min_samples_per_class = min_samples_per_class
        logging.info(f"Initialized TelcoSpecificSplitStrategy: test_size={test_size}")
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using Telco-specific logic with additional validations
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            # Separate features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Validate minimum samples per class
            class_counts = Counter(y)
            for class_label, count in class_counts.items():
                min_required = self.min_samples_per_class / (1 - self.test_size)  # Adjust for split ratio
                if count < min_required:
                    logging.warning(f"Class {class_label} has only {count} samples, which may be insufficient")
            
            # Perform stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                stratify=y,
                random_state=self.random_state
            )
            
            # Validate splits
            self._validate_splits(X_train, X_test, y_train, y_test, df)
            
            # Additional Telco-specific validations
            if self.ensure_tenure_distribution:
                self._validate_tenure_distribution(X_train, X_test)
            
            self._validate_service_distribution(X_train, X_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error in Telco-specific split: {str(e)}")
            raise
    
    def _validate_splits(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series,
        original_df: pd.DataFrame
    ) -> None:
        """Validate the quality of the splits"""
        
        # Check that splits don't have overlapping indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        if train_indices.intersection(test_indices):
            raise ValueError("Train and test sets have overlapping indices")
        
        # Check that all data is accounted for
        total_samples = len(X_train) + len(X_test)
        if total_samples != len(original_df):
            raise ValueError(f"Sample count mismatch: {total_samples} != {len(original_df)}")
        
        # Check minimum samples per class in each split
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        
        for class_label in train_class_counts.keys():
            if train_class_counts[class_label] < self.min_samples_per_class:
                logging.warning(f"Train set has only {train_class_counts[class_label]} samples for class {class_label}")
            
            if test_class_counts[class_label] < self.min_samples_per_class:
                logging.warning(f"Test set has only {test_class_counts[class_label]} samples for class {class_label}")
        
        logging.info("Split validation completed successfully")
    
    def _validate_tenure_distribution(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """Validate that tenure distribution is maintained across splits"""
        if 'tenure' not in X_train.columns:
            logging.info("Tenure column not found, skipping tenure distribution validation")
            return
        
        # Compare tenure distributions using statistical measures
        train_tenure_mean = X_train['tenure'].mean()
        test_tenure_mean = X_test['tenure'].mean()
        
        train_tenure_std = X_train['tenure'].std()
        test_tenure_std = X_test['tenure'].std()
        
        logging.info(f"Tenure distribution validation:")
        logging.info(f"  - Train tenure: mean={train_tenure_mean:.2f}, std={train_tenure_std:.2f}")
        logging.info(f"  - Test tenure: mean={test_tenure_mean:.2f}, std={test_tenure_std:.2f}")
        
        # Check if distributions are significantly different (simple heuristic)
        mean_diff_ratio = abs(train_tenure_mean - test_tenure_mean) / max(train_tenure_mean, test_tenure_mean)
        if mean_diff_ratio > 0.1:  # 10% difference threshold
            logging.warning(f"Large difference in tenure means: {mean_diff_ratio:.3f}")
    
    def _validate_service_distribution(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """Validate that service adoption patterns are maintained across splits"""
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Filter for existing service columns
        existing_service_cols = [col for col in service_columns if col in X_train.columns]
        
        if not existing_service_cols:
            logging.info("No service columns found, skipping service distribution validation")
            return
        
        logging.info("Service distribution validation:")
        for col in existing_service_cols[:3]:  # Check first 3 to avoid too much output
            train_dist = X_train[col].value_counts(normalize=True)
            test_dist = X_test[col].value_counts(normalize=True)
            
            logging.info(f"  - {col}: Train {train_dist.to_dict()}, Test {test_dist.to_dict()}")

class DataSplitter:
    """Main data splitter class that coordinates different splitting strategies"""
    
    def __init__(self, artifacts_path: str = "artifacts/data"):
        """
        Initialize data splitter
        
        Args:
            artifacts_path (str): Path to save split data artifacts
        """
        self.artifacts_path = artifacts_path
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized DataSplitter")
    
    def split_and_save(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: DataSplittingStrategy,
        save_splits: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using specified strategy and optionally save results
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            strategy (DataSplittingStrategy): Splitting strategy to use
            save_splits (bool): Whether to save split data to files
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            # Perform split
            X_train, X_test, y_train, y_test = strategy.split_data(df, target_column)
            
            # Save splits if requested
            if save_splits:
                self._save_splits(X_train, X_test, y_train, y_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error in split and save: {str(e)}")
            raise
    
    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """Save split data to CSV files"""
        try:
            # Save feature splits
            X_train.to_csv(os.path.join(self.artifacts_path, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(self.artifacts_path, "X_test.csv"), index=False)
            
            # Save target splits
            y_train_df = pd.DataFrame({'y_train': y_train})
            y_test_df = pd.DataFrame({'y_test': y_test})
            
            y_train_df.to_csv(os.path.join(self.artifacts_path, "y_train.csv"), index=False)
            y_test_df.to_csv(os.path.join(self.artifacts_path, "y_test.csv"), index=False)
            
            logging.info(f"Saved split data to: {self.artifacts_path}")
            logging.info(f"  - X_train: {X_train.shape}")
            logging.info(f"  - X_test: {X_test.shape}")
            logging.info(f"  - y_train: {y_train.shape}")
            logging.info(f"  - y_test: {y_test.shape}")
            
        except Exception as e:
            logging.error(f"Error saving splits: {str(e)}")
            raise

def split_telco_data(
    df: pd.DataFrame,
    target_column: str = 'Churn',
    split_type: str = 'stratified',
    test_size: float = 0.2,
    output_path: Optional[str] = None,
    save_splits: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main function to split Telco customer churn data
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        split_type (str): Type of split ('simple', 'stratified', 'telco_specific')
        test_size (float): Proportion of data for test set
        output_path (Optional[str]): Path to save split data
        save_splits (bool): Whether to save split data to files
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    try:
        logging.info(f"Splitting Telco data using {split_type} strategy")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Set output path
        artifacts_path = output_path or "artifacts/data"
        
        # Initialize splitter
        splitter = DataSplitter(artifacts_path)
        
        # Select appropriate strategy
        if split_type == 'simple':
            strategy = SimpleTrainTestSplitStrategy(test_size=test_size, **kwargs)
        elif split_type == 'stratified':
            strategy = StratifiedTrainTestSplitStrategy(test_size=test_size, **kwargs)
        elif split_type == 'telco_specific':
            strategy = TelcoSpecificSplitStrategy(test_size=test_size, **kwargs)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        # Perform split
        X_train, X_test, y_train, y_test = splitter.split_and_save(
            df, target_column, strategy, save_splits
        )
        
        # Log final results
        logging.info(f"Data splitting completed successfully:")
        logging.info(f"  - Strategy: {split_type}")
        logging.info(f"  - Total samples: {len(df)}")
        logging.info(f"  - Train samples: {len(X_train)}")
        logging.info(f"  - Test samples: {len(X_test)}")
        logging.info(f"  - Test size ratio: {len(X_test)/len(df):.3f}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Data splitting failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from previous steps)
        input_file = "data/processed/telco_scaled.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Split data using stratified strategy
            X_train, X_test, y_train, y_test = split_telco_data(
                df=df,
                target_column='Churn',
                split_type='stratified',
                test_size=0.2,
                output_path="artifacts/data",
                save_splits=True,
                random_state=42
            )
            
            print(f"Data splitting completed successfully!")
            print(f"Train set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Train target distribution: {Counter(y_train)}")
            print(f"Test target distribution: {Counter(y_test)}")
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run previous data processing steps first")
            
    except Exception as e:
        print(f"Data splitting failed: {e}")
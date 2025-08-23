"""
Telco Customer Churn Feature Encoding Module

This module provides strategies for encoding categorical features in the Telco dataset,
including nominal encoding, ordinal encoding, and one-hot encoding for various categorical variables.

Based on Week 05_06 feature encoding pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VariableType(str, Enum):
    """Enumeration for variable types"""
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'
    BINARY = 'binary'

class FeatureEncodingStrategy(ABC):
    """Abstract base class for feature encoding strategies"""
    
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to encode features"""
        pass
    
    @abstractmethod
    def save_encoders(self, path: str) -> None:
        """Abstract method to save encoder artifacts"""
        pass

class LabelEncodingStrategy(FeatureEncodingStrategy):
    """Label encoding strategy for nominal categorical variables"""
    
    def __init__(self, columns: List[str], artifacts_path: str = "artifacts/encode"):
        """
        Initialize label encoding strategy
        
        Args:
            columns (List[str]): Columns to encode
            artifacts_path (str): Path to save encoder artifacts
        """
        self.columns = columns
        self.artifacts_path = artifacts_path
        self.encoders = {}
        self.encoder_mappings = {}
        
        # Ensure artifacts directory exists
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info(f"Initialized LabelEncodingStrategy for columns: {columns}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using label encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with label encoded features
        """
        try:
            df_encoded = df.copy()
            
            for column in self.columns:
                if column not in df_encoded.columns:
                    logging.warning(f"Column {column} not found in dataframe")
                    continue
                
                # Create and fit label encoder
                encoder = LabelEncoder()
                
                # Handle missing values by filling with 'Unknown'
                df_encoded[column] = df_encoded[column].fillna('Unknown')
                
                # Fit and transform
                encoded_values = encoder.fit_transform(df_encoded[column])
                df_encoded[column] = encoded_values
                
                # Store encoder and mapping
                self.encoders[column] = encoder
                self.encoder_mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                
                logging.info(f"Label encoded {column}: {len(encoder.classes_)} unique values")
                logging.info(f"  Mapping: {self.encoder_mappings[column]}")
            
            return df_encoded
            
        except Exception as e:
            logging.error(f"Error in label encoding: {str(e)}")
            raise
    
    def save_encoders(self, path: str = None) -> None:
        """Save encoders and mappings to disk"""
        save_path = path or self.artifacts_path
        
        for column, encoder in self.encoders.items():
            # Save sklearn encoder
            encoder_file = os.path.join(save_path, f"{column}_label_encoder.pkl")
            joblib.dump(encoder, encoder_file)
            
            # Save mapping as JSON
            mapping_file = os.path.join(save_path, f"{column}_label_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(self.encoder_mappings[column], f, indent=2)
            
            logging.info(f"Saved {column} label encoder to {encoder_file}")

class OneHotEncodingStrategy(FeatureEncodingStrategy):
    """One-hot encoding strategy for nominal categorical variables"""
    
    def __init__(self, columns: List[str], drop_first: bool = True, artifacts_path: str = "artifacts/encode"):
        """
        Initialize one-hot encoding strategy
        
        Args:
            columns (List[str]): Columns to encode
            drop_first (bool): Whether to drop first category to avoid dummy variable trap
            artifacts_path (str): Path to save encoder artifacts
        """
        self.columns = columns
        self.drop_first = drop_first
        self.artifacts_path = artifacts_path
        self.encoders = {}
        self.feature_names = {}
        
        # Ensure artifacts directory exists
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info(f"Initialized OneHotEncodingStrategy for columns: {columns}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using one-hot encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with one-hot encoded features
        """
        try:
            df_encoded = df.copy()
            
            for column in self.columns:
                if column not in df_encoded.columns:
                    logging.warning(f"Column {column} not found in dataframe")
                    continue
                
                # Handle missing values
                df_encoded[column] = df_encoded[column].fillna('Unknown')
                
                # Create one-hot encoded features using pandas get_dummies
                dummy_df = pd.get_dummies(
                    df_encoded[column], 
                    prefix=column,
                    drop_first=self.drop_first
                )
                
                # Store feature names for later use
                self.feature_names[column] = dummy_df.columns.tolist()
                
                # Add dummy columns to dataframe
                df_encoded = pd.concat([df_encoded, dummy_df], axis=1)
                
                # Remove original column
                df_encoded = df_encoded.drop(columns=[column])
                
                logging.info(f"One-hot encoded {column}: created {len(dummy_df.columns)} features")
                logging.info(f"  Features: {dummy_df.columns.tolist()}")
            
            return df_encoded
            
        except Exception as e:
            logging.error(f"Error in one-hot encoding: {str(e)}")
            raise
    
    def save_encoders(self, path: str = None) -> None:
        """Save feature names and encoding information"""
        save_path = path or self.artifacts_path
        
        # Save feature names
        feature_names_file = os.path.join(save_path, "onehot_feature_names.json")
        with open(feature_names_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        logging.info(f"Saved one-hot feature names to {feature_names_file}")

class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """Ordinal encoding strategy for ordered categorical variables"""
    
    def __init__(self, ordinal_mappings: Dict[str, List[str]], artifacts_path: str = "artifacts/encode"):
        """
        Initialize ordinal encoding strategy
        
        Args:
            ordinal_mappings (Dict[str, List[str]]): Mapping of columns to ordered categories
            artifacts_path (str): Path to save encoder artifacts
        """
        self.ordinal_mappings = ordinal_mappings
        self.artifacts_path = artifacts_path
        self.encoders = {}
        
        # Ensure artifacts directory exists
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info(f"Initialized OrdinalEncodingStrategy for columns: {list(ordinal_mappings.keys())}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using ordinal encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with ordinal encoded features
        """
        try:
            df_encoded = df.copy()
            
            for column, categories in self.ordinal_mappings.items():
                if column not in df_encoded.columns:
                    logging.warning(f"Column {column} not found in dataframe")
                    continue
                
                # Create mapping dictionary
                mapping = {category: i for i, category in enumerate(categories)}
                
                # Handle missing values
                df_encoded[column] = df_encoded[column].fillna('Unknown')
                
                # Add 'Unknown' to mapping if needed
                if 'Unknown' not in mapping and 'Unknown' in df_encoded[column].values:
                    mapping['Unknown'] = len(categories)
                
                # Apply mapping
                df_encoded[column] = df_encoded[column].map(mapping)
                
                # Handle unmapped values
                unmapped_mask = df_encoded[column].isna()
                if unmapped_mask.any():
                    df_encoded.loc[unmapped_mask, column] = -1  # Code for unknown
                    logging.warning(f"Found unmapped values in {column}, encoded as -1")
                
                self.encoders[column] = mapping
                
                logging.info(f"Ordinal encoded {column}: {len(mapping)} categories")
                logging.info(f"  Mapping: {mapping}")
            
            return df_encoded
            
        except Exception as e:
            logging.error(f"Error in ordinal encoding: {str(e)}")
            raise
    
    def save_encoders(self, path: str = None) -> None:
        """Save ordinal mappings to disk"""
        save_path = path or self.artifacts_path
        
        for column, mapping in self.encoders.items():
            mapping_file = os.path.join(save_path, f"{column}_ordinal_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            
            logging.info(f"Saved {column} ordinal mapping to {mapping_file}")

class TelcoSpecificEncodingStrategy(FeatureEncodingStrategy):
    """Telco domain-specific encoding strategy"""
    
    def __init__(self, artifacts_path: str = "artifacts/encode"):
        """
        Initialize Telco-specific encoding strategy
        
        Args:
            artifacts_path (str): Path to save encoder artifacts
        """
        self.artifacts_path = artifacts_path
        self.encoders = {}
        
        # Define Telco-specific ordinal mappings
        self.ordinal_mappings = {
            'Contract': ['Month-to-month', 'One year', 'Two year'],  # Increasing commitment
            'InternetService': ['No', 'DSL', 'Fiber optic'],  # Increasing speed/quality
            'PaymentMethod': ['Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
        }
        
        # Define binary columns (Yes/No)
        self.binary_columns = [
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Define nominal columns for one-hot encoding
        self.nominal_columns = ['gender', 'PaymentMethod']
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized TelcoSpecificEncodingStrategy")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Telco-specific encoding strategies
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Telco-specific encoding applied
        """
        try:
            df_encoded = df.copy()
            
            # Apply binary encoding
            df_encoded = self._encode_binary_features(df_encoded)
            
            # Apply ordinal encoding
            df_encoded = self._encode_ordinal_features(df_encoded)
            
            # Apply one-hot encoding for nominal features
            df_encoded = self._encode_nominal_features(df_encoded)
            
            # Handle special cases
            df_encoded = self._handle_special_cases(df_encoded)
            
            return df_encoded
            
        except Exception as e:
            logging.error(f"Error in Telco-specific encoding: {str(e)}")
            raise
    
    def _encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode binary Yes/No features"""
        for column in self.binary_columns:
            if column in df.columns:
                # Handle missing values
                df[column] = df[column].fillna('No')
                
                # Create binary mapping
                if column == 'Churn':
                    mapping = {'No': 0, 'Yes': 1}  # Target variable
                else:
                    mapping = {'No': 0, 'Yes': 1}
                
                # Handle special cases for service columns
                if column in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                    mapping.update({'No internet service': 0, 'No phone service': 0})
                
                df[column] = df[column].map(mapping).fillna(0).astype(int)
                self.encoders[f'{column}_binary'] = mapping
                
                logging.info(f"Binary encoded {column}: {mapping}")
        
        return df
    
    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ordinal features with natural ordering"""
        for column, categories in self.ordinal_mappings.items():
            if column in df.columns:
                mapping = {category: i for i, category in enumerate(categories)}
                
                # Handle missing values
                df[column] = df[column].fillna(categories[0])  # Default to first category
                
                # Apply mapping
                df[column] = df[column].map(mapping).fillna(0).astype(int)
                self.encoders[f'{column}_ordinal'] = mapping
                
                logging.info(f"Ordinal encoded {column}: {mapping}")
        
        return df
    
    def _encode_nominal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode nominal features using one-hot encoding"""
        for column in self.nominal_columns:
            if column in df.columns:
                # Handle missing values
                df[column] = df[column].fillna('Unknown')
                
                # Create dummy variables
                dummy_df = pd.get_dummies(df[column], prefix=column, drop_first=True)
                
                # Add dummy columns and remove original
                df = pd.concat([df, dummy_df], axis=1)
                df = df.drop(columns=[column])
                
                self.encoders[f'{column}_onehot'] = dummy_df.columns.tolist()
                
                logging.info(f"One-hot encoded {column}: {dummy_df.columns.tolist()}")
        
        return df
    
    def _handle_special_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle special encoding cases for Telco dataset"""
        # Handle SeniorCitizen (already binary 0/1)
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
            logging.info("SeniorCitizen: kept as binary (0/1)")
        
        # Handle customerID (should be dropped for modeling)
        if 'customerID' in df.columns:
            # Create a flag if needed, then drop
            df['customer_id_length'] = df['customerID'].str.len()
            df = df.drop(columns=['customerID'])
            logging.info("Dropped customerID, created customer_id_length feature")
        
        return df
    
    def save_encoders(self, path: str = None) -> None:
        """Save all encoders and mappings"""
        save_path = path or self.artifacts_path
        
        # Save all encoder mappings
        encoders_file = os.path.join(save_path, "telco_encoders.json")
        with open(encoders_file, 'w') as f:
            json.dump(self.encoders, f, indent=2)
        
        logging.info(f"Saved Telco encoders to {encoders_file}")

class FeatureEncoder:
    """Main feature encoder class that coordinates different encoding strategies"""
    
    def __init__(self, artifacts_path: str = "artifacts/encode"):
        """
        Initialize feature encoder
        
        Args:
            artifacts_path (str): Path to save encoder artifacts
        """
        self.artifacts_path = artifacts_path
        self.applied_strategies = []
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized FeatureEncoder")
    
    def apply_encoding_strategy(
        self, 
        df: pd.DataFrame, 
        strategy: FeatureEncodingStrategy
    ) -> pd.DataFrame:
        """
        Apply an encoding strategy to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (FeatureEncodingStrategy): Encoding strategy to apply
            
        Returns:
            pd.DataFrame: Encoded dataframe
        """
        try:
            logging.info(f"Applying {type(strategy).__name__}")
            df_encoded = strategy.encode(df)
            strategy.save_encoders(self.artifacts_path)
            self.applied_strategies.append(strategy)
            return df_encoded
            
        except Exception as e:
            logging.error(f"Error applying encoding strategy: {str(e)}")
            raise

def encode_telco_features(
    df: pd.DataFrame,
    encoding_type: str = 'telco_specific',
    output_path: Optional[str] = None,
    artifacts_path: str = "artifacts/encode",
    **kwargs
) -> pd.DataFrame:
    """
    Main function to encode categorical features in Telco dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoding_type (str): Type of encoding ('label', 'onehot', 'ordinal', 'telco_specific')
        output_path (Optional[str]): Path to save processed data
        artifacts_path (str): Path to save encoder artifacts
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        pd.DataFrame: Encoded dataframe
    """
    try:
        logging.info(f"Encoding categorical features using {encoding_type} strategy")
        
        # Initialize encoder
        encoder = FeatureEncoder(artifacts_path)
        
        if encoding_type == 'telco_specific':
            # Use Telco-specific encoding strategy
            strategy = TelcoSpecificEncodingStrategy(artifacts_path)
            df_encoded = encoder.apply_encoding_strategy(df, strategy)
            
        elif encoding_type == 'label':
            # Label encoding for all categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if 'customerID' in categorical_columns:
                categorical_columns.remove('customerID')  # Exclude ID column
            
            strategy = LabelEncodingStrategy(categorical_columns, artifacts_path)
            df_encoded = encoder.apply_encoding_strategy(df, strategy)
            
        elif encoding_type == 'onehot':
            # One-hot encoding for all categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if 'customerID' in categorical_columns:
                categorical_columns.remove('customerID')  # Exclude ID column
            
            strategy = OneHotEncodingStrategy(categorical_columns, artifacts_path=artifacts_path, **kwargs)
            df_encoded = encoder.apply_encoding_strategy(df, strategy)
            
        elif encoding_type == 'ordinal':
            # Ordinal encoding with custom mappings
            ordinal_mappings = kwargs.get('ordinal_mappings', {})
            if not ordinal_mappings:
                raise ValueError("ordinal_mappings must be provided for ordinal encoding")
            
            strategy = OrdinalEncodingStrategy(ordinal_mappings, artifacts_path)
            df_encoded = encoder.apply_encoding_strategy(df, strategy)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # Log encoding results
        categorical_before = df.select_dtypes(include=['object']).shape[1]
        categorical_after = df_encoded.select_dtypes(include=['object']).shape[1]
        
        logging.info(f"Feature encoding completed:")
        logging.info(f"  - Categorical columns before: {categorical_before}")
        logging.info(f"  - Categorical columns after: {categorical_after}")
        logging.info(f"  - Total columns: {df.shape[1]} -> {df_encoded.shape[1]}")
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_encoded.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df_encoded
        
    except Exception as e:
        logging.error(f"Feature encoding failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from previous steps)
        input_file = "data/processed/telco_binned.csv"
        output_file = "data/processed/telco_encoded.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Apply Telco-specific encoding
            df_processed = encode_telco_features(
                df=df,
                encoding_type='telco_specific',
                output_path=output_file,
                artifacts_path="artifacts/encode"
            )
            
            print(f"Feature encoding completed successfully!")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"Data types after encoding:")
            print(df_processed.dtypes.value_counts().to_string())
            print(f"\nCategorical columns remaining:")
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            print(categorical_cols if categorical_cols else "None")
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run previous data processing steps first")
            
    except Exception as e:
        print(f"Feature encoding failed: {e}")
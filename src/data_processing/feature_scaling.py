"""
Telco Customer Churn Feature Scaling Module

This module provides strategies for scaling numerical features in the Telco dataset,
specifically for MonthlyCharges, TotalCharges, and other numerical features.

Based on Week 05_06 feature scaling pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScalingType(str, Enum):
    """Enumeration for scaling types"""
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    POWER = 'power'

class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies"""
    
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Abstract method to scale features"""
        pass
    
    @abstractmethod
    def save_scaler(self, path: str) -> None:
        """Abstract method to save scaler artifacts"""
        pass

class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling (Z-score normalization) strategy"""
    
    def __init__(self, artifacts_path: str = "artifacts/scalers"):
        """
        Initialize standard scaling strategy
        
        Args:
            artifacts_path (str): Path to save scaler artifacts
        """
        self.scaler = StandardScaler()
        self.fitted = False
        self.artifacts_path = artifacts_path
        self.scaled_columns = []
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized StandardScalingStrategy")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply standard scaling to specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Filter columns that exist in dataframe
            existing_columns = [col for col in columns_to_scale if col in df_scaled.columns]
            
            if not existing_columns:
                logging.warning("No columns found for scaling")
                return df_scaled
            
            # Apply scaling
            scaled_values = self.scaler.fit_transform(df_scaled[existing_columns])
            df_scaled[existing_columns] = scaled_values
            
            self.fitted = True
            self.scaled_columns = existing_columns
            
            # Log scaling statistics
            for i, col in enumerate(existing_columns):
                mean = self.scaler.mean_[i]
                std = self.scaler.scale_[i]
                logging.info(f"Standard scaled {col}: mean={mean:.4f}, std={std:.4f}")
            
            logging.info(f"Applied Standard scaling to columns: {existing_columns}")
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error in standard scaling: {str(e)}")
            raise
    
    def save_scaler(self, path: str = None) -> None:
        """Save the fitted scaler"""
        if not self.fitted:
            logging.warning("Scaler not fitted yet, cannot save")
            return
        
        save_path = path or self.artifacts_path
        scaler_file = os.path.join(save_path, "standard_scaler.pkl")
        
        scaler_info = {
            'scaler': self.scaler,
            'scaled_columns': self.scaled_columns,
            'scaling_type': 'standard'
        }
        
        joblib.dump(scaler_info, scaler_file)
        logging.info(f"Saved standard scaler to {scaler_file}")

class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy"""
    
    def __init__(self, feature_range: tuple = (0, 1), artifacts_path: str = "artifacts/scalers"):
        """
        Initialize Min-Max scaling strategy
        
        Args:
            feature_range (tuple): Target range for scaling
            artifacts_path (str): Path to save scaler artifacts
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.fitted = False
        self.feature_range = feature_range
        self.artifacts_path = artifacts_path
        self.scaled_columns = []
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info(f"Initialized MinMaxScalingStrategy with range {feature_range}")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply Min-Max scaling to specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Filter columns that exist in dataframe
            existing_columns = [col for col in columns_to_scale if col in df_scaled.columns]
            
            if not existing_columns:
                logging.warning("No columns found for scaling")
                return df_scaled
            
            # Apply scaling
            scaled_values = self.scaler.fit_transform(df_scaled[existing_columns])
            df_scaled[existing_columns] = scaled_values
            
            self.fitted = True
            self.scaled_columns = existing_columns
            
            # Log scaling statistics
            for i, col in enumerate(existing_columns):
                min_val = self.scaler.data_min_[i]
                max_val = self.scaler.data_max_[i]
                logging.info(f"MinMax scaled {col}: original_range=[{min_val:.4f}, {max_val:.4f}] -> {self.feature_range}")
            
            logging.info(f"Applied Min-Max scaling to columns: {existing_columns}")
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error in Min-Max scaling: {str(e)}")
            raise
    
    def save_scaler(self, path: str = None) -> None:
        """Save the fitted scaler"""
        if not self.fitted:
            logging.warning("Scaler not fitted yet, cannot save")
            return
        
        save_path = path or self.artifacts_path
        scaler_file = os.path.join(save_path, "minmax_scaler.pkl")
        
        scaler_info = {
            'scaler': self.scaler,
            'scaled_columns': self.scaled_columns,
            'scaling_type': 'minmax',
            'feature_range': self.feature_range
        }
        
        joblib.dump(scaler_info, scaler_file)
        logging.info(f"Saved MinMax scaler to {scaler_file}")

class RobustScalingStrategy(FeatureScalingStrategy):
    """Robust scaling strategy (uses median and IQR)"""
    
    def __init__(self, artifacts_path: str = "artifacts/scalers"):
        """
        Initialize robust scaling strategy
        
        Args:
            artifacts_path (str): Path to save scaler artifacts
        """
        self.scaler = RobustScaler()
        self.fitted = False
        self.artifacts_path = artifacts_path
        self.scaled_columns = []
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized RobustScalingStrategy")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply robust scaling to specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Filter columns that exist in dataframe
            existing_columns = [col for col in columns_to_scale if col in df_scaled.columns]
            
            if not existing_columns:
                logging.warning("No columns found for scaling")
                return df_scaled
            
            # Apply scaling
            scaled_values = self.scaler.fit_transform(df_scaled[existing_columns])
            df_scaled[existing_columns] = scaled_values
            
            self.fitted = True
            self.scaled_columns = existing_columns
            
            # Log scaling statistics
            for i, col in enumerate(existing_columns):
                median = self.scaler.center_[i]
                scale = self.scaler.scale_[i]
                logging.info(f"Robust scaled {col}: median={median:.4f}, scale={scale:.4f}")
            
            logging.info(f"Applied Robust scaling to columns: {existing_columns}")
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error in robust scaling: {str(e)}")
            raise
    
    def save_scaler(self, path: str = None) -> None:
        """Save the fitted scaler"""
        if not self.fitted:
            logging.warning("Scaler not fitted yet, cannot save")
            return
        
        save_path = path or self.artifacts_path
        scaler_file = os.path.join(save_path, "robust_scaler.pkl")
        
        scaler_info = {
            'scaler': self.scaler,
            'scaled_columns': self.scaled_columns,
            'scaling_type': 'robust'
        }
        
        joblib.dump(scaler_info, scaler_file)
        logging.info(f"Saved Robust scaler to {scaler_file}")

class PowerTransformScalingStrategy(FeatureScalingStrategy):
    """Power transformation scaling strategy (Yeo-Johnson or Box-Cox)"""
    
    def __init__(self, method: str = 'yeo-johnson', artifacts_path: str = "artifacts/scalers"):
        """
        Initialize power transform scaling strategy
        
        Args:
            method (str): Transformation method ('yeo-johnson' or 'box-cox')
            artifacts_path (str): Path to save scaler artifacts
        """
        self.scaler = PowerTransformer(method=method, standardize=True)
        self.fitted = False
        self.method = method
        self.artifacts_path = artifacts_path
        self.scaled_columns = []
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info(f"Initialized PowerTransformScalingStrategy with {method}")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply power transform scaling to specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Filter columns that exist in dataframe
            existing_columns = [col for col in columns_to_scale if col in df_scaled.columns]
            
            if not existing_columns:
                logging.warning("No columns found for scaling")
                return df_scaled
            
            # For Box-Cox, ensure all values are positive
            if self.method == 'box-cox':
                for col in existing_columns:
                    if (df_scaled[col] <= 0).any():
                        logging.warning(f"Box-Cox requires positive values. Adding constant to {col}")
                        df_scaled[col] = df_scaled[col] - df_scaled[col].min() + 1
            
            # Apply scaling
            scaled_values = self.scaler.fit_transform(df_scaled[existing_columns])
            df_scaled[existing_columns] = scaled_values
            
            self.fitted = True
            self.scaled_columns = existing_columns
            
            # Log scaling statistics
            for i, col in enumerate(existing_columns):
                lambda_val = self.scaler.lambdas_[i]
                logging.info(f"Power transformed {col}: lambda={lambda_val:.4f}")
            
            logging.info(f"Applied Power Transform scaling to columns: {existing_columns}")
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error in power transform scaling: {str(e)}")
            raise
    
    def save_scaler(self, path: str = None) -> None:
        """Save the fitted scaler"""
        if not self.fitted:
            logging.warning("Scaler not fitted yet, cannot save")
            return
        
        save_path = path or self.artifacts_path
        scaler_file = os.path.join(save_path, f"power_transform_{self.method}_scaler.pkl")
        
        scaler_info = {
            'scaler': self.scaler,
            'scaled_columns': self.scaled_columns,
            'scaling_type': 'power',
            'method': self.method
        }
        
        joblib.dump(scaler_info, scaler_file)
        logging.info(f"Saved Power Transform scaler to {scaler_file}")

class TelcoSpecificScalingStrategy(FeatureScalingStrategy):
    """Telco domain-specific scaling strategy"""
    
    def __init__(self, artifacts_path: str = "artifacts/scalers"):
        """
        Initialize Telco-specific scaling strategy
        
        Args:
            artifacts_path (str): Path to save scaler artifacts
        """
        self.artifacts_path = artifacts_path
        self.scalers = {}
        self.scaled_columns = []
        
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized TelcoSpecificScalingStrategy")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str] = None) -> pd.DataFrame:
        """
        Apply Telco-specific scaling to numerical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_scale (List[str]): Columns to scale (auto-detected if None)
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Auto-detect numerical columns if not specified
            if columns_to_scale is None:
                columns_to_scale = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
                # Remove ID-like columns
                columns_to_scale = [col for col in columns_to_scale if 'id' not in col.lower()]
            
            # Apply different scaling strategies for different types of features
            df_scaled = self._scale_charge_features(df_scaled, columns_to_scale)
            df_scaled = self._scale_tenure_features(df_scaled, columns_to_scale)
            df_scaled = self._scale_count_features(df_scaled, columns_to_scale)
            df_scaled = self._scale_other_features(df_scaled, columns_to_scale)
            
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error in Telco-specific scaling: {str(e)}")
            raise
    
    def _scale_charge_features(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Scale charge-related features using robust scaling (less sensitive to outliers)"""
        charge_columns = [col for col in ['MonthlyCharges', 'TotalCharges'] if col in columns_to_scale]
        
        if charge_columns:
            scaler = RobustScaler()
            df[charge_columns] = scaler.fit_transform(df[charge_columns])
            self.scalers['charge_scaler'] = scaler
            logging.info(f"Robust scaled charge features: {charge_columns}")
        
        return df
    
    def _scale_tenure_features(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Scale tenure using Min-Max scaling"""
        tenure_columns = [col for col in ['tenure'] if col in columns_to_scale]
        
        if tenure_columns:
            scaler = MinMaxScaler()
            df[tenure_columns] = scaler.fit_transform(df[tenure_columns])
            self.scalers['tenure_scaler'] = scaler
            logging.info(f"MinMax scaled tenure features: {tenure_columns}")
        
        return df
    
    def _scale_count_features(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Scale count features using Standard scaling"""
        count_columns = [col for col in columns_to_scale if 'count' in col.lower() or 'total_services' in col]
        count_columns = [col for col in count_columns if col in df.columns]
        
        if count_columns:
            scaler = StandardScaler()
            df[count_columns] = scaler.fit_transform(df[count_columns])
            self.scalers['count_scaler'] = scaler
            logging.info(f"Standard scaled count features: {count_columns}")
        
        return df
    
    def _scale_other_features(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Scale other numerical features using Standard scaling"""
        # Get remaining numerical columns
        scaled_columns = []
        for scaler_info in self.scalers.values():
            if hasattr(scaler_info, 'feature_names_in_'):
                scaled_columns.extend(scaler_info.feature_names_in_)
        
        other_columns = [col for col in columns_to_scale if col not in scaled_columns and col in df.columns]
        other_columns = [col for col in other_columns if df[col].dtype in ['int64', 'float64']]
        
        if other_columns:
            scaler = StandardScaler()
            df[other_columns] = scaler.fit_transform(df[other_columns])
            self.scalers['other_scaler'] = scaler
            logging.info(f"Standard scaled other features: {other_columns}")
        
        return df
    
    def save_scaler(self, path: str = None) -> None:
        """Save all fitted scalers"""
        save_path = path or self.artifacts_path
        
        for scaler_name, scaler in self.scalers.items():
            scaler_file = os.path.join(save_path, f"telco_{scaler_name}.pkl")
            joblib.dump(scaler, scaler_file)
            logging.info(f"Saved {scaler_name} to {scaler_file}")

class FeatureScaler:
    """Main feature scaler class that coordinates different scaling strategies"""
    
    def __init__(self, artifacts_path: str = "artifacts/scalers"):
        """
        Initialize feature scaler
        
        Args:
            artifacts_path (str): Path to save scaler artifacts
        """
        self.artifacts_path = artifacts_path
        self.applied_strategies = []
        os.makedirs(artifacts_path, exist_ok=True)
        logging.info("Initialized FeatureScaler")
    
    def apply_scaling_strategy(
        self, 
        df: pd.DataFrame, 
        strategy: FeatureScalingStrategy,
        columns_to_scale: List[str]
    ) -> pd.DataFrame:
        """
        Apply a scaling strategy to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (FeatureScalingStrategy): Scaling strategy to apply
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: Scaled dataframe
        """
        try:
            logging.info(f"Applying {type(strategy).__name__}")
            df_scaled = strategy.scale(df, columns_to_scale)
            strategy.save_scaler(self.artifacts_path)
            self.applied_strategies.append(strategy)
            return df_scaled
            
        except Exception as e:
            logging.error(f"Error applying scaling strategy: {str(e)}")
            raise

def scale_numerical_features(
    df: pd.DataFrame,
    scaling_type: str = 'standard',
    columns_to_scale: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    artifacts_path: str = "artifacts/scalers",
    **kwargs
) -> pd.DataFrame:
    """
    Main function to scale numerical features in Telco dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        scaling_type (str): Type of scaling ('standard', 'minmax', 'robust', 'power', 'telco_specific')
        columns_to_scale (Optional[List[str]]): Columns to scale (auto-detected if None)
        output_path (Optional[str]): Path to save processed data
        artifacts_path (str): Path to save scaler artifacts
        **kwargs: Additional arguments for specific scalers
        
    Returns:
        pd.DataFrame: Dataframe with scaled features
    """
    try:
        logging.info(f"Scaling numerical features using {scaling_type} strategy")
        
        # Auto-detect numerical columns if not specified
        if columns_to_scale is None:
            columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID-like and already scaled columns
            columns_to_scale = [col for col in columns_to_scale if 
                               'id' not in col.lower() and 
                               not col.endswith('_encoded') and
                               not col.endswith('_binned')]
        
        logging.info(f"Columns to scale: {columns_to_scale}")
        
        # Initialize scaler
        scaler = FeatureScaler(artifacts_path)
        
        if scaling_type == 'standard':
            strategy = StandardScalingStrategy(artifacts_path)
            df_scaled = scaler.apply_scaling_strategy(df, strategy, columns_to_scale)
            
        elif scaling_type == 'minmax':
            feature_range = kwargs.get('feature_range', (0, 1))
            strategy = MinMaxScalingStrategy(feature_range, artifacts_path)
            df_scaled = scaler.apply_scaling_strategy(df, strategy, columns_to_scale)
            
        elif scaling_type == 'robust':
            strategy = RobustScalingStrategy(artifacts_path)
            df_scaled = scaler.apply_scaling_strategy(df, strategy, columns_to_scale)
            
        elif scaling_type == 'power':
            method = kwargs.get('method', 'yeo-johnson')
            strategy = PowerTransformScalingStrategy(method, artifacts_path)
            df_scaled = scaler.apply_scaling_strategy(df, strategy, columns_to_scale)
            
        elif scaling_type == 'telco_specific':
            strategy = TelcoSpecificScalingStrategy(artifacts_path)
            df_scaled = strategy.scale(df, columns_to_scale)
            strategy.save_scaler(artifacts_path)
            
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
        
        # Log scaling results
        logging.info(f"Feature scaling completed:")
        logging.info(f"  - Scaling type: {scaling_type}")
        logging.info(f"  - Columns scaled: {len(columns_to_scale)}")
        logging.info(f"  - Dataset shape: {df_scaled.shape}")
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_scaled.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df_scaled
        
    except Exception as e:
        logging.error(f"Feature scaling failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from previous steps)
        input_file = "data/processed/telco_encoded.csv"
        output_file = "data/processed/telco_scaled.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Apply standard scaling to numerical features
            df_processed = scale_numerical_features(
                df=df,
                scaling_type='standard',
                output_path=output_file,
                artifacts_path="artifacts/scalers"
            )
            
            print(f"Feature scaling completed successfully!")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"Numerical columns scaled:")
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_cols:
                print(f"  - {col}: mean={df_processed[col].mean():.4f}, std={df_processed[col].std():.4f}")
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run previous data processing steps first")
            
    except Exception as e:
        print(f"Feature scaling failed: {e}")
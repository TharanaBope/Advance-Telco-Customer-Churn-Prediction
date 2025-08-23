"""
Telco Customer Churn Feature Binning Module

This module provides strategies for binning numerical features into categorical groups,
specifically for tenure categorization and service-related groupings in the Telco dataset.

Based on Week 05_06 feature binning pattern with Telco-specific adaptations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureBinningStrategy(ABC):
    """Abstract base class for feature binning strategies"""
    
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Abstract method to bin a feature"""
        pass

class TenureBinningStrategy(FeatureBinningStrategy):
    """Strategy for binning tenure into meaningful customer lifecycle categories"""
    
    def __init__(self, keep_original: bool = True):
        """
        Initialize tenure binning strategy
        
        Args:
            keep_original (bool): Whether to keep the original column
        """
        self.keep_original = keep_original
        logging.info("Initialized TenureBinningStrategy")
    
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Bin tenure into customer lifecycle categories
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to bin (should be 'tenure')
            
        Returns:
            pd.DataFrame: Dataframe with tenure binned into categories
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in dataframe")
            
            df_binned = df.copy()
            
            # Define tenure categories based on telecom business logic
            def categorize_tenure(tenure_months):
                if pd.isna(tenure_months):
                    return 'Unknown'
                elif tenure_months == 0:
                    return 'New'  # New customers (0 months)
                elif 1 <= tenure_months <= 12:
                    return 'Short'  # Short-term customers (1-12 months)
                elif 13 <= tenure_months <= 24:
                    return 'Medium'  # Medium-term customers (13-24 months)
                elif 25 <= tenure_months <= 48:
                    return 'Long'  # Long-term customers (25-48 months)
                elif tenure_months > 48:
                    return 'Loyal'  # Loyal customers (>48 months)
                else:
                    return 'Unknown'
            
            # Create binned column
            df_binned['tenure_category'] = df_binned[column].apply(categorize_tenure)
            
            # Create numerical groups for additional analysis
            df_binned['tenure_group'] = pd.cut(
                df_binned[column],
                bins=[-1, 0, 12, 24, 48, float('inf')],
                labels=['New', 'Short', 'Medium', 'Long', 'Loyal']
            )
            
            # Log binning results
            category_counts = df_binned['tenure_category'].value_counts()
            logging.info(f"Tenure binning completed:")
            for category, count in category_counts.items():
                percentage = (count / len(df_binned)) * 100
                logging.info(f"  - {category}: {count} ({percentage:.1f}%)")
            
            # Remove original column if requested
            if not self.keep_original:
                df_binned = df_binned.drop(columns=[column])
                logging.info(f"Removed original {column} column")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in tenure binning: {str(e)}")
            raise

class MonthlyChargesBinningStrategy(FeatureBinningStrategy):
    """Strategy for binning MonthlyCharges into service tiers"""
    
    def __init__(self, keep_original: bool = True):
        """
        Initialize monthly charges binning strategy
        
        Args:
            keep_original (bool): Whether to keep the original column
        """
        self.keep_original = keep_original
        logging.info("Initialized MonthlyChargesBinningStrategy")
    
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Bin MonthlyCharges into service tier categories
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to bin (should be 'MonthlyCharges')
            
        Returns:
            pd.DataFrame: Dataframe with monthly charges binned into tiers
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in dataframe")
            
            df_binned = df.copy()
            
            # Define charge tiers based on telecom business logic
            def categorize_charges(charges):
                if pd.isna(charges):
                    return 'Unknown'
                elif charges < 30:
                    return 'Basic'  # Basic service tier
                elif 30 <= charges < 65:
                    return 'Standard'  # Standard service tier
                elif 65 <= charges < 95:
                    return 'Premium'  # Premium service tier
                elif charges >= 95:
                    return 'Enterprise'  # Enterprise service tier
                else:
                    return 'Unknown'
            
            # Create binned column
            df_binned[f'{column}_tier'] = df_binned[column].apply(categorize_charges)
            
            # Create quartile-based binning for additional insights
            df_binned[f'{column}_quartile'] = pd.qcut(
                df_binned[column],
                q=4,
                labels=['Q1_Low', 'Q2_Medium_Low', 'Q3_Medium_High', 'Q4_High']
            )
            
            # Log binning results
            tier_counts = df_binned[f'{column}_tier'].value_counts()
            logging.info(f"MonthlyCharges binning completed:")
            for tier, count in tier_counts.items():
                percentage = (count / len(df_binned)) * 100
                logging.info(f"  - {tier}: {count} ({percentage:.1f}%)")
            
            # Remove original column if requested
            if not self.keep_original:
                df_binned = df_binned.drop(columns=[column])
                logging.info(f"Removed original {column} column")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in monthly charges binning: {str(e)}")
            raise

class TotalChargesBinningStrategy(FeatureBinningStrategy):
    """Strategy for binning TotalCharges into customer value segments"""
    
    def __init__(self, keep_original: bool = True):
        """
        Initialize total charges binning strategy
        
        Args:
            keep_original (bool): Whether to keep the original column
        """
        self.keep_original = keep_original
        logging.info("Initialized TotalChargesBinningStrategy")
    
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Bin TotalCharges into customer value segments
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to bin (should be 'TotalCharges')
            
        Returns:
            pd.DataFrame: Dataframe with total charges binned into value segments
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in dataframe")
            
            df_binned = df.copy()
            
            # Define value segments based on telecom business logic
            def categorize_total_charges(total_charges):
                if pd.isna(total_charges):
                    return 'Unknown'
                elif total_charges == 0:
                    return 'New_Customer'  # New customers with no charges
                elif 0 < total_charges < 500:
                    return 'Low_Value'  # Low-value customers
                elif 500 <= total_charges < 2000:
                    return 'Medium_Value'  # Medium-value customers
                elif 2000 <= total_charges < 5000:
                    return 'High_Value'  # High-value customers
                elif total_charges >= 5000:
                    return 'Premium_Value'  # Premium-value customers
                else:
                    return 'Unknown'
            
            # Create binned column
            df_binned['customer_value_segment'] = df_binned[column].apply(categorize_total_charges)
            
            # Create decile-based binning for detailed analysis
            df_binned[f'{column}_decile'] = pd.qcut(
                df_binned[column],
                q=10,
                labels=[f'D{i+1}' for i in range(10)]
            )
            
            # Log binning results
            segment_counts = df_binned['customer_value_segment'].value_counts()
            logging.info(f"TotalCharges binning completed:")
            for segment, count in segment_counts.items():
                percentage = (count / len(df_binned)) * 100
                logging.info(f"  - {segment}: {count} ({percentage:.1f}%)")
            
            # Remove original column if requested
            if not self.keep_original:
                df_binned = df_binned.drop(columns=[column])
                logging.info(f"Removed original {column} column")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in total charges binning: {str(e)}")
            raise

class CustomBinningStrategy(FeatureBinningStrategy):
    """Custom binning strategy with user-defined bins"""
    
    def __init__(self, bin_definitions: Dict[str, Union[List, Tuple]], keep_original: bool = True):
        """
        Initialize custom binning strategy
        
        Args:
            bin_definitions (Dict): Dictionary defining bins {label: [min, max]} or {label: [threshold]}
            keep_original (bool): Whether to keep the original column
        """
        self.bin_definitions = bin_definitions
        self.keep_original = keep_original
        logging.info(f"Initialized CustomBinningStrategy with {len(bin_definitions)} bins")
    
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Bin feature based on custom definitions
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to bin
            
        Returns:
            pd.DataFrame: Dataframe with custom binned feature
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in dataframe")
            
            df_binned = df.copy()
            
            def assign_bin(value):
                if pd.isna(value):
                    return 'Unknown'
                
                # Assign bins based on custom definitions
                for bin_label, bin_range in self.bin_definitions.items():
                    if len(bin_range) == 2:  # Range [min, max]
                        if bin_range[0] <= value <= bin_range[1]:
                            return bin_label
                    elif len(bin_range) == 1:  # Threshold [min]
                        if value >= bin_range[0]:
                            return bin_label
                
                return 'Other'
            
            # Create binned column
            df_binned[f'{column}_bins'] = df_binned[column].apply(assign_bin)
            
            # Log binning results
            bin_counts = df_binned[f'{column}_bins'].value_counts()
            logging.info(f"Custom binning for {column} completed:")
            for bin_label, count in bin_counts.items():
                percentage = (count / len(df_binned)) * 100
                logging.info(f"  - {bin_label}: {count} ({percentage:.1f}%)")
            
            # Remove original column if requested
            if not self.keep_original:
                df_binned = df_binned.drop(columns=[column])
                logging.info(f"Removed original {column} column")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in custom binning: {str(e)}")
            raise

class ServiceCountBinningStrategy(FeatureBinningStrategy):
    """Strategy for creating service adoption count features"""
    
    def __init__(self):
        """Initialize service count binning strategy"""
        logging.info("Initialized ServiceCountBinningStrategy")
    
    def bin_feature(self, df: pd.DataFrame, column: str = None) -> pd.DataFrame:
        """
        Create service count and adoption features
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Not used for this strategy
            
        Returns:
            pd.DataFrame: Dataframe with service count features
        """
        try:
            df_binned = df.copy()
            
            # Define service columns
            service_columns = [
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            
            # Filter for columns that exist in the dataframe
            available_services = [col for col in service_columns if col in df_binned.columns]
            
            if not available_services:
                logging.warning("No service columns found for service count binning")
                return df_binned
            
            # Count active services (assuming 'Yes' means active)
            def count_active_services(row):
                count = 0
                for service in available_services:
                    if row[service] == 'Yes':
                        count += 1
                    elif service == 'InternetService' and row[service] not in ['No', 'No internet service']:
                        count += 1  # DSL or Fiber Optic counts as a service
                return count
            
            # Create service count feature
            df_binned['total_services'] = df_binned.apply(count_active_services, axis=1)
            
            # Create service adoption categories
            def categorize_service_adoption(service_count):
                if service_count == 0:
                    return 'No_Services'
                elif service_count <= 2:
                    return 'Low_Adoption'
                elif service_count <= 5:
                    return 'Medium_Adoption'
                elif service_count <= 7:
                    return 'High_Adoption'
                else:
                    return 'Full_Adoption'
            
            df_binned['service_adoption_level'] = df_binned['total_services'].apply(categorize_service_adoption)
            
            # Create internet-based services count
            internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                               'TechSupport', 'StreamingTV', 'StreamingMovies']
            available_internet_services = [col for col in internet_services if col in df_binned.columns]
            
            def count_internet_services(row):
                if 'InternetService' in df_binned.columns and row['InternetService'] == 'No':
                    return 0
                count = 0
                for service in available_internet_services:
                    if row[service] == 'Yes':
                        count += 1
                return count
            
            df_binned['internet_services_count'] = df_binned.apply(count_internet_services, axis=1)
            
            # Log binning results
            adoption_counts = df_binned['service_adoption_level'].value_counts()
            logging.info(f"Service adoption binning completed:")
            for level, count in adoption_counts.items():
                percentage = (count / len(df_binned)) * 100
                logging.info(f"  - {level}: {count} ({percentage:.1f}%)")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in service count binning: {str(e)}")
            raise

class FeatureBinner:
    """Main feature binner class that applies various binning strategies"""
    
    def __init__(self):
        """Initialize feature binner"""
        logging.info("Initialized FeatureBinner")
    
    def apply_telco_binning(
        self, 
        df: pd.DataFrame, 
        strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply Telco-specific binning strategies
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategies (Optional[List[str]]): List of strategies to apply
            
        Returns:
            pd.DataFrame: Dataframe with binned features
        """
        try:
            if strategies is None:
                strategies = ['tenure', 'monthly_charges', 'total_charges', 'service_count']
            
            df_binned = df.copy()
            
            for strategy in strategies:
                if strategy == 'tenure' and 'tenure' in df_binned.columns:
                    binner = TenureBinningStrategy(keep_original=True)
                    df_binned = binner.bin_feature(df_binned, 'tenure')
                
                elif strategy == 'monthly_charges' and 'MonthlyCharges' in df_binned.columns:
                    binner = MonthlyChargesBinningStrategy(keep_original=True)
                    df_binned = binner.bin_feature(df_binned, 'MonthlyCharges')
                
                elif strategy == 'total_charges' and 'TotalCharges' in df_binned.columns:
                    binner = TotalChargesBinningStrategy(keep_original=True)
                    df_binned = binner.bin_feature(df_binned, 'TotalCharges')
                
                elif strategy == 'service_count':
                    binner = ServiceCountBinningStrategy()
                    df_binned = binner.bin_feature(df_binned)
                
                else:
                    logging.warning(f"Strategy {strategy} not recognized or required columns not found")
            
            return df_binned
            
        except Exception as e:
            logging.error(f"Error in Telco binning application: {str(e)}")
            raise

def apply_feature_binning(
    df: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    custom_bins: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main function to apply feature binning to Telco dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategies (Optional[List[str]]): List of binning strategies to apply
        custom_bins (Optional[Dict]): Custom binning definitions
        output_path (Optional[str]): Path to save processed data
        
    Returns:
        pd.DataFrame: Dataframe with binned features
    """
    try:
        logging.info("Applying feature binning to Telco dataset")
        
        # Initialize feature binner
        binner = FeatureBinner()
        
        # Apply Telco-specific binning
        df_binned = binner.apply_telco_binning(df, strategies)
        
        # Apply custom binning if provided
        if custom_bins:
            for column, bin_definitions in custom_bins.items():
                if column in df_binned.columns:
                    custom_binner = CustomBinningStrategy(bin_definitions, keep_original=True)
                    df_binned = custom_binner.bin_feature(df_binned, column)
        
        # Log final results
        new_columns = [col for col in df_binned.columns if col not in df.columns]
        logging.info(f"Feature binning completed. New columns created: {new_columns}")
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_binned.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df_binned
        
    except Exception as e:
        logging.error(f"Feature binning failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load data (assuming it exists from previous steps)
        input_file = "data/processed/telco_outliers_removed.csv"
        output_file = "data/processed/telco_binned.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Apply feature binning
            df_processed = apply_feature_binning(
                df=df,
                strategies=['tenure', 'monthly_charges', 'total_charges', 'service_count'],
                output_path=output_file
            )
            
            print(f"Feature binning completed successfully!")
            print(f"Dataset shape: {df_processed.shape}")
            print(f"New categorical features:")
            new_columns = [col for col in df_processed.columns if col not in df.columns]
            for col in new_columns:
                print(f"  - {col}: {df_processed[col].value_counts().to_dict()}")
            
        else:
            print(f"Input file not found: {input_file}")
            print("Please run previous data processing steps first")
            
    except Exception as e:
        print(f"Feature binning failed: {e}")
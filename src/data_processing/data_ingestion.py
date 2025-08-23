"""
Telco Customer Churn Data Ingestion Module

This module handles loading and initial processing of the Telco Customer Churn dataset,
specifically handling the TotalCharges data type conversion issue.

Based on Week 05_06 data ingestion pattern with Telco-specific adaptations.
"""

import os
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestor(ABC):
    """Abstract base class for data ingestion strategies"""
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """Abstract method to ingest data from various sources"""
        pass

class TelcoCSVIngestor(DataIngestor):
    """
    Specialized CSV ingestor for Telco Customer Churn dataset
    Handles specific data type issues and initial cleaning
    """
    
    def __init__(self, handle_totalcharges: bool = True):
        """
        Initialize Telco CSV ingestor
        
        Args:
            handle_totalcharges (bool): Whether to handle TotalCharges data type conversion
        """
        self.handle_totalcharges = handle_totalcharges
        logging.info("Initialized TelcoCSVIngestor")
    
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Ingest Telco CSV data with specific handling for data type issues
        
        Args:
            file_path_or_link (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and initially processed dataframe
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path_or_link):
                raise FileNotFoundError(f"File not found: {file_path_or_link}")
            
            logging.info(f"Loading Telco data from: {file_path_or_link}")
            
            # Load CSV data
            df = pd.read_csv(file_path_or_link)
            initial_shape = df.shape
            logging.info(f"Loaded data with shape: {initial_shape}")
            
            # Handle TotalCharges data type conversion (specific to Telco dataset)
            if self.handle_totalcharges and 'TotalCharges' in df.columns:
                df = self._handle_totalcharges_conversion(df)
            
            # Basic data validation
            self._validate_telco_data(df)
            
            logging.info(f"Successfully ingested Telco data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise
    
    def _handle_totalcharges_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle TotalCharges column conversion from object to numeric
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with TotalCharges converted to numeric
        """
        try:
            logging.info("Handling TotalCharges data type conversion")
            
            # Check current data type
            original_dtype = df['TotalCharges'].dtype
            logging.info(f"TotalCharges original dtype: {original_dtype}")
            
            # Count non-numeric values
            non_numeric_mask = pd.to_numeric(df['TotalCharges'], errors='coerce').isna()
            non_numeric_count = non_numeric_mask.sum()
            
            if non_numeric_count > 0:
                logging.warning(f"Found {non_numeric_count} non-numeric values in TotalCharges")
                
                # Log some examples of non-numeric values
                non_numeric_values = df.loc[non_numeric_mask, 'TotalCharges'].unique()
                logging.info(f"Non-numeric TotalCharges values: {non_numeric_values[:10]}")
            
            # Convert to numeric (will convert non-numeric to NaN)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Log conversion results
            new_na_count = df['TotalCharges'].isna().sum()
            logging.info(f"TotalCharges converted to numeric. New NaN count: {new_na_count}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error in TotalCharges conversion: {str(e)}")
            raise
    
    def _validate_telco_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the loaded data has expected Telco dataset structure
        
        Args:
            df (pd.DataFrame): Dataframe to validate
        """
        try:
            # Expected key columns for Telco dataset
            expected_columns = [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ]
            
            # Check for missing critical columns
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logging.warning(f"Missing expected columns: {missing_columns}")
            
            # Check for target variable
            if 'Churn' not in df.columns:
                raise ValueError("Target variable 'Churn' not found in dataset")
            
            # Validate target variable values
            churn_values = df['Churn'].unique()
            expected_churn_values = ['Yes', 'No']
            if not all(val in expected_churn_values for val in churn_values if pd.notna(val)):
                logging.warning(f"Unexpected Churn values: {churn_values}")
            
            # Log dataset summary
            logging.info(f"Dataset validation completed:")
            logging.info(f"  - Columns: {len(df.columns)}")
            logging.info(f"  - Rows: {len(df)}")
            logging.info(f"  - Churn distribution: {df['Churn'].value_counts().to_dict()}")
            
        except Exception as e:
            logging.error(f"Data validation error: {str(e)}")
            raise

class ExcelIngestor(DataIngestor):
    """Excel file ingestor for additional data sources"""
    
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Ingest data from Excel files
        
        Args:
            file_path_or_link (str): Path to Excel file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if not os.path.exists(file_path_or_link):
                raise FileNotFoundError(f"File not found: {file_path_or_link}")
            
            logging.info(f"Loading Excel data from: {file_path_or_link}")
            df = pd.read_excel(file_path_or_link)
            logging.info(f"Successfully loaded Excel data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading Excel file: {str(e)}")
            raise

class DataIngestionFactory:
    """Factory class to create appropriate data ingestors"""
    
    @staticmethod
    def create_ingestor(file_path: str, **kwargs) -> DataIngestor:
        """
        Create appropriate data ingestor based on file extension
        
        Args:
            file_path (str): Path to the data file
            **kwargs: Additional arguments for specific ingestors
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return TelcoCSVIngestor(**kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return ExcelIngestor()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

def ingest_telco_data(
    file_path: str, 
    output_path: Optional[str] = None,
    handle_totalcharges: bool = True
) -> pd.DataFrame:
    """
    Main function to ingest Telco customer churn data
    
    Args:
        file_path (str): Path to the input data file
        output_path (Optional[str]): Path to save processed data
        handle_totalcharges (bool): Whether to handle TotalCharges conversion
        
    Returns:
        pd.DataFrame: Loaded and initially processed data
    """
    try:
        # Create appropriate ingestor
        ingestor = DataIngestionFactory.create_ingestor(
            file_path, 
            handle_totalcharges=handle_totalcharges
        )
        
        # Ingest data
        df = ingestor.ingest(file_path)
        
        # Save processed data if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        logging.error(f"Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_file = "data/processed/telco_ingested.csv"
    
    try:
        df = ingest_telco_data(
            file_path=input_file,
            output_path=output_file,
            handle_totalcharges=True
        )
        
        print(f"Data ingestion completed successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Data ingestion failed: {e}")
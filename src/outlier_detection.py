import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutlierDetector:
    def __init__(self, mape_threshold: float = 100, window_size: int = 3):
        """
        Initialize outlier detector

        Args:
            mape_threshold (float): Threshold for identifying high MAPE values
            window_size (int): Window size for rolling average smoothing
        """
        self.mape_threshold = mape_threshold
        self.window_size = window_size

    def detect_statistical_outliers(self, data: pd.Series, z_threshold: float = 3) -> pd.Series:
        """
        Detect outliers using z-score method

        Args:
            data (pd.Series): Time series data
            z_threshold (float): Z-score threshold for outlier detection

        Returns:
            pd.Series: Boolean mask of outliers
        """
        z_scores = np.abs(stats.zscore(data))
        return pd.Series(z_scores > z_threshold, index=data.index)

    def detect_iqr_outliers(self, data: pd.Series) -> pd.Series:
        """
        Detect outliers using IQR method

        Args:
            data (pd.Series): Time series data

        Returns:
            pd.Series: Boolean mask of outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (data < lower_bound) | (data > upper_bound)

    def smooth_data(self, data: pd.Series) -> pd.Series:
        """
        Apply rolling average smoothing to the data

        Args:
            data (pd.Series): Time series data

        Returns:
            pd.Series: Smoothed data
        """
        return data.rolling(window=self.window_size, min_periods=1, center=True).mean()

    def handle_outliers(self, data: pd.Series, method: str = 'smooth') -> pd.Series:
        """
        Handle outliers using specified method

        Args:
            data (pd.Series): Time series data
            method (str): Method to handle outliers ('smooth' or 'interpolate')

        Returns:
            pd.Series: Data with handled outliers
        """
        # Detect outliers using both methods
        statistical_outliers = self.detect_statistical_outliers(data)
        iqr_outliers = self.detect_iqr_outliers(data)
        
        # Combine outlier detection methods
        combined_outliers = statistical_outliers | iqr_outliers
        
        if method == 'smooth':
            # Apply smoothing only to outlier points
            smoothed_data = self.smooth_data(data)
            data[combined_outliers] = smoothed_data[combined_outliers]
        elif method == 'interpolate':
            # Interpolate outlier points
            data[combined_outliers] = np.nan
            data = data.interpolate(method='linear')
        
        return data

    def process_store_data(self, profits_data: pd.DataFrame, 
                          best_model_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean data for all stores

        Args:
            profits_data (pd.DataFrame): Original profits data
            best_model_data (pd.DataFrame): Best model information for each store

        Returns:
            pd.DataFrame: Processed data with handled outliers
        """
        processed_data = profits_data.copy()
        
        # Identify stores with high MAPE
        high_mape_stores = best_model_data[
            best_model_data['MAPE (%)'] > self.mape_threshold
        ]['Store Name'].unique()
        
        # Process each store with high MAPE
        for store_name in high_mape_stores:
            try:
                # Get store data excluding 'Store Name' column
                store_data = processed_data[
                    processed_data['Store Name'] == store_name
                ].iloc[0, 1:]
                
                # Convert to time series
                store_series = pd.Series(
                    store_data.values,
                    index=pd.date_range(
                        start='2023-01-01',
                        periods=len(store_data),
                        freq='M'
                    )
                )
                
                # Handle outliers
                cleaned_series = self.handle_outliers(store_series)
                
                # Update the processed data
                processed_data.loc[
                    processed_data['Store Name'] == store_name,
                    processed_data.columns[1:]
                ] = cleaned_series.values
                
                logger.info(f"Processed outliers for store: {store_name}")
                
            except Exception as e:
                logger.error(f"Error processing store {store_name}: {str(e)}")
        
        return processed_data

    def save_processed_data(self, data: pd.DataFrame, output_path: str):
        """
        Save processed data to Excel file

        Args:
            data (pd.DataFrame): Processed data
            output_path (str): Path to save the output file
        """
        data.to_excel(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

def main():
    # Load data
    profits_data = pd.read_excel('data/processed/Cleaned_Combined_Profit_Data_2023_2024.xlsx')
    best_model_data = pd.read_csv('results/store_forecasting_best_models.csv')
    
    # Initialize outlier detector
    detector = OutlierDetector()
    
    # Process data
    processed_data = detector.process_store_data(profits_data, best_model_data)
    
    # Save processed data
    detector.save_processed_data(
        processed_data,
        'data/processed/Smoothed_Combined_Profit_Data_2023_2024.xlsx'
    )

    logger.info("Outlier detection and handling completed successfully")

if __name__ == '__main__':
    main()
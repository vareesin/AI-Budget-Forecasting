import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, data_dir: str):
        """
        Initialize data preparation class
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_2023_data(self, file_path: str) -> pd.DataFrame:
        """Load and process 2023 profit data"""
        months_2023 = [
            'Jan23', 'Feb23', 'mar23', 'Apr23', 'may23', 'Jun23',
            'Jul23', 'Aug23', 'Sep23', 'Oct23', 'Nov23', 'Dec23'
        ]
        
        df_list = []
        for month in months_2023:
            try:
                df = pd.read_excel(file_path, sheet_name=month)
                df = df[['Store Name', 'กำไรหลังAudit/Wo']].rename(
                    columns={'กำไรหลังAudit/Wo': month}
                )
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error processing month {month}: {str(e)}")
                
        df_2023 = pd.concat(df_list, axis=1)
        return df_2023.loc[:, ~df_2023.columns.duplicated()]

    def load_2024_data(self, file_path: str) -> pd.DataFrame:
        """Load and process 2024 profit data"""
        # Process Jan-Feb
        jan_feb_months = ['มค', 'กพ']
        df_2024_jan_feb = pd.concat([
            pd.read_excel(file_path, sheet_name=month)[['Store Name', 'กำไรหลังAudit/Wo']]
            .rename(columns={'กำไรหลังAudit/Wo': month})
            for month in jan_feb_months
        ], axis=1)
        
        # Process Mar-Jun with different headers
        df_2024_mar = pd.read_excel(file_path, sheet_name='มีค', header=1)[
            ['Store Name', 'กำไรหลังAudit/Wo']
        ].rename(columns={'กำไรหลังAudit/Wo': 'มีค'})
        
        df_2024_apr_jun = pd.concat([
            pd.read_excel(
                file_path, 
                sheet_name=sheet, 
                header=header
            )[['Store Name', 'กำไรระดับสาขารวมโบนัส+ปันส่วนจาก 7-11']]
            .rename(columns={'กำไรระดับสาขารวมโบนัส+ปันส่วนจาก 7-11': month})
            for sheet, header, month in [
                ('เมย', 1, 'เมย'),
                ('พค', 1, 'พค'),
                ('มิย', 2, 'มิย')
            ]
        ], axis=1)
        
        # Combine all 2024 data
        dfs = [df_2024_jan_feb, df_2024_mar, df_2024_apr_jun]
        df_2024 = dfs[0]
        for df in dfs[1:]:
            df_2024 = pd.merge(df_2024, df, on='Store Name', how='outer')
            
        return df_2024.loc[:, ~df_2024.columns.duplicated()]

    def combine_and_clean_data(self) -> pd.DataFrame:
        """Combine and clean 2023 and 2024 data"""
        # Load data
        df_2023 = self.load_2023_data(os.path.join(self.raw_dir, 'Profits_2023.xlsx'))
        df_2024 = self.load_2024_data(os.path.join(self.raw_dir, 'Profits_2024.xlsx'))
        
        # Combine data
        combined = pd.merge(df_2023, df_2024, on='Store Name', how='outer')
        
        # Clean duplicates
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        # Rename columns to standardized format
        month_mapping = {
            'Jan23': 'Jan2023', 'Feb23': 'Feb2023', 'mar23': 'Mar2023',
            'Apr23': 'Apr2023', 'may23': 'May2023', 'Jun23': 'Jun2023',
            'Jul23': 'Jul2023', 'Aug23': 'Aug2023', 'Sep23': 'Sep2023',
            'Oct23': 'Oct2023', 'Nov23': 'Nov2023', 'Dec23': 'Dec2023',
            'มค': 'Jan2024', 'กพ': 'Feb2024', 'มีค': 'Mar2024',
            'เมย': 'Apr2024', 'พค': 'May2024', 'มิย': 'Jun2024'
        }
        combined = combined.rename(columns=month_mapping)
        
        # Remove rows with missing store names
        combined = combined.dropna(subset=['Store Name']).reset_index(drop=True)
        
        return combined

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to Excel file"""
        output_path = os.path.join(self.processed_dir, filename)
        df.to_excel(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

def main():
    # Initialize data preparation
    data_prep = DataPreparation('data')
    
    # Process and combine data
    combined_data = data_prep.combine_and_clean_data()
    
    # Save processed data
    data_prep.save_processed_data(
        combined_data, 
        'Cleaned_Combined_Profit_Data_2023_2024.xlsx'
    )

if __name__ == '__main__':
    main()
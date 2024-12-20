import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset(dataset: np.ndarray, look_back: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time series dataset with look_back window size
    
    Args:
        dataset (np.ndarray): Input time series data
        look_back (int): Number of previous time steps to use as input variables
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and Y (target) arrays
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics for predictions
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    metrics = {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = "Actual vs Predicted Values",
                    store_name: str = None) -> None:
    """
    Plot actual vs predicted values
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        store_name (str): Store name for labeling
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='s')
    plt.title(f"{title} - {store_name}" if store_name else title)
    plt.xlabel('Time Period')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_error_distribution(errors: np.ndarray, 
                          title: str = "Prediction Error Distribution") -> None:
    """
    Plot distribution of prediction errors
    
    Args:
        errors (np.ndarray): Array of prediction errors
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(title)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

def calculate_seasonal_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate seasonal metrics for each store
    
    Args:
        data (pd.DataFrame): Input data with monthly profits
    
    Returns:
        pd.DataFrame: Seasonal metrics for each store
    """
    metrics = []
    
    for _, row in data.iterrows():
        store_name = row['Store Name']
        monthly_data = row.iloc[1:].values  # Exclude store name
        
        # Calculate seasonal metrics
        seasonal_data = {
            'Store Name': store_name,
            'Average Monthly Profit': np.mean(monthly_data),
            'Profit Volatility': np.std(monthly_data),
            'Peak Month': np.argmax(monthly_data) + 1,
            'Lowest Month': np.argmin(monthly_data) + 1,
            'Trend': np.polyfit(np.arange(len(monthly_data)), monthly_data, 1)[0]
        }
        
        metrics.append(seasonal_data)
    
    return pd.DataFrame(metrics)

def save_plot(plt: plt, filename: str, results_dir: str = 'results/visualizations'):
    """
    Save plot to file with timestamp
    
    Args:
        plt: Matplotlib plot object
        filename (str): Base filename
        results_dir (str): Directory to save results
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filename = f"{filename}_{timestamp}.png"
    full_path = os.path.join(results_dir, full_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {full_path}")

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Load data and perform basic validations
    
    Args:
        file_path (str): Path to data file
    
    Returns:
        pd.DataFrame: Validated DataFrame
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Basic validations
        assert 'Store Name' in df.columns, "Missing 'Store Name' column"
        assert not df['Store Name'].isna().any(), "Found missing store names"
        assert df.shape[1] > 1, "No profit data columns found"
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
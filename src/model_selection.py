import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
import logging
from typing import Tuple, Dict, List
import os

from utils import create_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSelection:
    def __init__(self, look_back: int = 3):
        """Initialize model selection class"""
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_lstm(self) -> Sequential:
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        return model

    def build_gru(self) -> Sequential:
        """Build GRU model"""
        model = Sequential([
            GRU(100, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            GRU(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        return model

    def build_cnn_lstm(self) -> Sequential:
        """Build CNN-LSTM model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', 
                  input_shape=(self.look_back, 1)),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        return model

    def evaluate_model(self, model_name: str, model, X_train: np.ndarray, 
                      y_train: np.ndarray, X_test: np.ndarray, 
                      y_test: np.ndarray) -> float:
        """Evaluate model performance"""
        if model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
            model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)
            predictions = model.predict(X_test)
            predictions = self.scaler.inverse_transform(predictions)
        else:
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
            model.fit(X_train_2d, y_train)
            predictions = model.predict(X_test_2d).reshape(-1, 1)

        y_test_transformed = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        mape = mean_absolute_percentage_error(y_test_transformed, predictions)
        return mape * 100

    def find_best_model(self, store_data: pd.Series) -> Tuple[str, float]:
        """Find the best model for given store data"""
        # Prepare data
        scaled_data = self.scaler.fit_transform(store_data.values.reshape(-1, 1))
        train_size = int(len(scaled_data) * 0.8)
        
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        X_train, y_train = create_dataset(train_data, self.look_back)
        X_test, y_test = create_dataset(test_data, self.look_back)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Initialize models
        models = {
            'LSTM': self.build_lstm(),
            'GRU': self.build_gru(),
            'CNN-LSTM': self.build_cnn_lstm(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }

        # Evaluate each model
        model_scores = {}
        for name, model in models.items():
            try:
                mape = self.evaluate_model(name, model, X_train, y_train, 
                                         X_test, y_test)
                model_scores[name] = mape
                logger.info(f"Model {name} MAPE: {mape:.2f}%")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")

        # Find best model
        best_model = min(model_scores.items(), key=lambda x: x[1])
        return best_model[0], best_model[1]

def main():
    # Load data
    data_path = 'data/processed/Cleaned_Combined_Profit_Data_2023_2024.xlsx'
    data = pd.read_excel(data_path)
    
    # Initialize model selection
    model_selector = ModelSelection()
    
    # Store results
    results = []
    
    # Process each store
    for idx, row in data.iterrows():
        store_name = row['Store Name']
        store_data = row.iloc[1:]  # Exclude store name
        
        logger.info(f"Processing store: {store_name}")
        
        try:
            best_model, best_mape = model_selector.find_best_model(store_data)
            results.append({
                'Store Name': store_name,
                'Best Model': best_model,
                'MAPE (%)': best_mape
            })
        except Exception as e:
            logger.error(f"Error processing store {store_name}: {str(e)}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/store_forecasting_best_models.csv', index=False)
    logger.info("Completed model selection process")

if __name__ == '__main__':
    main()
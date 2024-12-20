import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

from utils import create_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, look_back: int = 3):
        """
        Initialize predictor class
        
        Args:
            look_back (int): Number of previous time steps to use for prediction
        """
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

    def prepare_data(self, store_data: pd.Series) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Prepare data for prediction
        
        Args:
            store_data (pd.Series): Time series data for a store
            
        Returns:
            Tuple[np.ndarray, pd.DatetimeIndex]: Scaled data and dates
        """
        dates = pd.date_range(start='2023-01-01', periods=len(store_data), freq='M')
        scaled_data = self.scaler.fit_transform(store_data.values.reshape(-1, 1))
        return scaled_data, dates

    def predict_next_periods(self, model_name: str, model: any, 
                           input_sequence: np.ndarray, periods: int = 6) -> np.ndarray:
        """
        Predict future periods using the specified model
        
        Args:
            model_name (str): Name of the model
            model: Trained model instance
            input_sequence (np.ndarray): Input sequence for prediction
            periods (int): Number of periods to predict
            
        Returns:
            np.ndarray: Predicted values
        """
        predictions = []
        current_sequence = input_sequence.copy()

        for _ in range(periods):
            if model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
                # Reshape for deep learning models
                current_sequence_3d = current_sequence.reshape(1, self.look_back, 1)
                next_pred = model.predict(current_sequence_3d, verbose=0)
            elif model_name in ['RandomForest', 'XGBoost']:
                # Reshape for traditional ML models
                current_sequence_2d = current_sequence.reshape(1, -1)
                next_pred = model.predict(current_sequence_2d).reshape(-1, 1)
            else:
                raise ValueError(f"Unknown model type: {model_name}")

            predictions.append(next_pred[0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0]

        return np.array(predictions)

    def predict_store(self, store_data: pd.Series, model_name: str, 
                     future_periods: int = 6) -> Dict[str, np.ndarray]:
        """
        Make predictions for a single store
        
        Args:
            store_data (pd.Series): Store's historical data
            model_name (str): Name of the model to use
            future_periods (int): Number of future periods to predict
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing predictions and dates
        """
        # Prepare data
        scaled_data, dates = self.prepare_data(store_data)
        
        # Create sequences
        X = scaled_data[-self.look_back:].reshape(1, self.look_back, 1)
        
        # Initialize appropriate model
        if model_name == 'LSTM':
            model = self.build_lstm()
        elif model_name == 'GRU':
            model = self.build_gru()
        elif model_name == 'CNN-LSTM':
            model = self.build_cnn_lstm()
        elif model_name == 'RandomForest':
            model = RandomForestRegressor()
        elif model_name == 'XGBoost':
            model = XG
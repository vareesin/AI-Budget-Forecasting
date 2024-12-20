import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_selection import ModelSelection
from src.prediction import Predictor
from src.data_preparation import DataPreparation
from src.outlier_detection import OutlierDetector
from src.utils import create_dataset, evaluate_predictions

class TestModelSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used by all test methods"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        cls.sample_data = pd.DataFrame({
            'Store Name': ['Test Store 1', 'Test Store 2'],
            'Jan2023': [1000, 2000],
            'Feb2023': [1100, 2100],
            'Mar2023': [1200, 2200],
            'Apr2023': [1300, 2300],
            'May2023': [1400, 2400],
            'Jun2023': [1500, 2500],
            'Jul2023': [1600, 2600],
            'Aug2023': [1700, 2700],
            'Sep2023': [1800, 2800],
            'Oct2023': [1900, 2900],
            'Nov2023': [2000, 3000],
            'Dec2023': [2100, 3100]
        })
        
        # Initialize model selection
        cls.model_selector = ModelSelection(look_back=3)

    def test_model_initialization(self):
        """Test if model selector initializes correctly"""
        self.assertEqual(self.model_selector.look_back, 3)
        self.assertIsNotNone(self.model_selector.scaler)

    def test_lstm_build(self):
        """Test LSTM model structure"""
        lstm_model = self.model_selector.build_lstm()
        self.assertEqual(len(lstm_model.layers), 4)  # Check number of layers
        self.assertEqual(lstm_model.input_shape, (None, 3, 1))  # Check input shape

    def test_find_best_model(self):
        """Test best model selection"""
        store_data = self.sample_data.iloc[0, 1:]  # Get first store's data
        best_model, mape = self.model_selector.find_best_model(store_data)
        
        # Check if results are valid
        self.assertIn(best_model, ['LSTM', 'GRU', 'CNN-LSTM', 'RandomForest', 'XGBoost', 'Prophet'])
        self.assertIsInstance(mape, float)
        self.assertGreater(mape, 0)

class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data for predictor"""
        cls.predictor = Predictor(look_back=3)
        
        # Create sample data
        cls.sample_series = pd.Series(np.linspace(1000, 2000, 12))

    def test_prepare_data(self):
        """Test data preparation"""
        scaled_data, dates = self.predictor.prepare_data(self.sample_series)
        
        # Check scaled data
        self.assertEqual(scaled_data.shape[0], len(self.sample_series))
        self.assertTrue(np.all(scaled_data >= 0) and np.all(scaled_data <= 1))
        
        # Check dates
        self.assertEqual(len(dates), len(self.sample_series))
        self.assertTrue(isinstance(dates[0], datetime))

    def test_predict_store(self):
        """Test store prediction"""
        result = self.predictor.predict_store(
            self.sample_series, 
            model_name='LSTM',
            future_periods=6
        )
        
        # Check prediction results
        self.assertIn('predictions', result)
        self.assertIn('dates', result)
        self.assertEqual(len(result['predictions']), 6)
        self.assertEqual(len(result['dates']), 6)

class TestOutlierDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data for outlier detection"""
        cls.detector = OutlierDetector(mape_threshold=100, window_size=3)
        
        # Create sample data with outliers
        cls.sample_data = pd.Series([100, 110, 120, 500, 130, 140, 150])

    def test_detect_statistical_outliers(self):
        """Test statistical outlier detection"""
        outliers = self.detector.detect_statistical_outliers(self.sample_data)
        self.assertTrue(outliers[3])  # Check if spike is detected
        self.assertEqual(sum(outliers), 1)  # Should only detect one outlier

    def test_smooth_data(self):
        """Test data smoothing"""
        smoothed = self.detector.smooth_data(self.sample_data)
        self.assertEqual(len(smoothed), len(self.sample_data))
        self.assertNotEqual(smoothed[3], self.sample_data[3])  # Check if outlier was smoothed

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.data_prep = DataPreparation('data')

    def test_combine_and_clean_data(self):
        """Test data combination and cleaning"""
        try:
            combined_data = self.data_prep.combine_and_clean_data()
            self.assertIsInstance(combined_data, pd.DataFrame)
            self.assertIn('Store Name', combined_data.columns)
            self.assertTrue(len(combined_data) > 0)
        except FileNotFoundError:
            self.skipTest("Raw data files not found")

class TestUtils(unittest.TestCase):
    def test_create_dataset(self):
        """Test dataset creation"""
        data = np.array([[1], [2], [3], [4], [5]])
        X, y = create_dataset(data, look_back=2)
        
        self.assertEqual(X.shape[0], 3)  # Number of samples
        self.assertEqual(X.shape[1], 2)  # Look back window size
        self.assertEqual(len(y), 3)      # Number of targets

    def test_evaluate_predictions(self):
        """Test prediction evaluation metrics"""
        y_true = np.array([100, 110, 120, 130])
        y_pred = np.array([105, 108, 125, 128])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        self.assertIn('MAPE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))

if __name__ == '__main__':
    unittest.main()
"""
Basic tests for the Saudi Real Estate Price Prediction system.
"""

import unittest
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the system."""
    
    def test_data_files_exist(self):
        """Test that required data files exist."""
        data_files = [
            'src/data/category_mapping.json',
            'src/data/city_mapping.json',
            'src/data/district_mapping.json'
        ]
        
        for file_path in data_files:
            with self.subTest(file_path=file_path):
                self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
    
    def test_model_files_exist(self):
        """Test that required model files exist."""
        model_files = [
            'src/models/price_prediction_model.keras',
            'src/models/preprocessor.pkl',
            'src/models/y_scaler.pkl'
        ]
        
        for file_path in model_files:
            with self.subTest(file_path=file_path):
                self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
    
    def test_json_files_are_valid(self):
        """Test that JSON files are valid."""
        json_files = [
            'src/data/category_mapping.json',
            'src/data/city_mapping.json',
            'src/data/district_mapping.json'
        ]
        
        for file_path in json_files:
            with self.subTest(file_path=file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.fail(f"File {file_path} is not valid JSON: {e}")
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        self.assertTrue(os.path.exists('requirements.txt'), "requirements.txt does not exist")
    
    def test_readme_exists(self):
        """Test that README.md exists."""
        self.assertTrue(os.path.exists('README.md'), "README.md does not exist")

if __name__ == '__main__':
    unittest.main()

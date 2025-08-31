import pandas as pd
import numpy as np
import sqlite3
import joblib
import tensorflow as tf
from tensorflow import keras
import random
import os

# Define paths to saved model components
MODEL_PATH = 'price_prediction_model.keras'
PREPROCESSOR_PATH = 'preprocessor.pkl'
SCALER_PATH = 'y_scaler.pkl'
DB_PATH = 'PandA.db'
OUTPUT_PATH = 'random_predictions.csv'

# Load the saved model and preprocessing components
try:
    model = keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    y_scaler = joblib.load(SCALER_PATH)
    print("Successfully loaded model and preprocessing components")
except Exception as e:
    print(f"Error loading model components: {e}")
    exit(1)

# Connect to the database and fetch all rows
try:
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM Listings"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Successfully loaded {len(df)} rows from database")
except Exception as e:
    print(f"Error loading data from database: {e}")
    exit(1)

# Preprocessing steps to match the original script
df.dropna(subset=['price'], inplace=True)
df = df[pd.to_numeric(df['price'], errors='coerce').notnull()]
df['price'] = df['price'].astype(float)
df = df[df['price'] > 0]

# Feature Selection: Define features to use
numerical_features = ['price', 'beds', 'livings', 'wc', 'area', 'street_width', 'user.review', 'width', 'length', 'refresh', 'last_update', 'create_time']
categorical_features = ['category', 'street_direction', 'city_id', 'district_id', 'advertiser_type']
text_feature = 'content_lemmatized'
bool_features_to_create = ['imgs', 'user.img']

# Check for existence of selected features
required_features = numerical_features + categorical_features + [text_feature] + bool_features_to_create
missing_cols = [col for col in required_features if col not in df.columns and col != 'price']
if missing_cols:
    print(f"Warning: The following selected features are missing from the DataFrame: {missing_cols}")
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    if text_feature not in df.columns:
        text_feature = None
        print("Disabling text feature processing as column is missing.")
    bool_features_to_create = [col for col in bool_features_to_create if col in df.columns]

# Drop rows where 'area' is missing
if 'area' in df.columns:
    df.dropna(subset=['area'], inplace=True)
    df = df[pd.to_numeric(df['area'], errors='coerce').notnull()]
    df['area'] = df['area'].astype(float)
else:
    print("Warning: 'area' column not found.")

# Normalize timestamp features
timestamp_features = ['refresh', 'last_update', 'create_time']
for col in timestamp_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Normalize by converting Unix timestamp to days since min timestamp
        df[col] = (df[col] - df[col].min()) / (24 * 3600)  # Convert to days
        df[col].fillna(df[col].median(), inplace=True)  # Impute missing with median
    else:
        print(f"Warning: '{col}' column not found.")

# Create Boolean Features
if 'imgs' in df.columns:
    df['imgs_bool'] = df['imgs'].apply(lambda x: 0 if pd.isna(x) or x in ['', '[]', '{}', None] else 1)
else:
    df['imgs_bool'] = 0
    
if 'user.img' in df.columns:
    df['user.img_bool'] = df['user.img'].apply(lambda x: 0 if pd.isna(x) or x in ['', None] else 1)
else:
    df['user.img_bool'] = 0

# Ensure categorical features are treated as strings
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Fill missing text data with empty string
if text_feature in df.columns:
    df[text_feature] = df[text_feature].fillna('')

# Get target variable
y_true = df['price'].values

# Sample 30 random rows (or less if dataset is smaller)
sample_size = min(100, len(df))
random_indices = random.sample(range(len(df)), sample_size)
df_sample = df.iloc[random_indices].copy()
y_sample = df_sample['price'].values

# Get features ready for prediction
X_sample = df_sample.copy()
X_sample_processed = preprocessor.transform(X_sample)

# Make predictions
y_pred_scaled = model.predict(X_sample_processed)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

# Calculate error percentage
error_pct = np.abs((y_sample - y_pred) / y_sample) * 100

# Create results DataFrame
results = pd.DataFrame({
    'row_number': random_indices,
    'id': df_sample['id'].values,
    'real_price': y_sample,
    'predicted_price': y_pred,
    'error_percentage': error_pct
})

# Save to CSV
results.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")

# Display summary
print("\nPrediction Summary:")
print(f"Average Error Percentage: {results['error_percentage'].mean():.2f}%")
print(f"Min Error Percentage: {results['error_percentage'].min():.2f}%")
print(f"Max Error Percentage: {results['error_percentage'].max():.2f}%")  
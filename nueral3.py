import pandas as pd
import numpy as np
import sqlite3
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. Data Loading ---
db_path = 'PandA.db'
table_name = 'Listings'

try:
    conn = sqlite3.connect(db_path)
    # Load only necessary columns initially to save memory
    # We'll refine the feature list later
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Successfully loaded {len(df)} rows from {table_name}")
    # print("Initial columns:", df.columns.tolist()) # Optional: see all loaded columns

except Exception as e:
    print(f"Error loading data from {db_path}: {e}")
    exit()

# --- 2. Data Cleaning & Preprocessing ---

print("Starting preprocessing...")

# Basic Cleaning: Drop rows where the target variable 'price' is missing or invalid
df.dropna(subset=['price'], inplace=True)
df = df[pd.to_numeric(df['price'], errors='coerce').notnull()]
df['price'] = df['price'].astype(float) # Ensure price is float
df = df[df['price'] > 0] # Assuming price must be positive

# Feature Selection: Define features to use
# Numerical features (potential candidates)
numerical_features = ['price', 'beds', 'livings', 'wc', 'area', 'street_width', 'user.review', 'width', 'length']
# Categorical features (potential candidates)
categorical_features = ['category', 'street_direction', 'city_id', 'district_id', 'advertiser_type']
# Text feature
text_feature = 'content_lemmatized'
# Boolean features to be created
bool_features_to_create = ['imgs', 'user.img'] # We'll create '_bool' versions

# Check for existence of selected features and drop rows missing crucial ones like 'area'
required_features = numerical_features + categorical_features + [text_feature] + bool_features_to_create
missing_cols = [col for col in required_features if col not in df.columns and col != 'price'] # price is target
if missing_cols:
    print(f"Warning: The following selected features are missing from the DataFrame: {missing_cols}")
    # Adjust features list or handle missing columns appropriately
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    if text_feature not in df.columns:
         text_feature = None # Disable text processing if column missing
         print("Disabling text feature processing as column is missing.")
    bool_features_to_create = [col for col in bool_features_to_create if col in df.columns]


# Drop rows where 'area' is missing, as it's important
if 'area' in df.columns:
    df.dropna(subset=['area'], inplace=True)
    df = df[pd.to_numeric(df['area'], errors='coerce').notnull()]
    df['area'] = df['area'].astype(float)
else:
    print("Warning: 'area' column not found. Cannot give it higher weight implicitly.")


# Create Boolean Features as requested
# Check if 'imgs' contains non-empty JSON array-like string or any content
if 'imgs' in df.columns:
    df['imgs_bool'] = df['imgs'].apply(lambda x: 0 if pd.isna(x) or x in ['', '[]', '{}', None] else 1)
else:
     df['imgs_bool'] = 0 # Default if column missing
# Check if 'user.img' contains non-empty string/URL
if 'user.img' in df.columns:
    df['user.img_bool'] = df['user.img'].apply(lambda x: 0 if pd.isna(x) or x in ['', None] else 1)
else:
    df['user.img_bool'] = 0 # Default if column missing

boolean_features = [col + '_bool' for col in bool_features_to_create if col in df.columns]

# Separate target variable
y = df['price']
# Select final features for X (exclude original bool columns, target, and irrelevant ones)
features_for_X = [col for col in numerical_features if col != 'price'] + categorical_features + boolean_features
if text_feature and text_feature in df.columns:
    features_for_X.append(text_feature)
    # Fill missing text data with empty string before vectorization
    df[text_feature] = df[text_feature].fillna('')
else:
    text_feature = None # Ensure it's None if not used

# Keep only necessary columns in X
X = df[features_for_X].copy() # Use .copy() to avoid SettingWithCopyWarning

# Define numerical features *for preprocessing* (excluding the target)
numerical_features_proc = [col for col in numerical_features if col != 'price' and col in X.columns]

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")


# --- Preprocessing Pipelines ---

# Numerical features: Impute missing with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Standard scaling is common for NNs
])

# Categorical features: Impute missing with a constant placeholder, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # NN needs dense arrays
])

# Text feature: TF-IDF Vectorization
if text_feature:
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words=None)) # Use None if already lemmatized/no stop words needed, or specify language like 'arabic' if needed
    ])

# --- Column Transformer ---
# Combine preprocessing steps for different column types

preprocessor_list = []

if numerical_features_proc:
    preprocessor_list.append(('num', numeric_transformer, numerical_features_proc))
if categorical_features:
     # Ensure categorical features are treated as strings for imputation/encoding
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    preprocessor_list.append(('cat', categorical_transformer, categorical_features))
if boolean_features:
     preprocessor_list.append(('bool', 'passthrough', boolean_features))

if text_feature:
    preprocessor_list.append(('text', text_transformer, text_feature))

# Create the ColumnTransformer
# If a feature type list is empty, it won't be added to preprocessor_list, avoiding errors
if not preprocessor_list:
     print("Error: No features selected or available for preprocessing.")
     exit()

preprocessor = ColumnTransformer(transformers=preprocessor_list, remainder='drop') # Drop any columns not specified


# --- Apply Preprocessing ---
# Fit the preprocessor on the training data and transform both train and test
print("Fitting preprocessor and transforming data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after transformation (useful for understanding input shape)
try:
    feature_names_out = preprocessor.get_feature_names_out()
    # print("Feature names after preprocessing:", feature_names_out) # Can be very long!
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")
except Exception as e:
    print(f"Could not get feature names: {e}")
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")


# Scale the target variable (Price) - Often helps NN training
# Important: Fit scaler ONLY on y_train
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))


# --- 4. Neural Network Design ---
input_dim = X_train_processed.shape[1] # Number of features after preprocessing

def build_nn_model(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2), # Regularization
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear") # Output layer for regression (linear activation)
        ]
    )
    # Adam optimizer is generally a good choice
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", 
                  optimizer=optimizer,
                  metrics=["mean_absolute_error"]) # Monitor MAE during training
    return model

model = build_nn_model(input_dim)
model.summary()

# --- 5. Model Training ---
print("Starting model training...")

# Use EarlyStopping to prevent overfitting and save time
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_processed,
    y_train_scaled,
    epochs=100, # Adjust number of epochs as needed
    batch_size=64, # Adjust batch size based on memory/data size
    validation_split=0.2, # Use part of the training data for validation during training
    callbacks=[early_stopping],
    verbose=1 # Set to 0 for less output, 1 for progress bar
)

print("Training finished.")

# --- 6. Model Evaluation ---
print("Evaluating model...")

# Predict on the processed test set
y_pred_scaled = model.predict(X_test_processed)

# Inverse transform the predictions and true values to the original price scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_orig = y_test.values # Already in original scale

# Calculate Metrics
r2 = r2_score(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)

print("\n--- Evaluation Metrics (Original Scale) ---")
print(f"R-squared ($R^2$): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} (SAR)")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (SAR)")
valid_indices = y_test_orig > 1e-5
ape = np.mean(np.abs((y_test_orig[valid_indices] - y_pred[valid_indices].flatten()) / y_test_orig[valid_indices])) * 100
print(f"Average Percentage Error (APE): {ape:.2f}%")
print("-------------------------------------------")


model.save('price_prediction_model.keras')  


import joblib

# Save the preprocessor (ColumnTransformer)
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessing pipeline saved to 'preprocessor.pkl'")

# Save the target scaler
joblib.dump(y_scaler, 'y_scaler.pkl')
print("Target scaler saved to 'y_scaler.pkl'")


import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
import schedule
import time
import threading
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add a lock for model access
model_lock = threading.RLock()

# Load the model, preprocessor, scaler, and mappings
model = tf.keras.models.load_model('price_prediction_model.keras')
preprocessor = joblib.load('preprocessor.pkl')
y_scaler = joblib.load('y_scaler.pkl')

with open('category_mapping.json', 'r', encoding='utf-8') as f:
    category_mapping = json.load(f)
with open('city_mapping.json', 'r', encoding='utf-8') as f:
    city_mapping = json.load(f)
with open('district_mapping.json', 'r', encoding='utf-8') as f:
    district_mapping = json.load(f)

# Extract categories from preprocessor for street_direction and advertiser_type
categorical_features = ['category', 'street_direction', 'city_id', 'district_id', 'advertiser_type']
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
street_direction_categories = onehot_encoder.categories_[categorical_features.index('street_direction')]
advertiser_type_categories = onehot_encoder.categories_[categorical_features.index('advertiser_type')]

# File to store feedback data for retraining
FEEDBACK_DATA_FILE = 'feedback_training_data.csv'

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return metadata for the client application"""
    return jsonify({
        'category_mapping': category_mapping,
        'city_mapping': city_mapping,
        'district_mapping': district_mapping,
        'street_direction_categories': street_direction_categories.tolist(),
        'advertiser_type_categories': advertiser_type_categories.tolist()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receive features and return price prediction"""
    input_data = request.json
    
    # Print input features to terminal for debugging
    print("Input features:")
    for key, value in input_data.items():
        print(f"{key}: {value}")

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform input using preprocessor
    input_processed = preprocessor.transform(input_df)

    # Make prediction with lock to ensure model is not being retrained
    with model_lock:
        prediction_scaled = model.predict(input_processed)
        prediction = y_scaler.inverse_transform(prediction_scaled)
        predicted_price = prediction[0][0]
    
    return jsonify({
        'predicted_price': float(predicted_price)
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Receive feedback about predictions for model improvement"""
    data = request.json
    
    # Get the predicted and actual prices
    predicted_price = float(data['predicted_price'])
    actual_price = float(data['actual_price'])
    
    # Price range validation - reject feedback if outside acceptable range
    if actual_price < 300000 or actual_price > 6000000:
        return jsonify({
            'message': f"Feedback rejected: Actual price (${actual_price:.2f}) is outside our accepted range of $300,000 to $6,000,000.",
            'saved_for_retraining': False
        })
    
    # Calculate deviation percentage
    deviation_percent = abs(predicted_price - actual_price) / actual_price * 100
    
    # Extract the original input features
    feedback_data = data['input_features']
    
    # Add the actual price
    feedback_data['actual_price'] = actual_price
    
    # If deviation is more than 20%, save the data for retraining
    response = {
        'message': f"Thank you for your feedback! Our prediction was within 20% of the actual price. (Deviation: {deviation_percent:.2f}%)",
        'saved_for_retraining': False
    }
    
    if deviation_percent > 20:
        save_feedback_for_retraining(feedback_data)
        response = {
            'message': f"Thank you for your feedback! Your input will help improve our model. (Deviation: {deviation_percent:.2f}%)",
            'saved_for_retraining': True
        }
    
    return jsonify(response)

def save_feedback_for_retraining(feedback_data):
    """Save feedback data to CSV file for model retraining"""
    # Convert feedback data to DataFrame
    df = pd.DataFrame([feedback_data])
    
    # Ensure boolean fields are properly formatted
    if 'imgs_bool' in df.columns:
        df['imgs_bool'] = df['imgs_bool'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 't', 'y', 'yes'] else 0)
    if 'user.img_bool' in df.columns:
        df['user.img_bool'] = df['user.img_bool'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 't', 'y', 'yes'] else 0)
    
    # Ensure numerical fields are properly formatted
    for col in ['beds', 'livings', 'wc', 'area', 'street_width', 'user.review', 'width', 'length']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(FEEDBACK_DATA_FILE)
    
    # Append data to CSV file
    df.to_csv(FEEDBACK_DATA_FILE, mode='a', header=not file_exists, index=False)
    
    print(f"Saved feedback data for retraining. Current time: {datetime.now()}")

def retrain_model():
    """Retrain the model with feedback data and reset the feedback file"""
    global model
    
    print(f"Starting model retraining at {datetime.now()}")
    
    # Check if feedback data file exists and has data
    if not os.path.isfile(FEEDBACK_DATA_FILE) or os.stat(FEEDBACK_DATA_FILE).st_size == 0:
        print("No feedback data available for retraining")
        return
    
    try:
        # Load feedback data
        feedback_df = pd.read_csv(FEEDBACK_DATA_FILE)
        
        if len(feedback_df) == 0:
            print("Feedback file exists but contains no data")
            return
            
        print(f"Retraining model with {len(feedback_df)} new data points")
        
        # Ensure all required columns are present
        required_columns = [
            'beds', 'livings', 'wc', 'area', 'street_width', 'user.review', 
            'width', 'length', 'category', 'street_direction', 'city_id', 
            'district_id', 'advertiser_type', 'imgs_bool', 'user.img_bool', 
            'content_lemmatized', 'actual_price'
        ]
        
        for col in required_columns:
            if col not in feedback_df.columns and col != 'actual_price':
                if col in ['imgs_bool', 'user.img_bool']:
                    feedback_df[col] = 0  # Default for boolean columns
                elif col in ['beds', 'livings', 'wc', 'area', 'street_width', 'user.review', 'width', 'length']:
                    feedback_df[col] = np.nan  # Default for numerical columns
                else:
                    feedback_df[col] = ''  # Default for categorical/text columns
        
        # Extract features and target
        X_feedback = feedback_df.drop('actual_price', axis=1)
        y_feedback = feedback_df['actual_price'].values.reshape(-1, 1)
        
        # Preprocess features
        X_processed = preprocessor.transform(X_feedback)
        
        # Scale target
        y_scaled = y_scaler.transform(y_feedback)
        
        # Load a new copy of the model to train with, then update the global model with the lock
        temp_model = tf.keras.models.load_model('price_prediction_model.keras')
        
        # Retrain model with new data
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',    # Monitor loss on the validation set
            patience=10,           # Stop if no improvement after 10 epochs
            restore_best_weights=True, # Keep the best model weights found
            verbose=1              # Optional: print messages when stopping
        )

        # 2. Parameters for the model.fit() call:
        epochs = 50             
        batch_size = 64         # Consistent with main script, good standard
        validation_split = 0.15 # Use 15% of the training data for validation
                               
        callbacks = [early_stopping] # Include the early stopping callback
        verbose = 1             


        # Assuming 'temp_model' is your compiled model for the day
        # Assuming 'X_processed' and 'y_scaled' are the full, processed datasets for the day

        print("Starting daily model retraining...")

        history = temp_model.fit(
            X_processed,
            y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split, # Automatically splits data passed to fit
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save updated model
        temp_model.save('price_prediction_model.keras')
        
        # Update the global model with lock to ensure thread safety
        with model_lock:
            model = temp_model
            
        print(f"Model retrained and saved at {datetime.now()}")
        
        # Reset feedback file
        os.remove(FEEDBACK_DATA_FILE)
        print("Feedback data file reset")
        
    except Exception as e:
        print(f"Error during model retraining: {e}")

def run_scheduler():
    """Run the scheduler in a separate thread"""
    # Schedule the retraining task to run at 11:59 PM every day
    schedule.every().day.at("00:05").do(retrain_model)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=True) 
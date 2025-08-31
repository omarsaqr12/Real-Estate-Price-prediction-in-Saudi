# Real Estate Price Prediction - Client-Server Architecture

This application provides real estate price prediction using a machine learning model. It follows a client-server architecture where:

1. The server hosts the ML model and provides APIs for predictions and feedback
2. The client renders the UI and communicates with the server via HTTP requests

## Architecture Overview

### Server Component (`server.py`)

- Hosts the ML model, preprocessor, and all data files
- Provides RESTful APIs for:
  - `/api/metadata` - Returns category mappings and other metadata
  - `/api/predict` - Makes price predictions based on input features
  - `/api/feedback` - Collects feedback for model improvement
- Handles model retraining automatically based on collected feedback
- Runs on port 5000 by default

### Client Component (`client.py`)

- Renders the web UI (HTML/CSS)
- Collects user input via forms
- Communicates with the server API to get predictions
- Provides feedback mechanism to improve the model
- Runs on port 8000 by default

## Setup Instructions

### Manual Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Start the server:

   ```
   python server.py
   ```

3. Start the client (in a separate terminal):

   ```
   python client.py
   ```

4. Access the application in your browser at: http://localhost:8000

   ```

   ```

## Files

- `server.py` - The server application hosting the ML model
- `client.py` - The client application with the user interface
- `templates/index.html` - HTML template for the client interface
- `price_prediction_model.keras` - The trained ML model
- `preprocessor.pkl` - Feature preprocessing pipeline
- `y_scaler.pkl` - Target variable scaler
- `category_mapping.json`, `city_mapping.json`, `district_mapping.json` - Mapping files for categorical variables
- `feedback_training_data.csv` - File to store feedback data for retraining (created automatically)

## API Documentation

### GET /api/metadata

Returns metadata required by the client application including category mappings, city mappings, etc.

### POST /api/predict

Accepts property features and returns a price prediction.

Request body:

```json
{
  "beds": 3,
  "livings": 2,
  "wc": 2,
  "area": 150,
  "category": "apartment",
  ...
}
```

Response:

```json
{
  "predicted_price": 500000
}
```

### POST /api/feedback

Submits feedback for a prediction to improve the model.

Request body:

```json
{
  "predicted_price": 500000,
  "actual_price": 520000,
  "input_features": {
    "beds": 3,
    "livings": 2,
    ...
  }
}
```

Response:

```json
{
  "message": "Thank you for your feedback!",
  "saved_for_retraining": true
}
```

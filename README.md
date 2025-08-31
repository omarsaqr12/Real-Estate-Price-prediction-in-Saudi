# 🏠 Saudi Real Estate Price Prediction System

**A machine learning-powered real estate price prediction application built with Python, featuring a client-server architecture for scalable deployment and continuous model improvement.**

## ✨ Key Features

- **ML-Powered Predictions**: Neural network model trained on Saudi Arabia real estate data
- **Client-Server Architecture**: Scalable Flask backend with web-based client interface
- **Real-time Feedback Loop**: Collects user feedback to continuously improve predictions
- **Interactive Web UI**: User-friendly interface for property feature input
- **Automatic Model Retraining**: Self-improving system based on collected feedback
- **Arabic Text Processing**: Advanced NLP for Arabic property descriptions

## 🏗️ Project Structure

```
Real-Estate-Price-prediction-in-Saudi/
├── 📁 src/                    # Source code
│   ├── 📁 models/            # Trained ML models
│   ├── 📁 data/              # Data mapping files
│   ├── 📁 scripts/           # Training and preprocessing scripts
│   └── 📁 web/               # Web application
├── 📁 docs/                  # Documentation
├── 📁 examples/              # Sample outputs and examples
├── 📁 tests/                 # Test files
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation
└── README.md                 # This file
```

## 🛠️ Tech Stack

- **Backend**: Flask, TensorFlow, Pandas, NumPy
- **ML Pipeline**: Neural networks, feature preprocessing, automated retraining
- **Frontend**: HTML/CSS with responsive design
- **Data Processing**: JSON mappings, pickle serialization, Arabic NLP
- **Database**: SQLite with advanced preprocessing

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi.git
   cd Real-Estate-Price-prediction-in-Saudi
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Get the cleaned dataset from [Google Drive](https://drive.google.com/drive/folders/1PT3MuIW0eej5w4jTOENe_C3g1o3o7LdN)
   - Place `PandA.db` in the project root directory

### Running the Application

1. **Start the server:**
   ```bash
   python src/web/server.py
   ```

2. **Start the client (in a separate terminal):**
   ```bash
   python src/web/client.py
   ```

3. **Access the application:**
   - Open your browser and go to: http://localhost:8000

## 📊 Model Performance

Our trained model achieves excellent performance on Saudi real estate data:

- **R-squared (R²)**: 0.9235
- **Mean Absolute Error (MAE)**: 179,996.90 SAR
- **Root Mean Squared Error (RMSE)**: 303,539.51 SAR
- **Average Percentage Error (APE)**: 16.11%

## 🔧 Development

### Training New Models

```bash
python src/scripts/train_model.py
```

### Data Preprocessing

```bash
python src/scripts/preprocess_data.py
```

### Generating Sample Predictions

```bash
python src/scripts/generate_samples.py
```

## 📁 File Descriptions

### Core Application
- `src/web/server.py` - Flask server hosting the ML model and APIs
- `src/web/client.py` - Web client with user interface
- `src/web/templates/index.html` - HTML template for the client interface

### Machine Learning
- `src/models/price_prediction_model.keras` - Trained neural network model
- `src/models/preprocessor.pkl` - Feature preprocessing pipeline
- `src/models/y_scaler.pkl` - Target variable scaler
- `src/scripts/train_model.py` - Model training script (renamed from nueral3.py)
- `src/scripts/preprocess_data.py` - Data preprocessing script (renamed from lamm.py)

### Data Files
- `src/data/category_mapping.json` - Property category mappings
- `src/data/city_mapping.json` - City ID mappings
- `src/data/district_mapping.json` - District ID mappings

### Documentation
- `docs/performance_metrics.md` - Detailed model performance metrics
- `docs/feedback_notes.md` - Notes on model improvement
- `docs/dataset_info.md` - Information about the dataset

## 🌐 API Documentation

### GET /api/metadata
Returns metadata required by the client application including category mappings, city mappings, etc.

### POST /api/predict
Accepts property features and returns a price prediction.

**Request body:**
```json
{
  "beds": 3,
  "livings": 2,
  "wc": 2,
  "area": 150,
  "category": "apartment",
  "city_id": 1,
  "district_id": 15
}
```

**Response:**
```json
{
  "predicted_price": 500000
}
```

### POST /api/feedback
Submits feedback for a prediction to improve the model.

**Request body:**
```json
{
  "predicted_price": 500000,
  "actual_price": 520000,
  "input_features": {
    "beds": 3,
    "livings": 2,
    "area": 150
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Saudi real estate data providers
- CALIMA MSA Arabic NLP tools
- TensorFlow and scikit-learn communities

## 📞 Contact

- **Project Link**: [https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi](https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi)
- **Issues**: [GitHub Issues](https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi/issues)

---

**Made with ❤️ for the Saudi real estate market**

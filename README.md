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

## 📊 Model Selection & Evaluation (Phase III)

Before settling on our final Neural Network architecture, we conducted extensive evaluation of **9 different machine learning algorithms** to determine the best approach for Saudi real estate price prediction.

### Authors
- **Omar Saqr** - Computer Engineering, AUC, Cairo Egypt (omar_saqr@aucegypt.edu)
- **Aabed Elghadbaan** - Computer Engineering, AUC, Cairo Egypt (abed@aucegypt.edu)

### Dataset Characteristics
- **~650,000 rows** of real estate listings
- **~53,000 features** including TF-IDF matrix from Arabic text processing
- High dimensionality requiring careful algorithm selection

### 📈 Model Comparison Summary

| Algorithm | Type | R² / Accuracy | RMSE / Precision | Key Findings |
|-----------|------|---------------|------------------|--------------|
| **Neural Network** | Regression | 0.62 | 354,391 SAR | Best for non-linear patterns, scalable architecture |
| **Random Forest** | Regression | 0.65 | 340,114 SAR | Good ensemble approach, handles high dimensionality |
| **Linear Regression** | Regression | 0.60 | 363,598 SAR | Simple baseline, assumes linearity |
| **CART** | Regression | 0.46 | 452,675 SAR | Interpretable but prone to overfitting |
| **KNN** | Regression | 0.41 | 448,314 SAR | Struggles with high dimensionality |
| **SVM** | Classification | 66% acc | 66% precision | Good accuracy but computationally expensive |
| **C4.5 Decision Tree** | Classification | 50% acc | 51% precision | Better than ID3 due to pruning |
| **Logistic Regression** | Classification | 53% acc | 50% precision | Not suited for continuous predictions |
| **ID3 Decision Tree** | Classification | 45% acc | 40% precision | Baseline classification approach |

*All models evaluated using 5-fold cross-validation*

---

### 🔬 Detailed Model Analysis

<details>
<summary><b>1. Linear Regression</b></summary>

**Pros:**
- Simple and interpretable (linear equation parameters)
- Computationally lightweight, fast predictions
- Faster to train than random forests
- Compatible with regression output format
- Suitable for large datasets (650K rows, 53K features)

**Cons:**
- Assumes linearity (unlikely with TF-IDF features)
- Assumes feature independence
- Doesn't benefit from high row-count like KNN

**Parameters:**
- Loss function: `squared_loss` (standard, punishes large errors)
- Max iterations: 1000 (default, avoids computational waste)
- Tolerance: 1e-3 (prevents overfitting)
- Fixed random_state for reproducibility

**Results:** R² = 0.60 | MSE = 132,203,556,180 | RMSE = 363,598 SAR

</details>

<details>
<summary><b>2. Logistic Regression</b></summary>

**Pros:**
- Simple and interpretable via odds ratios
- Computationally lightweight
- No assumptions about input feature distributions

**Cons:**
- Not built for continuous predictions (incompatible with price regression)
- Requires discretizing prices into categories

**Parameters:**
- Price divided into 3 categories (low/medium/high by percentile)
- Max iterations: 1000
- n_jobs=-1 (utilize all CPU cores)
- Fixed random_state for reproducibility

**Results:** Accuracy = 53% | Precision = 50% | Recall = 65%

*Note: 53% is a significant improvement over 33% random guessing baseline*

</details>

<details>
<summary><b>3. K-Nearest Neighbors (KNN)</b></summary>

**Pros:**
- Intuitive algorithm (finds nearest neighbors)
- Captures local patterns across different dimensions
- Flexible for both regression and classification
- No assumptions about feature distributions
- Benefits from large datasets

**Cons:**
- Computationally expensive (requires traversing all points)
- Loses effectiveness with high dimensionality (curse of dimensionality)
- TF-IDF features make points concentrate on edges of space

**Parameters:**
- k = 199 (prime number less than √n)
- Fixed random_state for reproducibility

**Results:** R² = 0.41 | MSE = 198,305,634,270 | RMSE = 448,314 SAR

*Note: Poor performance due to high dimensionality distorting distance calculations*

</details>

<details>
<summary><b>4. Neural Network (Selected Model)</b></summary>

**Pros:**
- Excellent at capturing non-linear relationships
- Captures hidden trends and local patterns
- Flexible architecture (can be fast or slow)
- Works for both regression and classification
- No assumptions about input distributions
- Benefits from large datasets

**Cons:**
- Can be computationally expensive
- Lacks transparency ("black box" reasoning)
- Architecture selection is challenging

**Architecture:**
- Batch size: 32 (improves generalization)
- Three hidden layers:
  - Dense features layer (512 neurons)
  - TF-IDF features layer (256 neurons)
  - Concatenation layer (128 neurons)
- Activation: ReLU (handles sparse data efficiently)
- Loss function: MSE (standard for regression)
- Optimizer: Adam (adaptive learning rate)

**Initial Results:** R² = 0.62 | MSE = 125,593,378,370 | RMSE = 354,391 SAR

</details>

<details>
<summary><b>5. Decision Trees - ID3</b></summary>

**Pros:**
- Simple and interpretable (visualizable decision process)
- Fast predictions, computationally lightweight
- Doesn't need normalization
- Provides feature importance insights

**Cons:**
- Not built for continuous predictions
- Known for overfitting issues
- Outliers drastically affect tree structure
- Gives more importance to features with more values

**Parameters:**
- Continuous features discretized into 10 equal-width bins
- Max depth: 20 (prevents overfitting)
- Min samples split: 100
- Criterion: entropy (ID3 default)

**Results:** Accuracy = 45% | Precision = 40% | Recall = 49%

*Note: Significantly better than 10% random guessing for 10 categories*

</details>

<details>
<summary><b>6. Decision Trees - C4.5</b></summary>

**Pros:**
- All ID3 advantages
- Reduced overfitting through pruning
- Considers number of branches in splits

**Cons:**
- Still not ideal for continuous predictions
- Outlier sensitivity remains

**Parameters:**
- Same as ID3, with ccp_alpha=0.01 (prunes branches unless they reduce cost by ≥0.01)

**Results:** Accuracy = 50% | Precision = 51% | Recall = 46%

*Note: Slight improvement over ID3 due to pruning*

</details>

<details>
<summary><b>7. CART (Classification and Regression Trees)</b></summary>

**Pros:**
- Interpretable and visualizable
- Fast predictions
- Can be used for regression (unlike ID3/C4.5)
- Provides feature importance

**Cons:**
- Overfitting issues
- Outlier sensitivity
- Feature value bias (like ID3)

**Parameters:**
- Max depth: 20
- Min samples split: 100
- Criterion: MSE (regression default)

**Results:** R² = 0.46 | MSE = 204,915,512,079 | RMSE = 452,675 SAR

</details>

<details>
<summary><b>8. Random Forest</b></summary>

**Pros:**
- Reduces overfitting through ensemble approach
- Handles high dimensionality well (random feature subsets)
- Provides feature importance insights
- Supports regression

**Cons:**
- Less interpretable than single trees
- Can be computationally expensive
- Less transparent decision-making

**Parameters:**
- Max depth: None (ensemble reduces overfitting)
- Number of trees: 200
- Criterion: MSE

**Results:** R² = 0.65 | MSE = 115,678,111,657 | RMSE = 340,114 SAR

*Note: Best initial R² score among all models*

</details>

<details>
<summary><b>9. Support Vector Machine (SVM)</b></summary>

**Pros:**
- Effective with non-linear boundaries (RBF kernel)
- Maximizes margin for better generalization
- Performs well with high dimensionality
- Good noise resistance via regularization

**Cons:**
- Not built for continuous predictions
- High computational complexity (slow)
- No probability estimates

**Parameters:**
- Price discretized into 10 equal-width categories
- Regularization parameter: 1 (standard)
- Kernel: RBF

**Results:** Accuracy = 66% | Precision = 66% | Recall = 62%

*Note: Best classification accuracy but too slow for practical use*

</details>

---

### 🏆 Top 3 Algorithm Selection

For regression tasks, excluding classification-only models (Logistic Regression, ID3, C4.5, SVM), the top performers were:

| Rank | Algorithm | R² Score | Key Advantage |
|------|-----------|----------|---------------|
| 1 | Random Forest | 0.65 | Ensemble approach, handles dimensionality |
| 2 | Neural Network | 0.62 | Non-linear patterns, scalable architecture |
| 3 | Linear Regression | 0.60 | Simple baseline |

### 🎯 Why We Chose Neural Networks

Despite Random Forest having a slightly higher initial R² (0.65 vs 0.62), we selected **Neural Networks** for the final implementation:

1. **Architecture Flexibility**: Neural networks allow extensive experimentation with layer sizes, activation functions, and architecture design
2. **Scalability**: Better suited for the production environment with client-server architecture
3. **Non-linearity**: Excels at capturing complex, non-linear relationships in real estate pricing
4. **Feature Learning**: Separate layers for dense and TF-IDF features allow specialized pattern recognition
5. **Improvement Potential**: Initial results (R² = 0.62) were achieved with small layer sizes; significant room for optimization
6. **Real-time Feedback**: Neural networks integrate well with our continuous learning feedback loop

---

## 📊 Final Model Performance

After extensive tuning and optimization, our Neural Network achieves excellent performance:

- **R-squared (R²)**: 0.9235
- **Mean Absolute Error (MAE)**: 179,996.90 SAR
- **Root Mean Squared Error (RMSE)**: 303,539.51 SAR
- **Average Percentage Error (APE)**: 16.11%

*This represents a **49% improvement** in R² from initial evaluation (0.62 → 0.9235)*

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

# Dataset Information

## Data Source

The cleaned dataset used for training this real estate price prediction model is available at:
[Google Drive Link](https://drive.google.com/drive/folders/1PT3MuIW0eej5w4jTOENe_C3g1o3o7LdN)

## Dataset Description

- **Source**: Saudi Arabia real estate listings
- **Format**: SQLite database (PandA.db)
- **Size**: Large dataset with multiple property features
- **Features**: Includes property characteristics, location data, and pricing information

## Data Preprocessing

The dataset has been cleaned and preprocessed using:

- Arabic text lemmatization using CALIMA MSA disambiguator
- Feature engineering for numerical and categorical variables
- Text vectorization for property descriptions
- Standardization and scaling of numerical features

## Usage

To use this dataset:

1. Download from the provided Google Drive link
2. Place the database file in the appropriate directory
3. Run the preprocessing scripts in `src/scripts/`
4. Train the model using `src/scripts/train_model.py`

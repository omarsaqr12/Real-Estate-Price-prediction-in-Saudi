import flask
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import requests
import json
import re
from camel_tools.disambig.mle import MLEDisambiguator

app = Flask(__name__)
app.secret_key = 'realestateprediction'  # Required for flash messages

# Initialize the MSA disambiguator
try:
    mle_msa = MLEDisambiguator.pretrained("calima-msa-r13")
except Exception as e:
    print(f"Error loading disambiguator: {e}")
    mle_msa = None

# Helper function to clean suspicious characters
def safe_content(text):
    text = text.replace('\\', ' ')  # Remove problematic backslashes
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep Arabic, digits, and whitespace
    return text.strip()

# Lemmatize text using camel tools
def lemmatize_text(text):
    if not text:
        return ""
    
    if not mle_msa:
        return text  # Return original text if disambiguator not available
    
    cleaned = safe_content(text)
    
    if cleaned:
        try:
            analysis = mle_msa.disambiguate(cleaned.split())
            lemmas = [
                token.analyses[0].analysis["lex"] if token.analyses else token.word
                for token in analysis
            ]
            return " ".join(lemmas)
        except Exception as e:
            print(f"Lemmatization error: {e}")
            return text  # Fallback to original
    return ""

# Configure the server URL here
SERVER_URL = 'http://localhost:5000'  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    # Fetch metadata from the server API
    try:
        metadata_response = requests.get(f'{SERVER_URL}/api/metadata')
        metadata = metadata_response.json()
        category_mapping = metadata['category_mapping']
        city_mapping = metadata['city_mapping']
        district_mapping = metadata['district_mapping']
        street_direction_categories = metadata['street_direction_categories']
        advertiser_type_categories = metadata['advertiser_type_categories']
    except Exception as e:
        flash(f"Error connecting to server: {str(e)}")
        return render_template(
            'index.html',
            prediction=None,
            category_mapping={},
            city_mapping={},
            district_mapping_json="{}",
            street_direction_categories=[],
            advertiser_type_categories=[]
        )
    
    if request.method == 'POST':
        # Get form data
        category = request.form['category']
        city_id = request.form['city_id']
        district_id = request.form['district_id']
        street_direction = request.form['street_direction']
        advertiser_type = request.form['advertiser_type']

        beds = request.form.get('beds', '')
        livings = request.form.get('livings', '')
        wc = request.form.get('wc', '')
        area = request.form.get('area', '')
        street_width = request.form.get('street_width', '')
        user_review = request.form.get('user.review', '')
        width = request.form.get('width', '')
        length = request.form.get('length', '')

        description = request.form.get('description', '')
        # Lemmatize the description text using the same process as in lamm.py
        content_lemmatized = lemmatize_text(description)
        
        imgs_bool = 1 if 'imgs_bool' in request.form else 0
        user_img_bool = 1 if 'user.img_bool' in request.form else 0

        # Convert empty strings to None for numerical fields
        def convert_empty_to_none(value):
            if value == '':
                return None
            try:
                return float(value)
            except:
                return None

        # Create input dictionary with exact column names from training
        input_data = {
            'beds': convert_empty_to_none(beds),
            'livings': convert_empty_to_none(livings),
            'wc': convert_empty_to_none(wc),
            'area': convert_empty_to_none(area),
            'street_width': convert_empty_to_none(street_width),
            'user.review': convert_empty_to_none(user_review),
            'width': convert_empty_to_none(width),
            'length': convert_empty_to_none(length),
            'category': category,
            'street_direction': street_direction,
            'city_id': city_id,
            'district_id': district_id,
            'advertiser_type': advertiser_type,
            'imgs_bool': imgs_bool,
            'user.img_bool': user_img_bool,
            'content_lemmatized': content_lemmatized
        }

        try:
            # Send prediction request to server API
            response = requests.post(
                f'{SERVER_URL}/api/predict',
                json=input_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['predicted_price']
            else:
                flash(f"Error from server: {response.text}")
        except Exception as e:
            flash(f"Error communicating with server: {str(e)}")

    # Render template with form data and prediction
    return render_template(
        'index.html',
        prediction=prediction,
        category_mapping=category_mapping,
        city_mapping=city_mapping,
        district_mapping_json=json.dumps(district_mapping),
        street_direction_categories=street_direction_categories,
        advertiser_type_categories=advertiser_type_categories
    )

@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        # Get the predicted and actual prices
        predicted_price = float(request.form['predicted_price'])
        actual_price = float(request.form['actual_price'])
        
        # Extract the original input features from the hidden fields
        input_features = {}
        for key, value in request.form.items():
            if key.startswith('input_'):
                original_key = key[6:]  # Remove 'input_' prefix
                input_features[original_key] = value
        
        feedback_data = {
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'input_features': input_features
        }
        
        try:
            # Send feedback to server API
            response = requests.post(
                f'{SERVER_URL}/api/feedback',
                json=feedback_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                flash(result['message'])
            else:
                flash(f"Error sending feedback: {response.text}")
        except Exception as e:
            flash(f"Error communicating with server: {str(e)}")
        
    return redirect(url_for('index'))

@app.route('/clear')
def clear_form():
    # Redirect to the index page with a fresh form
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Run client on a different port than the server 
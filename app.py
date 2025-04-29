from flask import Flask, request, render_template, flash
import pandas as pd
import nltk
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data at startup
try:
    nltk.download('stopwords', quiet=True)
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    logger.error(f'Error initializing NLTK: {str(e)}')
    raise

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Initialize objects
ps = PorterStemmer()
vector = None
model = None

# Preprocessing function
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Load or train the model
def initialize_model():
    global vector, model
    
    try:
        model_file = 'fake_news_model.pkl'
        vectorizer_file = 'vectorizer.pkl'
        
        # Try loading pre-trained model
        if os.path.exists(model_file):
            logger.info('Loading pre-trained model...')
            try:
                # Try new format first
                with open(model_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, dict):
                        vector = loaded_data['vectorizer']
                        model = loaded_data['model']
                        logger.info('Model loaded successfully from new format')
                        return
                    else:
                        # Old format - separate files
                        model = loaded_data
                        if os.path.exists(vectorizer_file):
                            with open(vectorizer_file, 'rb') as f:
                                vector = pickle.load(f)
                            logger.info('Model loaded successfully from old format')
                            return
            except Exception as e:
                logger.warning(f'Could not load existing model: {str(e)}. Training new model...')
        else:
            logger.info('No existing model found. Training new model...')
            
        # If we get here, we need to train a new model
        logger.info('Training new model...')
        vector = TfidfVectorizer(max_features=5000)  # Limit features for faster training
        model = LogisticRegression(max_iter=1000)
        
        if not os.path.exists('Dataset/WELFake_Dataset.csv'):
            raise FileNotFoundError('Dataset file not found')
        
        logger.info('Loading dataset...')
        df = pd.read_csv('Dataset/WELFake_Dataset.csv')
        # Use only first 1000 rows for faster development
        df = df.head(1000)
        logger.info(f'Using {len(df)} rows for development')
        
        logger.info('Preprocessing data...')
        df = df.fillna('')
        df['content'] = df['title'] + ' ' + df['text']
        
        logger.info('Applying stemming...')
        df['content'] = df['content'].apply(stemming)
        
        logger.info('Preparing features...')
        X = df['content'].values
        y = df['label'].values
        
        logger.info('Vectorizing text...')
        X_vectorized = vector.fit_transform(X)
        logger.info(f'Vectorization complete. Shape: {X_vectorized.shape}')
        
        logger.info('Training logistic regression...')
        model.fit(X_vectorized, y)
        logger.info('Model training complete')
        
        # Save the model and vectorizer together
        logger.info('Saving model...')
        model_data = {
            'vectorizer': vector,
            'model': model
        }
        with open('fake_news_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        logger.info('Model trained and saved successfully')
            
    except Exception as e:
        logger.error(f'Error in initialize_model: {str(e)}')
        raise

# Prediction function
def predict_news(text):
    # Preprocess the input text
    processed_text = stemming(text)
    transformed = vector.transform([processed_text])
    pred = model.predict(transformed)[0]
    return 'Fake News' if pred == 1 else 'Real News'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news_text = request.form['news']
        if not news_text.strip():
            flash('Please enter some text.', 'error')
            return render_template('index.html')
            
        if model is None or vector is None:
            flash('Model not initialized. Please try again.', 'error')
            return render_template('index.html')
            
        prediction = predict_news(news_text)
        return render_template('result.html', prediction=prediction, news_text=news_text)
        
    except Exception as e:
        logger.error(f'Error in predict: {str(e)}')
        flash('An error occurred while processing your request.', 'error')
        return render_template('index.html')

if __name__ == '__main__':
    # Initialize the model before running the app
    initialize_model()
    app.run(debug=True)

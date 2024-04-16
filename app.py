import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the SVM model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Function for text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    # Debugging statement
    print("Preprocessed Text:", preprocessed_text)
    # Use the SVM model to predict sentiment
    predicted_label = svm_model.predict([preprocessed_text])[0]
    return predicted_label

# Streamlit app layout
st.title('Sentiment Analysis with SVM')
st.write('Enter a text to analyze its sentiment.')

# Text input for user
text_input = st.text_input('Input Text:', '')

# Button to predict sentiment
if st.button('Predict Sentiment'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        # Predict sentiment
        sentiment = predict_sentiment(text_input)
        if sentiment == 1:
            st.success('Sentiment: Positive')
        elif sentiment == 0:
            st.warning('Sentiment: Neutral')
        else:
            st.error('Sentiment: Negative')

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import numpy as np

# Load the vectorizer and ML model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

mlb = None  # Define globally for later use

# Load the MultiLabelBinarizer
with open('label_binarizer.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Load the Keras model
model = load_model('movie_genre_classifier.h5')

# Function to clean and vectorize text
def clean_and_vectorize(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    clean_words = [word for word in words if word not in stopwords.words('english')]
    clean_text = ' '.join(clean_words)
    vectorized_text = vectorizer.transform([clean_text]).toarray()
    return vectorized_text

# Function to predict genres
def predict_genres(description):
    cleaned_description = clean_and_vectorize(description)
    predictions = model.predict(cleaned_description)
    threshold = 0.3  # Adjust the threshold as needed
    binary_predictions = (predictions > threshold).astype(int)
    predicted_genres = mlb.inverse_transform(binary_predictions)
    return predicted_genres

# Streamlit App
def main():
    st.title('Movie Genre Classifier')

    # User input for description
    description = st.text_area('Enter Movie Description:', '')

    if st.button('Predict Genres'):
        if description.strip():
            genres = predict_genres(description)
            st.success(f'Predicted Genres: {genres}')
        else:
            st.warning('Please enter a description.')

    st.text('')
    st.text('Example Descriptions:')
    st.text('1. A team of explorers discover a mysterious island filled with prehistoric creatures and ancient civilizations.')
    st.text('2. A young man discovers he has superpowers and must navigate the complexities of being a hero.')

if __name__ == '__main__':
    main()

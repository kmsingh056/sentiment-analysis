import streamlit as st
import pickle
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Sentiment Analysis")

model = pickle.load(open("trained_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to perform sentiment analysis
def analyze_sentiment(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    text = " ".join(tokens)

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# Text input for manual entry
text_input = st.text_input("Write text for sentiment analysis")


# File uploader for CSV and Excel files
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Button for prediction
predict_button = st.button("Predict", type="primary")

if predict_button:
    if text_input:
        prediction = analyze_sentiment(text_input)
        if prediction == 0:
            st.write("Negative sentiment")
        else:
            st.write("Positive sentiment")

    elif uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)

        st.write("Sentiment Analysis Results:")
        st.write(df)
        df['Sentiment'] = df['Text'].apply(analyze_sentiment)
        df['Sentiment'] = df['Sentiment'].apply(lambda x: 'Negative' if x == 0 else 'Positive')
        st.write(df)

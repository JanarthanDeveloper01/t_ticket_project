import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("it_ticket_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")

# Input ticket text
ticket_text = st.text_area("Enter IT Ticket Text:")

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform text and predict category
        category = model.predict([ticket_text])[0]
        st.success(f"Predicted Category: {category}")

        # Optional: show a mock resolution (replace with real past data lookup)
        st.info("Suggested Resolution: Please refer to the IT knowledge base for similar tickets.")


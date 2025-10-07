import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("it_ticket_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")
st.write("""
This application automatically classifies IT tickets into categories (Hardware, Software, Network, etc.) 
and suggests the best resolution based on past incidents.
""")

# Input ticket text
ticket_text = st.text_area("Enter IT Ticket Text:", height=150)
if st.button("Classify & Suggest Resolution"):
    if ticket_text.strip() != "":
        # Transform text
        X_input = tfidf_vect.transform([ticket_text])
        
        # Predict category
        category = pipe.predict(X_input)[0]
        
        # Dummy resolution suggestion (replace with real logic)
        resolution = f"Suggested steps for {category}"
        
        st.success(f"**Predicted Category:** {category}")
        st.info(f"**Suggested Resolution:** {resolution}")
    else:
        st.warning("Please enter a ticket text to classify.")

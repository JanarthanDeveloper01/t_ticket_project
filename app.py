import streamlit as st
import joblib

# Load trained models
pipe = joblib.load("it_ticket_classifier.pkl")       # classifier
tfidf_vect = joblib.load("tfidf_vectorizer.pkl")     # TF-IDF vectorizer

# Streamlit UI
st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")

ticket_text = st.text_area(
    "Enter IT Ticket Text:", 
    height=150, 
    placeholder="Example: 'My laptop cannot connect to the VPN. Please help me resolve this issue as soon as possible.'"
)

if st.button("Classify Ticket"):
    if ticket_text and ticket_text.strip() != "":
        # Ensure input is a string
        X_input = tfidf_vect.transform([str(ticket_text)])
        
        # Predict category
        prediction = pipe.predict(X_input)[0]
        
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please enter a ticket text before clicking Classify.")

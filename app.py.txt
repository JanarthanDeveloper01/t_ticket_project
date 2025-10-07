# app.py

import streamlit as st
import joblib

# ----------------------------
# Load trained models
# ----------------------------
try:
    pipe = joblib.load("it_ticket_classifier.pkl")       # Classifier pipeline
    tfidf_vect = joblib.load("tfidf_vectorizer.pkl")     # TF-IDF vectorizer
except FileNotFoundError:
    st.error("Model files not found. Ensure 'it_ticket_classifier.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as app.py.")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="AI IT Ticket Classifier",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")
st.write("""
This application classifies IT support tickets into categories
and suggests possible actions based on the ticket content.
""")

# Input: Ticket Text
ticket_text = st.text_area("Enter IT Ticket Text:", height=200)

# Button: Classify Ticket
if st.button("Classify Ticket"):
    
    if ticket_text and ticket_text.strip() != "":
        try:
            # Ensure input is string
            X_input = tfidf_vect.transform([str(ticket_text)])
            
            # Predict category
            prediction = pipe.predict(X_input)[0]
            
            st.success(f"**Predicted Category:** {prediction}")
            
            # Optionally, you can add sample resolution suggestions
            suggestions = {
                "Technical Support": "Check system logs and remote into user machine if necessary.",
                "Product Support": "Review product documentation and provide troubleshooting steps.",
                "Customer Service": "Respond with polite acknowledgement and escalate if needed.",
                "IT Support": "Verify user credentials and network access.",
                "Billing and Payments": "Check invoice and payment history; escalate to finance team.",
                "Returns and Exchanges": "Provide RMA instructions and return label.",
                "Service Outages and Maintenance": "Notify affected users and provide estimated resolution time.",
                "Sales and Pre-Sales": "Provide product information and sales support contact.",
                "Human Resources": "Refer to HR team for employee-related queries.",
                "General Inquiry": "Respond with standard FAQ or route to correct department."
            }
            
            # Show suggestion
            st.info(f"**Suggested Action:** {suggestions.get(prediction, 'No suggestion available.')}")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    else:
        st.warning("Please enter some ticket text to classify.")

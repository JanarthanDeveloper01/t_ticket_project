import streamlit as st
import joblib

# Load trained models
pipe = joblib.load("it_ticket_classifier.pkl")        # your trained classifier pipeline
tfidf_vect = joblib.load("tfidf_vectorizer.pkl")      # TF-IDF vectorizer

# Define solution suggestions for each category
solution_dict = {
    "Technical Support": "Please restart your device and check your network connection. If issue persists, contact IT support.",
    "Product Support": "Try reinstalling or updating the product. For persistent issues, reach out to product support.",
    "Customer Service": "Our customer service team will contact you shortly to assist with your inquiry.",
    "IT Support": "Please provide system details and error messages. IT support will guide you through troubleshooting.",
    "Billing and Payments": "Check your billing details in your account. For discrepancies, contact billing department.",
    "Returns and Exchanges": "Fill out the returns/exchange form and follow the instructions provided.",
    "Service Outages and Maintenance": "Check our status page for ongoing maintenance or outages.",
    "Sales and Pre-Sales": "Our sales team will get back to you with pricing and product details.",
    "Human Resources": "Please reach out to HR via the internal HR portal for assistance.",
    "General Inquiry": "Please provide more details so we can route your inquiry appropriately."
}

# Streamlit app
st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")

ticket_text = st.text_area("Enter IT Ticket Text:")

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.error("Please enter a ticket text to predict!")
    else:
        try:
            # Transform input text using TF-IDF
            X_input = tfidf_vect.transform([ticket_text])
            
            # Make prediction
            prediction = pipe.predict(X_input)[0]
            
            # Get solution from dictionary
            solution = solution_dict.get(prediction, "No solution available for this category.")
            
            # Display results
            st.success(f"Predicted Category: {prediction}")
            st.info(f"Suggested Solution: {solution}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

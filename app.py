import streamlit as st
import joblib

# Load pipeline (TF-IDF + classifier)
pipe = joblib.load("it_ticket_classifier.pkl")

# Define solutions
solution_dict = {
    "Technical Support": "Restart device or check network.",
    "Product Support": "Reinstall/update product or contact support.",
    "Customer Service": "Customer service will reach out.",
    "IT Support": "Provide system details for troubleshooting.",
    "Billing and Payments": "Check billing account or contact billing.",
    "Returns and Exchanges": "Fill returns/exchange form.",
    "Service Outages and Maintenance": "Check status page for updates.",
    "Sales and Pre-Sales": "Sales team will respond with details.",
    "Human Resources": "Contact HR via internal portal.",
    "General Inquiry": "Provide more details for routing."
}

st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")

ticket_text = st.text_area("Enter IT Ticket Text:")

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.error("Please enter a ticket text to predict!")
    else:
        try:
            # **Pass raw text directly to pipeline**
            prediction = pipe.predict([ticket_text])[0]

            # Get solution
            solution = solution_dict.get(prediction, "No solution available.")

            st.success(f"Predicted Category: {prediction}")
            st.info(f"Suggested Solution: {solution}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

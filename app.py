import streamlit as st
import joblib

# Title
st.title("AI-Based IT Ticket Classifier & Resolution Suggestion")
st.write("Enter the IT ticket text below and click Predict to get the category.")

# Load the trained pipeline
try:
    pipe = joblib.load("it_ticket_classifier.pkl")  # This should be your Pipeline (TF-IDF + LogisticRegression)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if model fails to load

# Text input for the ticket
ticket_text = st.text_area("Enter IT Ticket Text:")

# Predict button
if st.button("Predict Category"):
    if ticket_text.strip() == "":
        st.warning("Please enter the ticket text to predict.")
    else:
        try:
            # Predict using the pipeline directly (no separate TF-IDF transform needed)
            prediction = pipe.predict([ticket_text])[0]
            st.success(f"Predicted Category: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Optional: Add instructions or info
st.write("""
**Instructions:**
- Enter the full text of the IT ticket in the box above.
- Click 'Predict Category' to see the predicted category.
- Make sure the trained model (`it_ticket_classifier.pkl`) is in the same folder as this app.py.
""")

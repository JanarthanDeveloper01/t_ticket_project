import streamlit as st
import joblib
import re

# Load trained pipeline
pipe = joblib.load("it_ticket_classifier.pkl")

# Define keywords for each category
keywords_dict = {
    "Technical Support": ["error", "crash", "bug", "issue", "restart", "network", "login"],
    "Product Support": ["install", "update", "software", "feature", "problem", "version", "configuration"],
    "Customer Service": ["complaint", "request", "feedback", "support", "assistance", "help"],
    "IT Support": ["server", "network", "permission", "access", "IT", "system"],
    "Billing and Payments": ["payment", "invoice", "billing", "charge", "refund", "transaction"],
    "Returns and Exchanges": ["return", "exchange", "warranty", "replacement", "policy"],
    "Service Outages and Maintenance": ["outage", "maintenance", "down", "update", "issue", "service"],
    "Sales and Pre-Sales": ["price", "quote", "demo", "product", "pre-sale", "offer", "availability"],
    "Human Resources": ["leave", "payroll", "policy", "HR", "benefits", "recruitment"],
    "General Inquiry": ["question", "information", "clarification", "general", "inquiry"]
}

# Define detailed solutions for each category
solution_dict = {
    "Technical Support": (
        "1. Restart the device and check if the issue persists.\n"
        "2. Verify network connectivity and VPN settings.\n"
        "3. Update drivers and software to the latest version.\n"
        "4. Contact the technical support team if the problem continues."
    ),
    "Product Support": (
        "1. Ensure the product is updated to the latest version.\n"
        "2. Reinstall the software or application.\n"
        "3. Check official product documentation for troubleshooting steps.\n"
        "4. Contact product support with detailed logs or screenshots."
    ),
    "Customer Service": (
        "1. Review the customer’s query carefully.\n"
        "2. Respond promptly with a polite and helpful message.\n"
        "3. Escalate to relevant teams if required.\n"
        "4. Keep track of the query until fully resolved."
    ),
    "IT Support": (
        "1. Gather system information (OS, version, error logs).\n"
        "2. Verify permissions and configurations.\n"
        "3. Check network or server issues.\n"
        "4. Provide step-by-step guidance to the user or schedule a remote session."
    ),
    "Billing and Payments": (
        "1. Check the billing account and transaction history.\n"
        "2. Verify if payments are pending or failed.\n"
        "3. Provide instructions to resolve payment issues.\n"
        "4. Contact the billing department for disputed charges."
    ),
    "Returns and Exchanges": (
        "1. Verify purchase and warranty details.\n"
        "2. Provide the return/exchange form and instructions.\n"
        "3. Schedule pick-up or drop-off if needed.\n"
        "4. Update the customer on status until completion."
    ),
    "Service Outages and Maintenance": (
        "1. Check if the reported service is under scheduled maintenance.\n"
        "2. Verify outage reports from the status page.\n"
        "3. Inform affected users about expected resolution time.\n"
        "4. Escalate to the maintenance team if not yet resolved."
    ),
    "Sales and Pre-Sales": (
        "1. Understand the customer’s requirements.\n"
        "2. Provide product/service options with features and pricing.\n"
        "3. Share demos or trial links if applicable.\n"
        "4. Follow up for feedback or next steps in the sales process."
    ),
    "Human Resources": (
        "1. Identify the HR-related issue (leave, payroll, policy, etc.).\n"
        "2. Provide instructions or forms as needed.\n"
        "3. Forward to the HR team for complex cases.\n"
        "4. Ensure timely follow-up and resolution."
    ),
    "General Inquiry": (
        "1. Request more details about the inquiry.\n"
        "2. Route the ticket to the appropriate department.\n"
        "3. Provide a standard response if applicable.\n"
        "4. Track until the inquiry is resolved."
    )
}

# Streamlit UI
st.title("AI-Based IT Ticket Classifier & Smart Suggestions")

ticket_text = st.text_area("Enter IT Ticket Text:")

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.error("Please enter a ticket text to predict!")
    else:
        try:
            # Predict category
            prediction = pipe.predict([ticket_text])[0]
            solution = solution_dict.get(prediction, "No solution available.")

            st.success(f"Predicted Category: {prediction}")
            st.info(f"Suggested Solution:\n{solution}")

            # Highlight keywords
            keywords = keywords_dict.get(prediction, [])
            highlighted_text = ticket_text
            for kw in keywords:
                highlighted_text = re.sub(f"({kw})", r"<mark>\1</mark>", highlighted_text, flags=re.IGNORECASE)

            st.markdown("### Ticket Text with Highlighted Keywords")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

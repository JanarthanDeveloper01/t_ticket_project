import streamlit as st
import joblib
import re
from collections import defaultdict

# Load trained pipeline
pipe = joblib.load("it_ticket_classifier.pkl")

# Keywords for each category
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

# Detailed solutions for each category
solution_dict = {
    "Technical Support": "Restart device, check network, update drivers/software, contact technical support if unresolved.",
    "Product Support": "Update product, reinstall software, check documentation, contact product support with logs/screenshots.",
    "Customer Service": "Review query, respond politely, escalate if needed, track until resolved.",
    "IT Support": "Gather system info, verify permissions, check server/network issues, provide step-by-step guidance.",
    "Billing and Payments": "Check account and transaction history, verify payments, provide resolution instructions, contact billing team if needed.",
    "Returns and Exchanges": "Verify purchase/warranty, provide forms, schedule pick-up/drop-off, update customer on status.",
    "Service Outages and Maintenance": "Check maintenance schedule, verify outages, inform users of resolution time, escalate if necessary.",
    "Sales and Pre-Sales": "Understand requirements, provide product options and pricing, share demos, follow up for next steps.",
    "Human Resources": "Identify HR issue, provide instructions/forms, forward complex cases to HR team, ensure timely follow-up.",
    "General Inquiry": "Request more details, route to appropriate department, provide standard response if applicable, track until resolved."
}

st.title("AI-Based IT Ticket Classifier & Smart Suggestions")

ticket_text = st.text_area("Enter IT Ticket Text:")

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.error("Please enter a ticket text to predict!")
    else:
        try:
            # Predict primary category
            prediction = pipe.predict([ticket_text])[0]

            # Suggest top categories based on keyword matches
            keyword_matches = defaultdict(int)
            for cat, kws in keywords_dict.items():
                for kw in kws:
                    if re.search(rf"\b{kw}\b", ticket_text, re.IGNORECASE):
                        keyword_matches[cat] += 1

            # Sort categories by match count
            sorted_categories = sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True)
            top_categories = [cat for cat, count in sorted_categories if count > 0]

            st.success(f"Predicted Category: {prediction}")
            st.info(f"Suggested Solution:\n{solution_dict.get(prediction, 'No solution available.')}")

            if top_categories:
                st.markdown("### Other Potential Categories Based on Keywords")
                for cat in top_categories:
                    st.markdown(f"**{cat}**: {solution_dict.get(cat, 'No solution available.')}")

            # Highlight keywords in the ticket
            highlighted_text = ticket_text
            for kw in keywords_dict.get(prediction, []):
                highlighted_text = re.sub(f"({kw})", r"<mark>\1</mark>", highlighted_text, flags=re.IGNORECASE)

            st.markdown("### Ticket Text with Highlighted Keywords")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

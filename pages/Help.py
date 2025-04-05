import streamlit as st
import uuid
import json
import re
import requests
import random
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Help & Support", layout="centered")

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR THE HELP PAGE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      body {
          background: #1c1c1c;
          font-family: 'Helvetica Neue', sans-serif;
          color: #f0f0f0;
      }
      .stApp {
          background-color: #1c1c1c;
      }
      .main-title {
          text-align: center;
          font-size: 3.5rem;
          font-weight: 900;
          color: #FF6F61;
          margin-bottom: 1.5rem;
      }
      .section-header {
          font-size: 2.2rem;
          font-weight: 700;
          margin-top: 2rem;
          margin-bottom: 1rem;
          color: #FFB74D;
      }
      .faq-item {
          margin-bottom: 1.5rem;
          padding: 1rem;
          background: #333333;
          border-left: 5px solid #FF6F61;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      }
      .faq-question {
          font-weight: bold;
          margin-bottom: 0.5rem;
          color: #FFB74D;
      }
      .faq-answer {
          margin-left: 1rem;
          color: #d0d0d0;
      }
      .contact-form {
          background: #2a2a2a;
          padding: 2rem;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.3);
          margin-top: 2rem;
      }
      .footer {
          text-align: center;
          font-size: 0.9rem;
          color: #a0a0a0;
          margin-top: 3rem;
          border-top: 1px solid #444;
          padding-top: 1rem;
      }
      .icon {
          margin-right: 0.5rem;
          color: #FF6F61;
      }
      .tab-header {
          text-align: center;
          font-size: 2rem;
          margin-bottom: 1rem;
          color: #FFB74D;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Font Awesome for icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
integrity="sha512-dBwEXZg+0ItZ3Y11rXwSTBPH3IyzYFHDxvxxrEC0Cjk0n6X8hEP1p7eKcK9F9D6b2ZLKl91r6Wip4v1qzK+qUg==" 
crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE TITLE & HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'><i class='fas fa-life-ring icon'></i>Help & Support</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTION: MongoDB Client
# -----------------------------------------------------------------------------
def get_mongo_client():
    connection_string = st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017")
    return MongoClient(connection_string)

def store_support_ticket(ticket_data):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        db["support_tickets"].insert_one(ticket_data)
        return True
    except Exception as e:
        st.error(f"Error storing support ticket: {e}")
        return False
    finally:
        client.close()

# -----------------------------------------------------------------------------
# CREATE TABS FOR HELP CONTENT
# -----------------------------------------------------------------------------
tabs = st.tabs(["FAQ", "Tutorial", "Troubleshooting", "Contact Support"])

# -----------------------------------------------------------------------------
# TAB 1: FAQ
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='section-header'>Frequently Asked Questions</div>", unsafe_allow_html=True)
    faqs = [
        {
            "question": "How do I register for an account?",
            "answer": "Click on the 'Register' button on the login page and fill in the required fields, including your unique CIN number (e.g., D922986)."
        },
        {
            "question": "How do I submit an idea?",
            "answer": "After logging in, navigate to the 'Submit Idea' tab. Fill in your idea or comment in the provided form. You can optionally link your idea to a municipal project."
        },
        {
            "question": "What happens to my idea after submission?",
            "answer": "Your idea will be marked as 'pending' until a moderator reviews it. Once approved, it will appear in the public Idea Feed."
        },
        {
            "question": "How is my data used?",
            "answer": "We use your data to improve our services and personalize your experience. For more details, please refer to our Privacy Policy."
        },
        {
            "question": "Who can I contact for further support?",
            "answer": "If you need additional help, please use the Contact Support form on this page or email support@civiccatalyst.org."
        }
    ]
    for faq in faqs:
        st.markdown(f"<div class='faq-item'><div class='faq-question'>Q: {faq['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='faq-answer'>A: {faq['answer']}</div></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: Tutorial
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='section-header'>Tutorial</div>", unsafe_allow_html=True)
    st.write("Welcome to Civic Catalyst! Here’s how to get started:")
    st.markdown("""
    1. **Registration & Login:** Register using your CIN number and log in with your credentials.
    2. **Submit Your Ideas:** Navigate to the 'Submit Idea' tab to share your civic ideas.
    3. **Browse Ideas:** View approved ideas in the 'Idea Feed' to see what others have submitted.
    4. **Project Linkage:** Optionally, link your idea to an existing municipal project.
    5. **Support:** Use this Help page or contact support for further assistance.
    """)
    st.image("https://via.placeholder.com/800x400?text=Tutorial+Screenshot", caption="Tutorial Screenshot", use_column_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Troubleshooting
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='section-header'>Troubleshooting</div>", unsafe_allow_html=True)
    st.write("Here are some common issues and their solutions:")
    troubleshooting = [
        {
            "issue": "I cannot log in.",
            "solution": "Ensure you are using the correct username and password. If you forgot your password, use the 'Forgot Password' link."
        },
        {
            "issue": "My idea doesn't appear in the feed.",
            "solution": "Ideas need to be approved by moderators. Please wait up to 24 hours or contact support if the issue persists."
        },
        {
            "issue": "The platform is slow.",
            "solution": "This could be due to high server load. Try refreshing the page or contact support for further assistance."
        }
    ]
    for item in troubleshooting:
        st.markdown(f"<b>Issue:</b> {item['issue']}", unsafe_allow_html=True)
        st.markdown(f"<b>Solution:</b> {item['solution']}", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: Contact Support
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='section-header'>Contact Support</div>", unsafe_allow_html=True)
    st.write("If you need further assistance, please fill out the form below. Our support team will respond within 48 hours.")
    
    with st.form("support_form", clear_on_submit=True):
        support_name = st.text_input("Your Name", help="Enter your full name.")
        support_email = st.text_input("Your Email", help="Enter your email address.")
        support_issue = st.text_area("Describe Your Issue", height=150, help="Provide as much detail as possible.")
        support_submit = st.form_submit_button("Submit Ticket")
    
    if support_submit:
        if not support_name.strip() or not support_email.strip() or not support_issue.strip():
            st.error("All fields are required.")
        else:
            ticket_id = str(uuid.uuid4())
            ticket_data = {
                "ticket_id": ticket_id,
                "name": support_name.strip(),
                "email": support_email.strip(),
                "issue": support_issue.strip(),
                "timestamp": datetime.utcnow()
            }
            if store_support_ticket(ticket_data):
                st.success(f"Your support ticket has been submitted successfully! Ticket ID: {ticket_id}")
            else:
                st.error("There was an error submitting your ticket. Please try again later.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div class='footer'>© 2025 Civic Catalyst. All rights reserved.</div>", unsafe_allow_html=True)

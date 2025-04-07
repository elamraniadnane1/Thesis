import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import re
import uuid
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# SETUP & CONFIGURATION
# -----------------------------------------------------------------------------

# Set your OpenAI API key (assumed to be stored in st.secrets)
openai.api_key = st.secrets["openai"]["api_key"]

# Set page configuration
#st.set_page_config(page_title="Contact Us", layout="centered")

# Load Font Awesome for icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
integrity="sha512-dBwEXZg+0ItZ3Y11rXwSTBPH3IyzYFHDxvxxrEC0Cjk0n6X8hEP1p7eKcK9F9D6b2ZLKl91r6Wip4v1qzK+qUg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)

# Custom CSS for the Contact Us page
st.markdown(
    """
    <style>
      body {
          background: #f4f4f4;
          font-family: 'Roboto', sans-serif;
          color: #333;
      }
      .contact-title {
          text-align: center;
          font-size: 3rem;
          font-weight: 800;
          color: #FF416C;
          margin-bottom: 1rem;
      }
      .contact-info {
          text-align: center;
          font-size: 1.2rem;
          margin-bottom: 2rem;
      }
      .contact-info i {
          color: #FF416C;
          margin-right: 0.5rem;
      }
      .contact-form {
          background: #FFFFFF;
          padding: 2rem;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_mongo_client():
    # Retrieve the connection string from st.secrets (or use a default for local testing)
    connection_string = st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017")
    return MongoClient(connection_string)

def store_contact_message(data):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        messages_collection = db["contact_messages"]
        messages_collection.insert_one(data)
        return True
    except Exception as e:
        st.error(f"Error storing message: {e}")
        return False
    finally:
        client.close()

def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

def is_valid_phone(phone):
    # Allow digits, spaces, dashes, and parentheses.
    pattern = r"^[\d\s\-\(\)]+$"
    return re.match(pattern, phone)

# -----------------------------------------------------------------------------
# STREAMLIT UI: CONTACT US PAGE
# -----------------------------------------------------------------------------

st.markdown("<div class='contact-title'>Contact Us</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='contact-info'>
      <i class='fas fa-phone'></i> +212668188054 &nbsp;&nbsp;
      <i class='fas fa-envelope'></i> contact@civiccatalyst.org
    </div>
    """,
    unsafe_allow_html=True
)

st.write("We'd love to hear from you! Please fill out the form below and we'll get back to you as soon as possible.")

with st.form("contact_form", clear_on_submit=True):
    name = st.text_input("Your Name", help="Enter your full name.")
    email = st.text_input("Your Email", help="Enter a valid email address.")
    phone = st.text_input("Your Phone (Optional)", help="Enter your phone number (digits, spaces, dashes allowed).")
    subject = st.text_input("Subject", help="Enter the subject of your message.")
    message = st.text_area("Message", height=200, help="Type your message here. Please be as detailed as possible.")
    submit = st.form_submit_button("Send Message")

if submit:
    if not name.strip():
        st.error("Please enter your name.")
    elif not email.strip() or not is_valid_email(email.strip()):
        st.error("Please enter a valid email address.")
    elif phone.strip() and not is_valid_phone(phone.strip()):
        st.error("Please enter a valid phone number.")
    elif not subject.strip():
        st.error("Please enter a subject for your message.")
    elif not message.strip():
        st.error("Please enter your message.")
    else:
        # Generate a unique contact_id using UUID
        contact_id = str(uuid.uuid4())
        data = {
            "contact_id": contact_id,
            "name": name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "subject": subject.strip(),
            "message": message.strip(),
            "timestamp": datetime.utcnow()
        }
        if store_contact_message(data):
            st.success("Your message has been sent successfully! Thank you for reaching out to us.")
            st.info("We will respond to your inquiry within 48 hours.")
        else:
            st.error("There was an error sending your message. Please try again later.")

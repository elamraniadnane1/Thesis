import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
#st.set_page_config(page_title="Your Privacy", layout="centered")

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR THE PRIVACY PAGE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      body {
          background: #121212;
          font-family: 'Roboto', sans-serif;
          color: #EAEAEA;
      }
      .stApp {
          background-color: #121212;
      }
      .main-title {
          text-align: center;
          font-size: 3.5rem;
          font-weight: 800;
          color: #FF6F61;
          margin-bottom: 1rem;
      }
      .section-header {
          font-size: 2.2rem;
          font-weight: 700;
          margin-top: 2rem;
          margin-bottom: 1rem;
          color: #FF4B2B;
      }
      .policy-text {
          font-size: 1.1rem;
          line-height: 1.8;
          margin: 2rem auto;
          max-width: 800px;
          background: #333333;
          padding: 2rem;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8);
          text-align: justify;
      }
      .footer {
          text-align: center;
          font-size: 0.9rem;
          color: #999999;
          margin-top: 3rem;
          border-top: 1px solid #444444;
          padding-top: 1rem;
      }
      .download-btn {
          margin: 2rem auto;
          display: block;
      }
      /* Icon styling */
      .icon {
          margin-right: 0.5rem;
          color: #FF6F61;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Font Awesome for icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
integrity="sha512-dBwEXZg+0ItZ3Y11rXwSTBPH3IyzYFHDxvxxrEC0Cjk0n6X8hEP1p7eKcK9F9D6b2ZLKl91r6Wip4v1qzK+qUg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE TITLE & HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'><i class='fas fa-user-shield icon'></i>Your Privacy Policy</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PRIVACY POLICY CONTENT
# -----------------------------------------------------------------------------
policy_text = """
_Last updated: October 2025_

## Introduction
At Civic Catalyst, your privacy is our priority. This Privacy Policy explains how we collect, use, and safeguard your information when you use our platform. We are committed to protecting your personal data and ensuring transparency in our operations.

## Data Collection
We may collect information such as:
- **Personal Information:** Your name, email address, and contact details provided during registration.
- **User Content:** Ideas, comments, and other submissions made on the platform.
- **Usage Data:** Information about how you interact with our services, such as IP addresses and browser type.
- **Cookies and Tracking Technologies:** To enhance your user experience, we may use cookies.

## Data Usage
The information we collect is used to:
- Provide and improve our services.
- Personalize your experience.
- Communicate with you about updates and offers.
- Ensure the security and integrity of our platform.

## Data Sharing & Disclosure
We do not sell your personal data. We may share your information with:
- **Service Providers:** Trusted third parties who help us operate our platform.
- **Legal Authorities:** When required by law or to protect our rights.
- **Business Partners:** In cases where you have explicitly consented to such sharing.

## Security Measures
We use industry-standard security practices to protect your data from unauthorized access, alteration, or disclosure. This includes encryption, access controls, and regular security reviews.

## Your Rights
You have the right to:
- Access your personal data.
- Correct inaccuracies in your information.
- Request deletion of your data.
- Object to certain processing activities.
- Withdraw consent at any time (where applicable).

## Cookies & Tracking
We use cookies to improve your experience. You can adjust your cookie preferences in your browser settings. However, disabling cookies may affect the functionality of our platform.

## Changes to This Policy
We may update our Privacy Policy periodically. We will notify you of any significant changes by updating the "Last updated" date on this page.

## Contact Us
If you have any questions about this Privacy Policy or our data practices, please contact our Privacy Officer:
- **Email:** privacy@civiccatalyst.org
- **Address:** AUI, Ifrane

Thank you for trusting Civic Catalyst with your data.
"""

# Display the policy text as formatted markdown within a styled div
st.markdown("<div class='policy-text'>" + policy_text + "</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DOWNLOAD PRIVACY POLICY BUTTON
# -----------------------------------------------------------------------------
def generate_privacy_file():
    filename = f"Privacy_Policy_{datetime.utcnow().strftime('%Y%m%d')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(policy_text)
    with open(filename, "r", encoding="utf-8") as f:
        file_content = f.read()
    return file_content, filename

privacy_content, privacy_filename = generate_privacy_file()

st.download_button(
    label="Download Privacy Policy as TXT",
    data=privacy_content,
    file_name=privacy_filename,
    mime="text/plain",
    key="download_privacy_policy"
)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div class='footer'>© 2025 Civic Catalyst. All rights reserved.</div>", unsafe_allow_html=True)

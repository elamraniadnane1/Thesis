import streamlit as st
import openai
import os
import uuid
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      body {
          background: #f0f2f6;
          font-family: 'Roboto', sans-serif;
          color: #333;
          margin: 0;
          padding: 0;
      }
      .main-title {
          text-align: center;
          font-size: 3.5rem;
          font-weight: 900;
          color: #FF416C;
          margin: 2rem 0 1.5rem 0;
      }
      .partner-card {
          background: #fff;
          border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          padding: 1rem;
          margin-bottom: 1.5rem;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
      }
      .partner-card:hover {
          transform: scale(1.03);
          box-shadow: 0 6px 16px rgba(0,0,0,0.15);
      }
      .partner-logo {
          max-width: 120px;
          max-height: 120px;
          margin-bottom: 0.5rem;
      }
      .partner-name {
          font-size: 1.8rem;
          font-weight: 700;
          color: #FF4B2B;
          margin-bottom: 0.5rem;
      }
      .partner-desc {
          font-size: 1.1rem;
          margin-bottom: 0.5rem;
          line-height: 1.5;
      }
      .partner-link {
          font-size: 1rem;
          color: #0077cc;
          text-decoration: none;
          font-weight: 500;
      }
      .partner-link:hover {
          text-decoration: underline;
      }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<div class='main-title'>Our Partners</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR MONGODB OPERATIONS
# -----------------------------------------------------------------------------
def get_mongo_client():
    # Retrieves connection string from st.secrets, defaulting to localhost.
    connection_string = st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017")
    return MongoClient(connection_string)

def get_partners():
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        partners_collection = db["partners"]
        partners = list(partners_collection.find({}, {"_id": 0}))
        return partners
    except Exception as e:
        st.error(f"Error fetching partners: {e}")
        return []
    finally:
        client.close()

def add_partner(data: dict):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        partners_collection = db["partners"]
        partners_collection.insert_one(data)
        return True
    except Exception as e:
        st.error(f"Error adding partner: {e}")
        return False
    finally:
        client.close()

def delete_partner(partner_id: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        result = db["partners"].delete_one({"partner_id": partner_id})
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting partner: {e}")
        return False
    finally:
        client.close()

def update_partner(partner_id: str, updated_data: dict):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        result = db["partners"].update_one({"partner_id": partner_id}, {"$set": updated_data})
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error updating partner: {e}")
        return False
    finally:
        client.close()

# -----------------------------------------------------------------------------
# REMACTO PARTNER: ENSURE IT EXISTS
# -----------------------------------------------------------------------------
def ensure_remacto_partner():
    """Check if REMACTO exists; if not, add it with a detailed explanation in English."""
    partners = get_partners()
    for partner in partners:
        if partner.get("name", "").strip().lower() == "remacto":
            return  # Already exists
    remacto_description = (
        "Detailed explanation of how open government initiatives work to strengthen citizen participation:\n\n"
        "The Moroccan Network of Open Territorial Collectivities (REMACTO) was launched to encourage and support the "
        "implementation of openness principles within Moroccan local governments: transparency, accountability, access to information, "
        "citizen participation, and digitalization. It relies on an organizational, legal, and methodological framework that fosters "
        "the co-creation of open government programs, citizen engagement, and public consultation."
    )
    new_partner = {
         "partner_id": str(uuid.uuid4()),
         "name": "REMACTO",
         "logo_url": "",  # Add a default logo URL if available.
         "description": remacto_description,
         "website": "http://collectivites-territoriales.gov.ma/en/remacto",  # Example English website URL.
         "contact_info": "",
         "timestamp": datetime.utcnow()
    }
    add_partner(new_partner)

# Automatically ensure REMACTO is present
ensure_remacto_partner()

# -----------------------------------------------------------------------------
# DISPLAY PARTNERS
# -----------------------------------------------------------------------------
partners = get_partners()
if partners:
    st.subheader("Our Partner Organizations")
    # Display partners in a card-like layout using columns
    cols = st.columns(3)
    for idx, partner in enumerate(partners):
        col = cols[idx % 3]
        with col:
            with st.container():
                st.markdown("<div class='partner-card'>", unsafe_allow_html=True)
                # Display logo if available
                if partner.get("logo_url"):
                    st.image(partner["logo_url"], use_container_width=True, output_format="auto", caption=partner.get("name", "Partner Logo"), width=120)
                st.markdown(f"<div class='partner-name'>{partner.get('name', 'Unnamed Partner')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='partner-desc'>{partner.get('description', 'No description provided.')}</div>", unsafe_allow_html=True)
                if partner.get("website"):
                    st.markdown(f"<a href='{partner.get('website')}' target='_blank' class='partner-link'>Visit Website</a>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No partner organizations found.")

# -----------------------------------------------------------------------------
# ADMIN MANAGEMENT FUNCTIONS (Only visible if user is admin)
# -----------------------------------------------------------------------------
if st.session_state.get("role") == "admin":
    st.markdown("---")
    st.subheader("Admin Partner Management")

    # Form to add a new partner
    with st.form("add_partner_form", clear_on_submit=True):
        partner_name = st.text_input("Partner Name", help="Enter the full name of the partner organization.")
        logo_url = st.text_input("Logo URL", help="Enter the URL of the partner's logo (optional).")
        description = st.text_area("Description", help="Enter a brief description of the partner.")
        website = st.text_input("Website", help="Enter the partner's website URL (optional).")
        contact_info = st.text_input("Contact Info", help="Enter contact details (e.g., email or phone, optional).")
        add_partner_btn = st.form_submit_button("Add Partner")
    
    if add_partner_btn:
        if not partner_name.strip():
            st.error("Partner name is required.")
        else:
            new_partner = {
                "partner_id": str(uuid.uuid4()),
                "name": partner_name.strip(),
                "logo_url": logo_url.strip(),
                "description": description.strip(),
                "website": website.strip(),
                "contact_info": contact_info.strip(),
                "timestamp": datetime.utcnow()
            }
            if add_partner(new_partner):
                st.success(f"Partner '{partner_name}' added successfully!")
                st.rerun()
            else:
                st.error("Error adding partner.")

    # Option to delete a partner
    st.markdown("### Delete a Partner")
    if partners:
        partner_options = {f"{p.get('name', 'Unnamed')} (ID: {p.get('partner_id')})": p.get("partner_id") for p in partners}
        selected_partner_id = st.selectbox("Select Partner to Delete", list(partner_options.keys()))
        if st.button("Delete Partner"):
            pid = partner_options[selected_partner_id]
            if delete_partner(pid):
                st.success("Partner deleted successfully!")
                st.rerun()
            else:
                st.error("Error deleting partner.")

    # Option to update a partner (here, update only the description)
    st.markdown("### Update Partner Description")
    if partners:
        update_options = {f"{p.get('name', 'Unnamed')} (ID: {p.get('partner_id')})": p.get("partner_id") for p in partners}
        selected_update_partner = st.selectbox("Select Partner to Update", list(update_options.keys()), key="update_partner")
        new_description = st.text_area("New Description", placeholder="Enter the updated description here...")
        if st.button("Update Description"):
            pid = update_options[selected_update_partner]
            if update_partner(pid, {"description": new_description.strip()}):
                st.success("Partner description updated successfully!")
                st.rerun()
            else:
                st.error("Error updating partner description.")

# -----------------------------------------------------------------------------
# END OF PAGE
# -----------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>© 2025 Civic Catalyst. All rights reserved.</div>", unsafe_allow_html=True)

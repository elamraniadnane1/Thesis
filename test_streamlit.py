import streamlit as st

# Attempt to use st.experimental_rerun
try:
    st.experimental_rerun()
except AttributeError:
    # Fallback to st.rerun if available
    st.rerun()

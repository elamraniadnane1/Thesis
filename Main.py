import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def main():
    st.title("Civic Catalyst")

    # Check for a stored JWT token in session state
    if "jwt_token" not in st.session_state or not st.session_state["jwt_token"]:
        # If no token, the user is not logged in
        st.write("Redirecting to Login...")
        st.switch_page("pages\\Login.py")  # Name of your Login page
    else:
        # User is logged in; show main content here
        st.success(f"Welcome back, {st.session_state.get('username', 'User')}!")
        st.write("This is the main content of the Civic Catalyst application.")

if __name__ == "__main__":
    main()

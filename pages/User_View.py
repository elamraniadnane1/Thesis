import streamlit as st
import pandas as pd
from auth_system import require_auth, verify_jwt_token

@require_auth
def main():
    st.title("👤 User View (My Dashboard)")
    st.write(
        """
        Welcome to your personal dashboard. Here you can find user-specific data,
        recent interactions, and any other information you want to display based
        on who is logged in.
        """
    )

    # Retrieve the current JWT token from session_state
    token = st.session_state.get('jwt_token', None)
    
    if token:
        # Decode the token to get username/role
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")

            # Example: Display some user-specific data
            st.subheader("Your Activity Overview")
            # (You could retrieve user-specific data from a CSV, DB, or your main DataFrame)
            # For demonstration, we'll create a dummy DataFrame:
            data = {
                'Action': ['Logged In', 'Commented', 'Upvoted', 'Downvoted'],
                'Date': ['2023-03-01', '2023-03-02', '2023-03-05', '2023-03-07']
            }
            user_df = pd.DataFrame(data)
            st.dataframe(user_df)

            # Add any additional user-only tools or analytics here

        else:
            st.warning("Token is invalid or expired. Please log in again.")
    else:
        st.info("No token found in session. Please go back and log in.")

if __name__ == "__main__":
    main()

import jwt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

# 1) Constants for JWT
JWT_SECRET = 'your-secret-key'  # In production, use a secure secret key
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DELTA = timedelta(hours=24)

def encode_jwt(payload: dict, secret: str, algorithm: str) -> str:
    """Encode JWT token using PyJWT."""
    try:
        token = jwt.encode(payload, secret, algorithm=algorithm)
        # PyJWT returns a string in newer versions, but bytes in older versions
        if isinstance(token, bytes):
            return token.decode('utf-8')
        return token
    except Exception as e:
        st.error(f"JWT encoding error: {str(e)}")
        return None

def init_users_file():
    """Initialize users.csv if it doesn't exist."""
    try:
        users_file = Path('users.csv')
        if not users_file.exists():
            df = pd.DataFrame(columns=['username', 'password_hash', 'role'])
            df.to_csv(users_file, index=False)
            # Create default admin user
            create_user('admin', 'admin123', 'admin')
    except Exception as e:
        st.error(f"Error initializing users file: {str(e)}")

def hash_password(password: str) -> str:
    """Create a SHA-256 hash of the password."""
    try:
        return hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        st.error(f"Error hashing password: {str(e)}")
        return None

def create_user(username: str, password: str, role: str = 'user') -> bool:
    """Create a new user in the CSV file."""
    try:
        users_df = pd.read_csv('users.csv')
        
        # Check if username already exists
        if username in users_df['username'].values:
            return False
        
        password_hash = hash_password(password)
        if password_hash is None:
            return False
            
        # Add new user
        new_user = pd.DataFrame({
            'username': [username],
            'password_hash': [password_hash],
            'role': [role]
        })
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv('users.csv', index=False)
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def verify_user(username: str, password: str) -> tuple:
    """Verify user credentials and return (success, role)."""
    try:
        users_df = pd.read_csv('users.csv')
        user_row = users_df[users_df['username'] == username]
        
        if user_row.empty:
            return False, None
        
        password_hash = hash_password(password)
        if password_hash is None:
            return False, None
            
        stored_hash = user_row.iloc[0]['password_hash']
        if password_hash == stored_hash:
            return True, user_row.iloc[0]['role']
        return False, None
    except Exception as e:
        st.error(f"Error verifying user: {str(e)}")
        return False, None

def create_jwt_token(username: str, role: str) -> str:
    """Create a JWT token for the user."""
    try:
        expiration = datetime.utcnow() + JWT_EXPIRATION_DELTA
        payload = {
            'username': username,
            'role': role,
            'exp': expiration
        }
        token = encode_jwt(payload, JWT_SECRET, JWT_ALGORITHM)
        return token
    except Exception as e:
        st.error(f"Error creating JWT token: {str(e)}")
        return None

def verify_jwt_token(token: str) -> tuple:
    """Verify a JWT token and return (is_valid, username, role)."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return True, payload['username'], payload['role']
    except jwt.ExpiredSignatureError:
        return False, None, None
    except jwt.InvalidTokenError:
        return False, None, None
    except Exception as e:
        st.error(f"Error verifying JWT token: {str(e)}")
        return False, None, None

def init_auth():
    """Initialize authentication system."""
    try:
        init_users_file()
        if 'jwt_token' not in st.session_state:
            st.session_state.jwt_token = None
    except Exception as e:
        st.error(f"Error initializing authentication: {str(e)}")

def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        try:
            if not st.session_state.get('jwt_token'):
                st.error("Please login to access this page")
                st.stop()
            is_valid, username, role = verify_jwt_token(st.session_state.jwt_token)
            if not is_valid:
                st.session_state.jwt_token = None
                st.error("Session expired. Please login again")
                st.stop()
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.stop()
    return wrapper

################################################################################
# FANCY CIVIC CATALYST LOGIN PAGE
################################################################################
def login_page():
    """
    Display a fancy login interface for Civic Catalyst with modern design,
    gradient backgrounds, and streamlined forms.
    """

    # 1) Inject custom CSS for a fancy look
    st.markdown(
        """
        <style>
        /* Custom CSS for a fancy, modern login page */

        /* Body background - subtle gradient */
        body {
            background: linear-gradient(to right, #F0F4F8, #D9E4F5);
            font-family: 'Poppins', sans-serif;
        }

        /* Center the main container */
        .main-login-container {
            max-width: 500px;
            margin: 5% auto;
            background: #ffffffd9; /* White with slight transparency */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 1rem;
            padding: 2rem;
        }

        /* Title styling */
        .login-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00b09b, #96c93d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
        }

        /* Subtle text shadows for labels */
        label {
            text-shadow: 0 1px 1px rgba(0,0,0,0.06);
            font-weight: 600 !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #3AAFA9 0%, #2B7A78 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 1rem;
            height: 3rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #2B7A78 0%, #3AAFA9 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(43,122,120,0.3);
        }
        .stButton>button:active {
            transform: scale(0.98);
        }

        /* Input fields */
        input[type="text"], input[type="password"] {
            border: 1px solid #ccc !important;
            padding: 0.6rem !important;
            border-radius: 0.4rem;
            font-size: 1rem !important;
        }

        /* Form container */
        .login-form, .registration-form {
            margin-bottom: 2rem;
        }

        /* Expanders for registration */
        .st-expanderHeader {
            font-weight: 700;
            font-size: 1.1rem;
            margin-top: 1rem;
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Align success/error messages center */
        .stAlert {
            text-align: center;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # 2) Title + Container
    st.markdown("<div class='main-login-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='login-title'>Civic Catalyst Login</h1>", unsafe_allow_html=True)

    if 'jwt_token' not in st.session_state:
        st.session_state.jwt_token = None

    # If user already has a valid token, skip login
    if st.session_state.jwt_token is not None:
        is_valid, username, role = verify_jwt_token(st.session_state.jwt_token)
        if is_valid:
            # Already logged in
            st.markdown("<p style='text-align:center;'><b>You are already logged in.</b></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # close container
            return True

    # 3) LOGIN FORM
    st.subheader("Login to Your Account")
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            success, user_role = verify_user(username, password)
            if success:
                token = create_jwt_token(username, user_role)
                if token:
                    st.session_state.jwt_token = token
                    st.success("Login successful! Redirecting...")
                    st.experimental_rerun()
                else:
                    st.error("Error creating session token.")
            else:
                st.error("Invalid username or password.")

    # 4) REGISTRATION SECTION
    with st.expander("New User? Register Here", expanded=False):
        st.write("Create a new account to explore the Civic Catalyst platform.")
        with st.form("registration_form", clear_on_submit=True):
            new_username = st.text_input("New Username", key="reg_username")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            register = st.form_submit_button("Register")

            if register:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long.")
                else:
                    if create_user(new_username, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists or registration failed.")

    st.markdown("</div>", unsafe_allow_html=True)  # End main container

    return False

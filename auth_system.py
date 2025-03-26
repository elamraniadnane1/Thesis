import jwt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
import time 
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

    st.markdown(
    """
    <style>
    /* 1) ANIMATED BACKGROUND GRADIENT */
    @keyframes backgroundGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(-45deg, #F0F4F8, #D9E4F5, #ACB9D7, #E4ECF7);
        background-size: 300% 300%;
        animation: backgroundGradient 15s ease infinite;
        margin: 0;
        padding: 0;
    }

    /* 2) MAIN CONTAINER - NEON GLOW & TRANSPARENCY */
    .main-login-container {
        max-width: 500px;
        margin: 5% auto;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(6px);
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        border-radius: 1rem;
        padding: 2rem;
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    .main-login-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 180deg,
            #00b09b, #96c93d,
            #96c93d, #00b09b
        );
        animation: rotateNeon 8s linear infinite;
        transform: translate(-50%, -50%);
        z-index: -1;
    }
    @keyframes rotateNeon {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }

    /* Fancy Text Inside */
    .login-message {
        font-size: 22px;
        font-weight: 600;
        color: #2B3E50;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }

    /* 3) TITLE - ANIMATED TEXT GRADIENT */
    .login-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: hueShift 5s linear infinite;
    }
    @keyframes hueShift {
        0%   { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }

    /* 4) LABELS - Subtle Shadow */
    label {
        text-shadow: 0 1px 1px rgba(0,0,0,0.06);
        font-weight: 600 !important;
    }

    /* 5) BUTTONS - GRADIENT, HOVER RAISE, NEON SHADOW */
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
        box-shadow: 0 3px 8px rgba(58, 175, 169, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2B7A78 0%, #3AAFA9 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(43, 122, 120, 0.3);
    }
    .stButton>button:active {
        transform: scale(0.96);
        box-shadow: 0 2px 5px rgba(43,122,120,0.2);
    }

    /* 6) INPUT FIELDS - Rounded + Focus Shadow */
    input[type="text"], input[type="password"] {
        border: 1px solid #ccc !important;
        padding: 0.6rem !important;
        border-radius: 0.4rem;
        font-size: 1rem !important;
        width: 100% !important;
        transition: box-shadow 0.3s ease;
    }
    input[type="text"]:focus, input[type="password"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(58,175,169,0.2);
    }

    /* 7) FORMS */
    .login-form, .registration-form {
        margin-bottom: 2rem;
    }

    /* 8) EXPANDER HEADERS */
    .st-expanderHeader {
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 1rem;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 9) ALERT MESSAGES (success/error) */
    .stAlert {
        text-align: center;
    }
    </style>

    <!-- HTML for the container + title in ONE snippet -->
    <div class="main-login-container">
        <h1 class="login-title">Civic Catalyst</h1>
        <h2 class="login-message">🌟 Welcome to Civic Catalyst AI Toolkit! 🚀</h2>
        <p>Your gateway to intelligent citizen participation.</p>
    </div>
    """,
    unsafe_allow_html=True
)


    if 'jwt_token' not in st.session_state:
        st.session_state.jwt_token = None


    if st.session_state.jwt_token is not None:
        is_valid, username, role = verify_jwt_token(st.session_state.jwt_token)
        
        if is_valid:
            message_placeholder = st.empty()
            message_placeholder.markdown("<p style='text-align:center;'><b>You are already logged in.</b></p>", unsafe_allow_html=True)
            
            # Wait for 2 seconds and then clear the message
            time.sleep(2)
            message_placeholder.empty()  # Removes the message
    
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

import jwt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

# Constants for JWT
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

def login_page():
    """Display the login interface."""
    try:
        if 'jwt_token' not in st.session_state:
            st.session_state.jwt_token = None

        if st.session_state.jwt_token is not None:
            # Verify existing token
            is_valid, username, role = verify_jwt_token(st.session_state.jwt_token)
            if is_valid:
                return True

        st.title("Login")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                success, role = verify_user(username, password)
                if success:
                    token = create_jwt_token(username, role)
                    if token:
                        st.session_state.jwt_token = token
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Error creating session token")
                else:
                    st.error("Invalid username or password")
        
        # Registration section
        with st.expander("New User? Register Here"):
            with st.form("registration_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("Register")
                
                if register:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        if create_user(new_username, new_password):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Username already exists or registration failed")

        return False
    except Exception as e:
        st.error(f"Error in login page: {str(e)}")
        return False

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

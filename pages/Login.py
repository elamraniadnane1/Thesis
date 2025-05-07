import streamlit as st
import openai
import os
import tempfile
import pandas as pd
from gtts import gTTS
from pymongo import MongoClient
import hashlib
import time
import jwt
import re
import uuid
import json
import base64
# Add this import at the top of your file, with your other imports
import streamlit.components.v1 as components
from io import BytesIO
from PIL import Image
import requests
from datetime import datetime, timedelta
from streamlit_lottie import st_lottie
import streamlit_authenticator as stauth
import importlib.util
import streamlit as st
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

# Add this simple session-based authentication class
class SessionAuth:
    """Simple session-based authentication that doesn't rely on cookies"""
    
    def __init__(self, prefix="civic_"):
        self.prefix = prefix
        # Initialize session state for auth data if not exists
        if f"{self.prefix}auth" not in st.session_state:
            st.session_state[f"{self.prefix}auth"] = {}
            
    def ready(self):
        """Always returns True as we don't need to wait for cookies"""
        return True
        
    def get(self, key, default=None):
        """Get a value from session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return auth_data.get(key, default)
        
    def __getitem__(self, key):
        """Dictionary-like access"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
        
    def __setitem__(self, key, value):
        """Set a value in session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        auth_data[key] = value
        st.session_state[f"{self.prefix}auth"] = auth_data
        
    def __delitem__(self, key):
        """Delete a value from session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        if key in auth_data:
            del auth_data[key]
            st.session_state[f"{self.prefix}auth"] = auth_data
            
    def __contains__(self, key):
        """Check if a key exists"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return key in auth_data
        
    def __len__(self):
        """Get number of keys"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return len(auth_data)
        
    def save(self):
        """No-op as we're using session state directly"""
        pass
class SimpleCookieManager:
    """Simple session-based cookie alternative when streamlit_cookies_manager fails"""
    
    def __init__(self, prefix="civic_", password="YOUR_STRONG_PASSWORD"):
        self.prefix = prefix
        self._is_ready = True
        
        # Create a key for encryption
        salt = b'streamlit_cookie_manager'  # Not for security, just for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet = Fernet(key)
        
        # Initialize cookies dict in session state if not exists
        if f"{self.prefix}cookies" not in st.session_state:
            st.session_state[f"{self.prefix}cookies"] = {}
    
    def ready(self):
        """Check if cookie manager is ready to use"""
        return self._is_ready
    
    def get(self, key, default=None):
        """Get a cookie value"""
        if not self.ready():
            return default
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return cookies.get(key, default)
    
    def __getitem__(self, key):
        """Dictionary-like access to cookies"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key, value):
        """Set a cookie value"""
        if not self.ready():
            return
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        cookies[key] = value
        st.session_state[f"{self.prefix}cookies"] = cookies
    
    def __delitem__(self, key):
        """Delete a cookie"""
        if not self.ready():
            return
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        if key in cookies:
            del cookies[key]
            st.session_state[f"{self.prefix}cookies"] = cookies
    
    def __contains__(self, key):
        """Check if a cookie exists"""
        if not self.ready():
            return False
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return key in cookies
    
    def __len__(self):
        """Get number of cookies"""
        if not self.ready():
            return 0
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return len(cookies)
    
    def save(self):
        """Save cookies - this is a no-op for the simple version as we're using session state"""
        pass
    
    def load(self):
        """Load cookies - this is a no-op for the simple version as we're using session state"""
        pass

# Add a check for required packages
def check_required_packages():
    """Check if required packages are installed and provide instructions if not"""
    missing_packages = []
    
    packages = {
        "streamlit_lottie": "streamlit-lottie",
        "pymongo": "pymongo[srv]",
        "streamlit_authenticator": "streamlit-authenticator",
        "cryptography": "cryptography"
    }
    
    for module, package in packages.items():
        if importlib.util.find_spec(module) is None:
            missing_packages.append(package)
    
    if missing_packages:
        st.error("Missing required packages. Please install them with:")
        st.code(f"pip install {' '.join(missing_packages)}")
        st.stop()
# =============================================================================
# GLOBAL CONFIG & CONSTANTS
# =============================================================================
SECRET_KEY = os.environ.get("SECRET_KEY", "mysecretkey")
JWT_ALGORITHM = "HS256"
COOKIE_PASSWORD = os.environ.get("COOKIE_PASSWORD", "YOUR_STRONG_PASSWORD")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

MAX_LOGIN_ATTEMPTS = 3  # Lock out user after 3 failed attempts
LOCKOUT_TIME = 15       # Lockout duration (minutes)

# Session persistence settings
SESSION_EXPIRY = 14     # Days before session expires (extended from 7)
REFRESH_TOKEN_EXPIRY = 30  # Days before refresh token expires

# Theme settings
DEFAULT_THEME = "light"
AVAILABLE_THEMES = ["light", "dark", "blue", "green", "burgundy"]

# =============================================================================
# ANIMATIONS & VISUAL ELEMENTS
# =============================================================================
def load_lottie_url(url: str):
    """Load Lottie animation from URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading animation: {e}")
        return None
def safe_mongo_operation(operation_name, function, *args, **kwargs):
    """Safely execute a MongoDB operation with proper error handling"""
    try:
        return function(*args, **kwargs)
    except Exception as e:
        st.error(f"Error in MongoDB operation '{operation_name}': {e}")
        return None
# Pre-defined Lottie animations
LOTTIE_ANIMATIONS = {
    "login": "https://assets8.lottiefiles.com/packages/lf20_hy4txm7l.json",
    "welcome": "https://assets5.lottiefiles.com/packages/lf20_g3dzz0po.json",
    "success": "https://assets3.lottiefiles.com/packages/lf20_rZQs19.json",
    "error": "https://assets5.lottiefiles.com/packages/lf20_qpwbiyxf.json",
    "loading": "https://assets3.lottiefiles.com/packages/lf20_Yey9E2.json",
    "profile": "https://assets3.lottiefiles.com/packages/lf20_qdexz4gx.json",
    "morocco": "https://assets10.lottiefiles.com/packages/lf20_UgZWvP.json"
}

def get_theme_colors(theme="light"):
    """Return color scheme based on selected theme."""
    themes = {
        "light": {
            "primary": "#3AAFA9",
            "secondary": "#2B7A78",
            "background": "linear-gradient(-45deg, #F0F4F8, #D9E4F5, #ACB9D7, #E4ECF7)",
            "text": "#2B3E50",
            "accent": "#DEF2F1"
        },
        "dark": {
            "primary": "#BB86FC",
            "secondary": "#3700B3",
            "background": "linear-gradient(-45deg, #121212, #1E1E1E, #2C2C2C, #1A1A1A)",
            "text": "#E0E0E0",
            "accent": "#03DAC6"
        },
        "blue": {
            "primary": "#4285F4",
            "secondary": "#1A73E8",
            "background": "linear-gradient(-45deg, #E8F0FE, #C2D7FE, #A1C2FA, #D2E3FC)",
            "text": "#1F1F1F",
            "accent": "#D2E3FC"
        },
        "green": {
            "primary": "#34A853",
            "secondary": "#188038",
            "background": "linear-gradient(-45deg, #E6F4EA, #CEEAD6, #A8DAB5, #CEEAD6)",
            "text": "#1F1F1F",
            "accent": "#CEEAD6"
        },
        "burgundy": {
            "primary": "#9E1B32",
            "secondary": "#6D1423",
            "background": "linear-gradient(-45deg, #F8E0E0, #F1C2C2, #E9A5A5, #F1C2C2)",
            "text": "#1F1F1F",
            "accent": "#F1C2C2"
        }
    }
    return themes.get(theme, themes["light"])

def apply_theme():
    """Apply the selected theme from session state."""
    theme = st.session_state.get("theme", DEFAULT_THEME)
    colors = get_theme_colors(theme)
    
    # Apply CSS with the color scheme
    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: {colors["primary"]};
            --secondary-color: {colors["secondary"]};
            --background-gradient: {colors["background"]};
            --text-color: {colors["text"]};
            --accent-color: {colors["accent"]};
        }}
        
        body {{
            font-family: 'Poppins', sans-serif;
            background: var(--background-gradient);
            background-size: 300% 300%;
            animation: backgroundGradient 15s ease infinite;
            color: var(--text-color) !important;
            margin: 0;
            padding: 0;
        }}
        
        @keyframes backgroundGradient {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .main-login-container {{
            max-width: 500px;
            margin: 5% auto;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(6px);
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            border-radius: 1rem;
            padding: 2rem;
            position: relative;
            text-align: center;
            overflow: hidden;
        }}
        
        .main-login-container::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 180deg, var(--primary-color), var(--secondary-color), var(--secondary-color), var(--primary-color));
            animation: rotateNeon 8s linear infinite;
            transform: translate(-50%, -50%);
            z-index: -1;
            opacity: 0.6;
        }}
        
        @keyframes rotateNeon {{
            0%   {{ transform: translate(-50%, -50%) rotate(0deg); }}
            100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
        }}
        
        .moroccan-flag {{
            width: 80px;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        @keyframes pulse {{
            0%   {{ transform: scale(1); }}
            50%  {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        
        .login-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: hueShift 5s linear infinite;
        }}
        
        @keyframes hueShift {{
            0%   {{ filter: hue-rotate(0deg); }}
            100% {{ filter: hue-rotate(360deg); }}
        }}
        
        .login-message {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-color);
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        
        .stButton>button {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 1rem;
            height: 3rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }}
        
        input[type="text"], input[type="password"], input[type="email"] {{
            border: 1px solid #ccc;
            padding: 0.8rem 1rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            width: 100%;
            transition: all 0.3s ease;
            background-color: rgba(255, 255, 255, 0.8);
        }}
        
        input[type="text"]:focus, input[type="password"]:focus, input[type="email"]:focus {{
            outline: none;
            box-shadow: 0 0 0 2px var(--primary-color);
            border-color: var(--primary-color);
            background-color: white;
        }}
        
        .fancy-card {{
            background: rgba(255, 255, 255, 0.8);
            border-radius: 0.8rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 5px solid var(--primary-color);
            transition: all 0.3s ease;
        }}
        
        .fancy-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        
        .profile-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .profile-avatar {{
            width: 120px;
            height: 120px;
            border-radius: 60px;
            object-fit: cover;
            border: 4px solid var(--primary-color);
            margin-bottom: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.4rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem;
            background-color: var(--primary-color);
            color: white;
        }}
        
        .form-container {{
            background: rgba(255, 255, 255, 0.8);
            border-radius: 0.8rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(5px);
        }}
        
        .form-header {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }}
        
        .input-group {{
            position: relative;
            margin-bottom: 1.5rem;
        }}
        
        .input-group i {{
            position: absolute;
            left: 1rem;
            top: 0.8rem;
            color: #888;
        }}
        
        .input-group input {{
            padding-left: 2.5rem;
        }}
        
        .stTabs {{
            background: rgba(255, 255, 255, 0.8);
            border-radius: 0.8rem;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .stAlert {{
            border-radius: 0.5rem !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }}
        
        /* Fancy animation for alerts */
        @keyframes slideIn {{
            0% {{ transform: translateY(-20px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}
        
        .stAlert {{
            animation: slideIn 0.3s ease-out forwards;
        }}
        
        /* Sidebar styling */
        .sidebar .sidebar-content {{
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            font-size: 0.9rem;
            color: var(--text-color);
            opacity: 0.7;
        }}
        
        /* Audio player styling */
        audio {{
            width: 100%;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
        }}
        
        /* Language selector */
        .language-selector {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .language-option {{
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.5);
            border: 2px solid transparent;
        }}
        
        .language-option:hover {{
            background: rgba(255, 255, 255, 0.8);
        }}
        
        .language-option.active {{
            border-color: var(--primary-color);
            background: white;
            font-weight: bold;
        }}
        
        /* Password strength indicator */
        .password-strength {{
            height: 5px;
            border-radius: 2px;
            margin-top: 0.5rem;
            transition: all 0.3s ease;
        }}
        
        .strength-text {{
            font-size: 0.8rem;
            margin-top: 0.3rem;
            text-align: right;
        }}
        
        .strength-weak {{
            background: linear-gradient(90deg, #ff4d4d, #ff9999);
        }}
        
        .strength-medium {{
            background: linear-gradient(90deg, #ffaa00, #ffcc66);
        }}
        
        .strength-strong {{
            background: linear-gradient(90deg, #00cc44, #5ee688);
        }}
        
        /* Toast notifications */
        .toast {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            z-index: 9999;
            animation: fadeInOut 3s forwards;
        }}
        
        @keyframes fadeInOut {{
            0% {{ opacity: 0; transform: translateY(-20px); }}
            10% {{ opacity: 1; transform: translateY(0); }}
            90% {{ opacity: 1; transform: translateY(0); }}
            100% {{ opacity: 0; transform: translateY(-20px); }}
        }}
        
        /* Loading spinner */
        .loader {{
            border: 5px solid #f3f3f3;
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Responsive layout adjustments */
        @media (max-width: 768px) {{
            .main-login-container {{
                max-width: 90%;
                margin: 10% auto;
                padding: 1.5rem;
            }}
            
            .login-title {{
                font-size: 2rem;
            }}
            
            .login-message {{
                font-size: 1rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_base64_encoded_image(image_path):
    """Get base64 encoded image for HTML embedding."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_image_base64_from_url(url):
    """Get base64 encoded image from URL for HTML embedding."""
    try:
        response = requests.get(url)
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

def add_gradient_bg():
    """Add animated gradient background to Streamlit app."""
    st.markdown(
        """
        <div class="bg-container"></div>
        <style>
        .bg-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def display_main_header_with_animation():
    """Display the main animated header with lottie animation and title."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Load welcome animation
        welcome_animation = load_lottie_url(LOTTIE_ANIMATIONS["morocco"])
        if welcome_animation:
            st_lottie(welcome_animation, height=180, key="welcome")
            
        st.markdown(
            f"""
            <h1 class="login-title">{t('welcome_title')}</h1>
            <h2 class="login-message">🌟 {t('welcome_message')} 🚀</h2>
            """,
            unsafe_allow_html=True
        )

def display_login_animation():
    """Display login-related animation."""
    login_animation = load_lottie_url(LOTTIE_ANIMATIONS["login"])
    if login_animation:
        st_lottie(login_animation, height=120, key="login")

# =============================================================================
# TRANSLATIONS & HELPER FUNCTIONS
# =============================================================================
translations = {
    "welcome_title": {
        "en": "Civic Catalyst",
        "fr": "Civic Catalyst",
        "ar": "المحفز المدني",
        "darija": "المحفز المدني"
    },
    "welcome_message": {
        "en": "Welcome to Civic Catalyst AI Toolkit! Your gateway to intelligent citizen participation.",
        "fr": "Bienvenue sur Civic Catalyst AI Toolkit ! Votre porte d'entrée vers une participation citoyenne intelligente.",
        "ar": "مرحباً بكم في مجموعة أدوات الذكاء الاصطناعي للمشاركة المدنية! بوابتك للمشاركة الذكية.",
        "darija": "مرحباً بكم فـ Civic Catalyst! البوابة ديالك للمشاركة المدنية الذكية."
    },
    "login_title": {
        "en": "Login to Your Account",
        "fr": "Connectez-vous à votre compte",
        "ar": "تسجيل الدخول إلى حسابك",
        "darija": "دخل لحسابك"
    },
    "username": {
        "en": "Username",
        "fr": "Nom d'utilisateur",
        "ar": "اسم المستخدم",
        "darija": "اسم المستخدم"
    },
    "password": {
        "en": "Password",
        "fr": "Mot de passe",
        "ar": "كلمة المرور",
        "darija": "كلمة السر"
    },
    "login_button": {
        "en": "Login",
        "fr": "Se connecter",
        "ar": "تسجيل الدخول",
        "darija": "دخول"
    },
    "register_header": {
        "en": "New User? Register Here",
        "fr": "Nouvel utilisateur ? Inscrivez-vous ici",
        "ar": "مستخدم جديد؟ سجل هنا",
        "darija": "مستعمل جديد؟ سجل هنا"
    },
    "gpt_key_prompt": {
        "en": "Optional: Provide Your GPT API Key",
        "fr": "Optionnel : Fournissez votre clé API GPT",
        "ar": "اختياري: أدخل مفتاح API الخاص بـ GPT",
        "darija": "اختياري: دخل مفتاح GPT ديالك"
    },
    "cin": {
        "en": "CIN Number",
        "fr": "Numéro CIN",
        "ar": "رقم بطاقة الهوية",
        "darija": "رقم بطاقة الهوية"
    },
    "forgot_password": {
        "en": "🔒 Forgot Password?",
        "fr": "🔒 Mot de passe oublié ?",
        "ar": "🔒هل نسيت كلمة المرور؟",
        "darija": "🔒نسيت كلمة السر؟"
    },
    "logout_button": {
        "en": "Logout",
        "fr": "Déconnexion",
        "ar": "تسجيل الخروج",
        "darija": "تسجيل الخروج"
    },
    "change_password": {
        "en": "Change Your Password",
        "fr": "Changer votre mot de passe",
        "ar": "تغيير كلمة المرور الخاصة بك",
        "darija": "بدل كلمة السر ديالك"
    },
    "profile_header": {
        "en": "Your Profile",
        "fr": "Votre profil",
        "ar": "ملفك الشخصي",
        "darija": "البروفيل ديالك"
    },
    "active_sessions": {
        "en": "Active Sessions",
        "fr": "Sessions actives",
        "ar": "الجلسات النشطة",
        "darija": "الجلسات النشيطة"
    },
    "reset_password_prompt": {
        "en": "Reset Your Password",
        "fr": "Réinitialiser votre mot de passe",
        "ar": "إعادة تعيين كلمة المرور الخاصة بك",
        "darija": "عاود تعيين كلمة السر ديالك"
    },
    "registration_success": {
        "en": "Registration successful! You can now log in.",
        "fr": "Inscription réussie ! Vous pouvez maintenant vous connecter.",
        "ar": "تم التسجيل بنجاح! يمكنك الآن تسجيل الدخول.",
        "darija": "تسجل بنجاح! دابا تقدر تدخل"
    },
    "invalid_credentials": {
        "en": "Invalid credentials",
        "fr": "Identifiants invalides",
        "ar": "بيانات اعتماد غير صحيحة",
        "darija": "معطيات خاطئة"
    },
    "error_message": {
        "en": "An error occurred. Please try again.",
        "fr": "Une erreur est survenue. Veuillez réessayer.",
        "ar": "حدث خطأ. الرجاء المحاولة مرة أخرى.",
        "darija": "وقع مشكل، عاود المحاولة"
    },
    "language_updated": {
        "en": "Language updated to: ",
        "fr": "Langue mise à jour : ",
        "ar": "تم تحديث اللغة إلى: ",
        "darija": "تم تغيير اللغة ل: "
    },
    "logout_success": {
        "en": "Successfully logged out.",
        "fr": "Déconnexion réussie.",
        "ar": "تم تسجيل الخروج بنجاح.",
        "darija": "خرجت بنجاح"
    },
    "session_expired": {
        "en": "Session expired. Please log in again.",
        "fr": "Session expirée. Veuillez vous reconnecter.",
        "ar": "انتهت الجلسة. الرجاء تسجيل الدخول مرة أخرى.",
        "darija": "الجلسة سالات. دخل من جديد"
    },
    "password_reset_success": {
        "en": "Password reset successful! Please log in again.",
        "fr": "Réinitialisation du mot de passe réussie ! Veuillez vous reconnecter.",
        "ar": "تم إعادة تعيين كلمة المرور بنجاح! الرجاء تسجيل الدخول مرة أخرى.",
        "darija": "تعاود تعيين كلمة السر بنجاح! دخل مرة أخرى"
    },
    "password_reset_failed": {
        "en": "Password reset failed. Link may be invalid or expired.",
        "fr": "Échec de la réinitialisation du mot de passe. Le lien peut être invalide ou expiré.",
        "ar": "فشل إعادة تعيين كلمة المرور. قد يكون الرابط غير صالح أو منتهي الصلاحية.",
        "darija": "فشل تعيين كلمة السر. الرابط ربما ما بقا صالحش"
    },
    "change_password_success": {
        "en": "Password updated successfully!",
        "fr": "Mot de passe mis à jour avec succès !",
        "ar": "تم تحديث كلمة المرور بنجاح!",
        "darija": "كلمة السر تبدلات بنجاح!"
    },
    "change_password_error": {
        "en": "Error updating password. Please try again.",
        "fr": "Erreur lors de la mise à jour du mot de passe. Veuillez réessayer.",
        "ar": "حدث خطأ أثناء تحديث كلمة المرور. الرجاء المحاولة مرة أخرى.",
        "darija": "وقع مشكل ف تغيير كلمة السر، عاود المحاولة"
    },
    "profile_image_saved": {
        "en": "Profile image saved successfully!",
        "fr": "Image de profil enregistrée avec succès !",
        "ar": "تم حفظ صورة الملف الشخصي بنجاح!",
        "darija": "تصويرة البروفيل تسجات بنجاح!"
    },
    "remember_me": {
        "en": "Remember Me",
        "fr": "Se souvenir de moi",
        "ar": "تذكرني",
        "darija": "تفكرني"
    },
    "theme_settings": {
        "en": "Theme Settings",
        "fr": "Paramètres du thème",
        "ar": "إعدادات المظهر",
        "darija": "إعدادات المظهر"
    },
    "voice_feedback": {
        "en": "Voice Feedback",
        "fr": "Retour vocal",
        "ar": "تعليقات صوتية",
        "darija": "تعليقات صوتية"
    },
    "password_strength": {
        "en": "Password Strength",
        "fr": "Force du mot de passe",
        "ar": "قوة كلمة المرور",
        "darija": "قوة كلمة السر"
    },
    "weak": {
        "en": "Weak",
        "fr": "Faible",
        "ar": "ضعيف",
        "darija": "ضعيف"
    },
    "medium": {
        "en": "Medium",
        "fr": "Moyen",
        "ar": "متوسط",
        "darija": "متوسط"
    },
    "strong": {
        "en": "Strong",
        "fr": "Fort",
        "ar": "قوي",
        "darija": "قوي"
    },
    "welcome_back": {
        "en": "Welcome back",
        "fr": "Bon retour",
        "ar": "مرحبًا بعودتك",
        "darija": "مرحبا بك من جديد"
    },
    "stay_signed_in": {
        "en": "Stay signed in",
        "fr": "Rester connecté",
        "ar": "البقاء مسجل الدخول",
        "darija": "بقى مسجل الدخول"
    },
    "account_settings": {
        "en": "Account Settings",
        "fr": "Paramètres du compte",
        "ar": "إعدادات الحساب",
        "darija": "إعدادات الحساب"
    },
    "privacy_policy": {
        "en": "Privacy Policy",
        "fr": "Politique de confidentialité",
        "ar": "سياسة الخصوصية",
        "darija": "سياسة الخصوصية"
    }
}

def add_to_translations(key, translations_dict):
    """Add new translation entries dynamically."""
    global translations
    if key not in translations:
        translations[key] = translations_dict

def t(key):
    """Translate text based on current session language."""
    lang = st.session_state.get("site_language", "en")
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("en", key))

def text_to_speech(text, lang="en"):
    """Convert text to speech and return audio player HTML."""
    try:
        tts = gTTS(text=text, lang=lang)
        with BytesIO() as fp:
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay="false">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def set_language_in_cookie(cookies, language):
    """Store chosen language in cookie and session state with enhanced persistence."""
    st.session_state["site_language"] = language
    cookies["site_language"] = language
    
    # Also store in more persistent localStorage via HTML/JS
    js_code = f"""
    <script>
        localStorage.setItem('civic_language', '{language}');
        console.log('Language set in localStorage: {language}');
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    
    # Save cookies
    cookies.save()
    
    # Optional: Play language confirmation
    lang_confirmations = {
        "en": "Language set to English",
        "fr": "Langue définie sur Français",
        "ar": "تم تعيين اللغة إلى العربية",
        "darija": "تم تعيين اللغة إلى الدارجة"
    }
    
    if st.session_state.get("voice_feedback", False):
        audio_html = text_to_speech(lang_confirmations.get(language, "Language updated"), 
                                    "en" if language not in ["ar", "fr"] else language)
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)

def get_language_from_cookie(cookies):
    """Retrieve language setting from session state or cookies"""
    # First check session state
    if "site_language" in st.session_state:
        return
    
    # Try to get from cookies
    lang = cookies.get("site_language")
    if lang:
        st.session_state["site_language"] = lang
        return
    
    # Default language if not found
    st.session_state["site_language"] = "en"

def set_theme_in_storage(theme):
    """Store theme preference in session state and local storage."""
    st.session_state["theme"] = theme
    
    # Store in localStorage for persistence
    js_code = f"""
    <script>
        localStorage.setItem('civic_theme', '{theme}');
        console.log('Theme set in localStorage: {theme}');
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

def get_theme_from_storage():
    """Retrieve theme from localStorage if available."""
    if "theme" in st.session_state:
        return
    
    js_get_theme = """
    <script>
        var storedTheme = localStorage.getItem('civic_theme');
        if (storedTheme) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: storedTheme,
                dataType: 'str',
                componentInstance: 'theme_storage'
            }, '*');
        }
    </script>
    """
    st.markdown(js_get_theme, unsafe_allow_html=True)
    
    # Create a placeholder for the localStorage value
    if 'theme_from_storage' not in st.session_state:
        st.session_state.theme_from_storage = None
    
    # This could use streamlit components, but simplified here
    if st.session_state.get('theme_from_storage'):
        st.session_state["theme"] = st.session_state.theme_from_storage
    else:
        st.session_state["theme"] = DEFAULT_THEME

# =============================================================================
# ENHANCED SECURITY & DB FUNCTIONS
# =============================================================================
def init_auth():
    """Check MongoDB connection on startup and initialize collections if needed."""
    try:
        client = MongoClient(MONGO_URI)
        client.server_info()  # Will throw exception if connection fails
        
        # Initialize database collections if they don't exist
        db = client["CivicCatalyst"]
        
        # Clean up null session_id values before creating the index
        db.sessions.delete_many({"session_id": None})
        
        # Ensure indexes for performance
        if "users" in db.list_collection_names():
            try:
                db.users.create_index("username", unique=True)
                db.users.create_index("email", unique=True)
            except Exception as e:
                print(f"Warning: Error creating user indexes: {e}")
        
        if "sessions" in db.list_collection_names():
            try:
                db.sessions.create_index("username")
                # Make sure we don't have any null session_ids before creating unique index
                db.sessions.delete_many({"session_id": None})
                db.sessions.create_index("session_id", unique=True)
                # TTL index with expiry
                try:
                    db.sessions.create_index("expires", expireAfterSeconds=0)
                except Exception as e:
                    print(f"Warning: Error creating TTL index: {e}")
            except Exception as e:
                print(f"Warning: Error creating session indexes: {e}")
        
        # Clean up expired sessions
        try:
            db.sessions.delete_many({"expires": {"$lt": datetime.utcnow()}})
        except Exception as e:
            print(f"Warning: Error cleaning up expired sessions: {e}")
        
        return True
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def hash_password(password: str) -> str:
    """Hash password with SHA-256 and add salt."""
    salt = os.environ.get("PASSWORD_SALT", "default_salt")
    salted = password + salt
    return hashlib.sha256(salted.encode()).hexdigest()

def generate_password_hash(password: str) -> tuple:
    """Generate a stronger password hash with salt for new accounts."""
    salt = os.urandom(32).hex()  # Generate a random salt
    hashed = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000  # Number of iterations
    ).hex()
    return hashed, salt

def verify_password_hash(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored hash."""
    hashed = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    return hashed == stored_hash

def check_password_strength(password):
    """Check password strength and return (is_valid, message, strength)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long", 0
    
    checks = {
        'uppercase': re.search(r'[A-Z]', password),
        'lowercase': re.search(r'[a-z]', password),
        'digit': re.search(r'\d', password),
        'special': re.search(r'[!@#$%^&*(),.?\":{}|<>]', password)
    }
    
    strength = sum(1 for check in checks.values() if check)
    
    # Check for common password patterns
    common_patterns = ['password', '123456', 'qwerty', 'admin']
    if any(pattern in password.lower() for pattern in common_patterns):
        strength -= 1
    
    # Check for sequential characters
    if re.search(r'(abc|bcd|cde|def|efg|123|234|345|456|567|678|789)', password.lower()):
        strength -= 1
    
    # Adjust strength score
    strength = max(0, min(strength, 4))
    
    # Generate missing requirements message
    missing = []
    if not checks['uppercase']: missing.append("uppercase letters")
    if not checks['lowercase']: missing.append("lowercase letters")
    if not checks['digit']:     missing.append("numbers")
    if not checks['special']:   missing.append("special characters")
    
    if strength < 2:
        joined = ", ".join(missing)
        return False, f"Weak password. Please include: {joined}", strength
    elif strength < 3:
        joined = ", ".join(missing)
        return True, f"Medium strength. Consider adding: {joined}", strength
    else:
        return True, "Strong password", strength

def create_user(username: str, password: str, email: str, role: str = 'citizen', cin: str = None) -> bool:
    """Create user with enhanced security and data validation."""
    # Validate email format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        st.error("Invalid email format")
        return False
    
    # Validate username (alphanumeric + some special chars, 3-20 chars)
    if not re.match(r"^[a-zA-Z0-9_.-]{3,20}$", username):
        st.error("Username must be 3-20 characters and contain only letters, numbers, underscore, dot, or hyphen")
        return False

    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Check if username or email is already used
        if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):
            st.error("Username or email already exists")
            return False
        
        # Check password strength
        is_valid, msg, _ = check_password_strength(password)
        if not is_valid:
            st.error(msg)
            return False
        
        # Generate secure password hash
        password_hash, salt = generate_password_hash(password)
        
        # Create user with extended profile fields
        new_user = {
            "username": username,
            "password_hash": password_hash,
            "password_salt": salt,
            "email": email,
            "role": role,
            "cin": cin,
            "failed_attempts": 0,
            "locked_until": None,
            "created_at": datetime.utcnow(),
            "profile": {
                "last_login": None,
                "login_count": 0,
                "preferences": {
                    "language": st.session_state.get("site_language", "en"),
                    "theme": st.session_state.get("theme", DEFAULT_THEME)
                }
            },
            "security": {
                "2fa_enabled": False,
                "password_last_changed": datetime.utcnow(),
                "password_expires": datetime.utcnow() + timedelta(days=90)  # Password expires in 90 days
            }
        }
        db.users.insert_one(new_user)
        
        # Audit log for registration
        db.activity_log.insert_one({
            "user": username,
            "action": "registration",
            "timestamp": datetime.utcnow(),
            "ip_address": None,  # Could store IP if available
            "details": "New user registration"
        })
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        client.close()

def verify_user(username: str, password: str):
    """
    Enhanced verification with account lockout and session management.
    Returns (is_valid, role, error_message).
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        user = db.users.find_one({"username": username})
        
        if not user:
            # For security, use the same response as for invalid password
            # to prevent username enumeration
            return False, None, "Invalid username or password"
        
        # Check lockout
        if user.get('locked_until') and user['locked_until'] > datetime.utcnow():
            remaining_secs = (user['locked_until'] - datetime.utcnow()).total_seconds()
            remaining_mins = int(remaining_secs // 60) + 1
            return False, None, f"Account locked. Try again in {remaining_mins} minutes"
        
        # Check password using proper hash verification
        # For backward compatibility, check both methods
        if 'password_salt' in user:
            # Preferred new method with salt
            is_valid = verify_password_hash(password, user["password_hash"], user["password_salt"])
        else:
            # Legacy method (backward compatibility)
            is_valid = user["password_hash"] == hash_password(password)
        
        if is_valid:
            # Reset lockout counters
            db.users.update_one(
                {"_id": user["_id"]}, 
                {
                    "$set": {
                        "failed_attempts": 0,
                        "locked_until": None,
                        "profile.last_login": datetime.utcnow(),
                    },
                    "$inc": {"profile.login_count": 1}
                }
            )
            
            # Store user preferences in session state
            if "profile" in user and "preferences" in user["profile"]:
                prefs = user["profile"]["preferences"]
                if "language" in prefs:
                    st.session_state["site_language"] = prefs["language"]
                if "theme" in prefs:
                    st.session_state["theme"] = prefs["theme"]
            
            # Audit log for successful login
            db.activity_log.insert_one({
                "user": username,
                "action": "login",
                "timestamp": datetime.utcnow(),
                "ip_address": None,  # Could store IP if available
                "details": "Successful login"
            })
            
            return True, user["role"], None
        else:
            # Increase failed attempts
            new_attempts = user.get("failed_attempts", 0) + 1
            db.users.update_one({"_id": user["_id"]}, {"$set": {
                "failed_attempts": new_attempts
            }})
            
            # Audit log for failed login
            db.activity_log.insert_one({
                "user": username,
                "action": "login_failed",
                "timestamp": datetime.utcnow(),
                "details": f"Failed login attempt {new_attempts}"
            })
            
            if new_attempts >= MAX_LOGIN_ATTEMPTS:
                lock_time = datetime.utcnow() + timedelta(minutes=LOCKOUT_TIME)
                db.users.update_one({"_id": user["_id"]}, {"$set": {
                    "locked_until": lock_time
                }})
                
                # Log account lockout
                db.activity_log.insert_one({
                    "user": username,
                    "action": "account_locked",
                    "timestamp": datetime.utcnow(),
                    "details": f"Account locked for {LOCKOUT_TIME} minutes due to failed attempts"
                })
                
                return False, None, f"Account locked for {LOCKOUT_TIME} minutes"
            
            attempts_left = MAX_LOGIN_ATTEMPTS - new_attempts
            return False, None, f"Invalid password. {attempts_left} attempts remaining"
    except Exception as e:
        return False, None, f"Error verifying user: {e}"
    finally:
        client.close()

def create_jwt_token(username: str, role: str, remember_me: bool = False) -> str:
    """
    Create a JWT token with enhanced security features
    """
    try:
        # Set token lifetime based on "remember me" setting
        expiration_days = SESSION_EXPIRY if remember_me else 1
        
        # Generate a unique token ID for this session
        token_id = str(uuid.uuid4())  # Ensure this is never None
        
        # Current timestamp for various expiry calculations
        now = datetime.utcnow()
        
        # Create token payload
        payload = {
            "sub": username,  # Subject (username)
            "role": role,
            "jti": token_id,  # Unique token ID
            "iat": now,       # Issued at
            "nbf": now,       # Not valid before
            "exp": now + timedelta(days=expiration_days),  # Expiration
            "remember_me": remember_me
        }
        
        # Sign the token
        token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Record the token in database for tracking active sessions
        try:
            client = MongoClient(MONGO_URI)
            db = client["CivicCatalyst"]
            
            # Make sure session_id is never null
            if not token_id or token_id.strip() == "":
                token_id = str(uuid.uuid4())
                
            session_record = {
                "session_id": token_id,  # Now guaranteed not to be None
                "username": username,
                "token": token,
                "user_agent": None,  # Could store user agent if available
                "issued_at": now,
                "expires": now + timedelta(days=expiration_days),
                "revoked": False,
                "last_active": now
            }
            db.sessions.insert_one(session_record)
        except Exception as e:
            st.error(f"Error recording session: {e}")
        finally:
            client.close()
            
        return token
    except Exception as e:
        st.error(f"Error creating JWT token: {e}")
        return None

def verify_jwt_token(token: str) -> tuple:
    """
    Verify a JWT token with additional checks:
    - Check if token has been revoked in the database
    - Update last_active timestamp for the session
    
    Returns (is_valid, username, role).
    """
    try:
        # Decode and verify token signature
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        token_id = payload.get("jti")
        
        # Verify token in database (check if revoked, etc.)
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        session = db.sessions.find_one({"session_id": token_id})
        
        # Check if session exists and is not revoked
        if not session or session.get("revoked", False):
            return False, None, None
        
        # Update last_active timestamp
        db.sessions.update_one(
            {"session_id": token_id},
            {"$set": {"last_active": datetime.utcnow()}}
        )
        
        # Check if user still exists and has appropriate role
        user = db.users.find_one({"username": username})
        if not user or user.get("role") != role:
            return False, None, None
            
        return True, username, role
        
    except jwt.ExpiredSignatureError:
        st.error("Session expired. Please log in again.")
        return False, None, None
    except jwt.InvalidTokenError:
        st.error("Invalid token. Please log in again.")
        return False, None, None
    except Exception as e:
        st.error(f"Error verifying JWT token: {e}")
        return False, None, None
    finally:
        try:
            client.close()
        except:
            pass

def log_login_event(username: str):
    """Log a successful login event to MongoDB with extended information."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Get user profile for updating
        user = db.users.find_one({"username": username})
        if not user:
            return
            
        # Update user's last login time
        db.users.update_one(
            {"username": username},
            {"$set": {"profile.last_login": datetime.utcnow()},
             "$inc": {"profile.login_count": 1}}
        )
        
        # Log login history with additional data
        login_record = {
            "username": username,
            "timestamp": datetime.utcnow(),
            "user_agent": None,  # Could capture this if needed
            "ip_address": None,  # Could capture this if needed
            "success": True
        }
        db.login_history.insert_one(login_record)
    except Exception as e:
        st.error(f"Error logging login event: {e}")
    finally:
        client.close()

def revoke_session(session_id: str, username: str) -> bool:
    """Revoke a specific session."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Only allow users to revoke their own sessions
        result = db.sessions.update_one(
            {"session_id": session_id, "username": username},
            {"$set": {"revoked": True}}
        )
        
        # Log the event
        if result.modified_count > 0:
            db.activity_log.insert_one({
                "user": username,
                "action": "session_revoked",
                "timestamp": datetime.utcnow(),
                "details": f"Session {session_id} revoked"
            })
            return True
        return False
    except Exception as e:
        st.error(f"Error revoking session: {e}")
        return False
    finally:
        client.close()

def revoke_all_sessions(username: str, except_current: str = None) -> bool:
    """Revoke all sessions for a user except the current one."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        query = {"username": username, "revoked": False}
        if except_current:
            query["session_id"] = {"$ne": except_current}
            
        result = db.sessions.update_many(
            query,
            {"$set": {"revoked": True}}
        )
        
        # Log the event
        if result.modified_count > 0:
            db.activity_log.insert_one({
                "user": username,
                "action": "all_sessions_revoked",
                "timestamp": datetime.utcnow(),
                "details": f"Revoked {result.modified_count} sessions"
            })
            return True
        return False
    except Exception as e:
        st.error(f"Error revoking sessions: {e}")
        return False
    finally:
        client.close()

# =============================================================================
# PASSWORD RESET WORKFLOW
# =============================================================================
def generate_password_reset_token(email: str) -> str:
    """Generate JWT token for password reset with enhanced security."""
    try:
        client = MongoClient(MONGO_URI)
        user = client["CivicCatalyst"].users.find_one({"email": email})
        if not user:
            # Don't reveal if email exists for security
            return "dummy_token_for_nonexistent_email"
        
        # Generate a unique token ID
        token_id = str(uuid.uuid4())
        
        payload = {
            "sub": "password_reset",
            "email": email,
            "username": user["username"],
            "jti": token_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        # Record this reset token in the database
        client["CivicCatalyst"].password_resets.insert_one({
            "token_id": token_id,
            "email": email,
            "username": user["username"],
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=1),
            "used": False
        })
        
        # Log the password reset request
        client["CivicCatalyst"].activity_log.insert_one({
            "user": user["username"],
            "action": "password_reset_requested",
            "timestamp": datetime.utcnow(),
            "details": "Password reset link requested"
        })
        
        return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        st.error(f"Error generating reset token: {e}")
        return None
    finally:
        client.close()

def reset_password(token: str, new_password: str) -> bool:
    """Reset password using valid JWT token with improved security checks."""
    try:
        # Decode and verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Verify token type and expiration
        if payload.get("sub") != "password_reset":
            return False
            
        email = payload["email"]
        username = payload.get("username")
        token_id = payload.get("jti")
        
        # Check token in database
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Verify token hasn't been used already
        reset_record = db.password_resets.find_one({"token_id": token_id})
        if not reset_record or reset_record.get("used"):
            return False
            
        # Check password strength
        is_valid, msg, _ = check_password_strength(new_password)
        if not is_valid:
            st.error(msg)
            return False
            
        # Generate new password hash
        password_hash, salt = generate_password_hash(new_password)
        
        # Update the user's password
        result = db.users.update_one(
            {"email": email},
            {"$set": {
                "password_hash": password_hash,
                "password_salt": salt,
                "failed_attempts": 0,
                "locked_until": None,
                "security.password_last_changed": datetime.utcnow(),
                "security.password_expires": datetime.utcnow() + timedelta(days=90)
            }}
        )
        
        # Mark the token as used
        db.password_resets.update_one(
            {"token_id": token_id},
            {"$set": {"used": True}}
        )
        
        # Revoke all existing sessions for this user
        revoke_all_sessions(username)
        
        # Log the password reset success
        if result.modified_count > 0:
            db.activity_log.insert_one({
                "user": username,
                "action": "password_reset_complete",
                "timestamp": datetime.utcnow(),
                "details": "Password reset successful"
            })
            
        return result.modified_count > 0
    except jwt.ExpiredSignatureError:
        st.error("Password reset link has expired")
        return False
    except jwt.InvalidTokenError:
        st.error("Invalid reset token")
        return False
    except Exception as e:
        st.error(f"Error resetting password: {e}")
        return False
    finally:
        client.close()

# =============================================================================
# SESSION & LOGOUT
def logout_user(cookies):
    """Enhanced logout with proper session clearing."""
    try:
        # Get current session details
        username = st.session_state.get("username")
        
        # Clear session state
        for key in list(st.session_state.keys()):
            # Keep only language and theme preferences
            if key not in ["site_language", "theme"]:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Clear cookies
        if cookies:
            for key in ["jwt_token", "username", "role", "expires_at"]:
                if key in cookies:
                    cookies[key] = None
            cookies.save()
        
        # Show feedback
        st.success("Successfully logged out!")
        
        # Force refresh
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during logout: {e}")

def show_logout_button(cookies):
    """Improved logout button in sidebar using Streamlit's native button."""
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Logout", key="logout_button"):
        # Call logout function directly
        logout_user(cookies)
        return True
    
    return False

# =============================================================================
# ACTIVE SESSIONS MANAGEMENT
# =============================================================================
def display_active_sessions(username: str):
    """Enhanced display of user's active sessions with more details and improved UI."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Get current session ID for highlighting
        current_session_id = None
        if "jwt_token" in st.session_state:
            try:
                payload = jwt.decode(
                    st.session_state["jwt_token"], 
                    SECRET_KEY, 
                    algorithms=[JWT_ALGORITHM]
                )
                current_session_id = payload.get("jti")
            except:
                pass
                
        # Get all active sessions
        sessions = list(db.sessions.find({
            "username": username,
            "revoked": False,
            "expires": {"$gt": datetime.utcnow()}
        }).sort("last_active", -1))
        
        st.markdown("### 🔐 Active Sessions")
        
        if not sessions:
            st.info("No active sessions found.")
            return
            
        # Display sessions in a more elegant and informative way
        for i, session in enumerate(sessions):
            is_current = session.get("session_id") == current_session_id
            
            # Create a card for each session with enhanced styling
            with st.container():
                # Use custom HTML for prettier rendering
                session_html = f"""
                <div style="background: {'rgba(230, 255, 230, 0.7)' if is_current else 'rgba(255, 255, 255, 0.7)'};
                            border-radius: 0.7rem; padding: 1rem; margin-bottom: 1rem;
                            border-left: 4px solid {'#34A853' if is_current else '#4285F4'};
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: bold;">
                                {session.get('user_agent', 'Web Session')} 
                                {' (Current Session)' if is_current else ''}
                            </span>
                            <div style="font-size: 0.9rem; margin-top: 0.3rem; color: #666;">
                                <div>Login: {session.get('issued_at', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}</div>
                                <div>Last active: {session.get('last_active', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}</div>
                                <div>Expires: {session.get('expires', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}</div>
                            </div>
                        </div>
                        <div>
                            <div id="session_id_{i}" style="display: none;">{session.get('session_id', '')}</div>
                            <button onclick="revokeSession('session_id_{i}')"
                                    style="background: {'#DA4453' if not is_current else '#999'};
                                        color: white; border: none; border-radius: 0.4rem;
                                        padding: 0.3rem 0.7rem; cursor: pointer; font-size: 0.8rem;"
                                    {'disabled' if is_current else ''}>
                                {' ❌ Revoke' if not is_current else '(Current)'}
                            </button>
                        </div>
                    </div>
                </div>
                """
                st.markdown(session_html, unsafe_allow_html=True)
        
        # Add JavaScript for session revocation
        js_code = """
        <script>
        function revokeSession(id_element) {
            if (confirm('Are you sure you want to revoke this session?')) {
                const sessionId = document.getElementById(id_element).textContent;
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: sessionId,
                    dataType: 'str',
                    componentInstance: 'revoke_session'
                }, '*');
            }
        }
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)
        
        # Create a placeholder for the session ID to revoke
        if 'revoke_session' not in st.session_state:
            st.session_state.revoke_session = None
            
        # Check if a session should be revoked
        if st.session_state.revoke_session:
            session_id = st.session_state.revoke_session
            st.session_state.revoke_session = None  # Reset
            
            if revoke_session(session_id, username):
                st.success(f"Session revoked successfully!")
                st.rerun()
            else:
                st.error("Failed to revoke session.")
        
        # Add option to revoke all other sessions
        if len(sessions) > 1:
            if st.button("🔒 Revoke All Other Sessions"):
                if revoke_all_sessions(username, current_session_id):
                    st.success("All other sessions revoked!")
                    st.rerun()
                else:
                    st.error("Failed to revoke sessions.")
                    
    except Exception as e:
        st.error(f"Error retrieving sessions: {e}")

# =============================================================================
# PROFILE & FORGOT PASSWORD
# =============================================================================
def enhanced_user_profile(username: str, role: str):
    """Expanded profile with password change, session management, and preferences."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        user = db.users.find_one({"username": username})
        
        if not user:
            st.error(f"User profile not found")
            return
            
        # Profile tabs for better organization
        profile_tabs = st.tabs(["📋 Overview", "🔑 Security", "⚙️ Preferences", "📱 Sessions"])
        
        with profile_tabs[0]:
            # Overview Tab
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display profile image if available
                if "profile_image" in user:
                    try:
                        image_data = user["profile_image"]
                        if isinstance(image_data, bytes):
                            img = Image.open(BytesIO(image_data))
                            st.image(img, width=150, caption="Profile Photo")
                    except Exception as e:
                        st.error(f"Error displaying profile image: {e}")
                else:
                    # Display a default profile icon
                    profile_animation = load_lottie_url(LOTTIE_ANIMATIONS["profile"])
                    if profile_animation:
                        st_lottie(profile_animation, height=150, key="profile_animation")
                    else:
                        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", 
                                width=150, caption="Default Avatar")
            
            with col2:
                st.markdown(f"""
                ### {username}
                **Role:** {role}
                
                **Email:** {user.get('email', 'Not set')}
                **CIN:** {user.get('cin', 'Not provided')}
                **Account Created:** {user.get('created_at', 'Unknown').strftime('%Y-%m-%d')}
                **Last Login:** {user.get('profile', {}).get('last_login', 'Never').strftime('%Y-%m-%d %H:%M:%S') if user.get('profile', {}).get('last_login') else 'Never'}
                **Login Count:** {user.get('profile', {}).get('login_count', 0)}
                """)
                
            # Profile image upload
            st.subheader("Update Profile Picture")
            uploaded_file = st.file_uploader("Upload a profile image (JPEG/PNG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                try:
                    image_bytes = uploaded_file.read()
                    img = Image.open(BytesIO(image_bytes))
                    
                    # Resize image if too large
                    max_size = (400, 400)
                    img.thumbnail(max_size, Image.LANCZOS)
                    
                    # Convert back to bytes
                    buf = BytesIO()
                    img.save(buf, format=img.format or "JPEG")
                    processed_image = buf.getvalue()
                    
                    if st.button("Save Profile Picture"):
                        db.users.update_one(
                            {"username": username},
                            {"$set": {"profile_image": processed_image}}
                        )
                        st.success(t("profile_image_saved"))
                        # Show a preview of the uploaded image
                        st.image(img, width=150, caption="New Profile Picture")
                        
                        # Log the action
                        db.activity_log.insert_one({
                            "user": username,
                            "action": "profile_image_updated",
                            "timestamp": datetime.utcnow(),
                            "details": "Profile image updated"
                        })
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        
        with profile_tabs[1]:
            # Security Tab
            st.markdown("### 🔐 Security Settings")
            
            # Password change form
            with st.form("change_password_form"):
                st.markdown("#### " + t("change_password"))
                old_pass = st.text_input("Current Password", type="password")
                new_pass = st.text_input("New Password", type="password", key="new_password")
                confirm_pass = st.text_input("Confirm New Password", type="password")
                
                # Password strength meter
                if new_pass:
                    is_valid, msg, strength = check_password_strength(new_pass)
                    
                    # Determine strength class and color
                    strength_class = ""
                    strength_text = ""
                    
                    if strength <= 1:
                        strength_class = "strength-weak"
                        strength_text = t("weak")
                    elif strength <= 2:
                        strength_class = "strength-medium"
                        strength_text = t("medium")
                    else:
                        strength_class = "strength-strong"
                        strength_text = t("strong")
                    
                    # Display password strength meter
                    st.markdown(f"""
                    <div class="password-strength {strength_class}" style="width: {25 * strength}%;"></div>
                    <div class="strength-text">{t('password_strength')}: {strength_text}</div>
                    """, unsafe_allow_html=True)
                
                submit_button = st.form_submit_button("Update Password")
                
                if submit_button:
                    if new_pass != confirm_pass:
                        st.error("New passwords do not match")
                    else:
                        # Verify current password
                        valid_login, _, msg = verify_user(username, old_pass)
                        if not valid_login:
                            st.error(msg or "Invalid current password")
                        else:
                            # Check password strength
                            is_valid, msg, strength = check_password_strength(new_pass)
                            if not is_valid:
                                st.error(msg)
                            else:
                                try:
                                    # Generate new password hash
                                    password_hash, salt = generate_password_hash(new_pass)
                                    
                                    # Update password in database
                                    db.users.update_one(
                                        {"username": username},
                                        {"$set": {
                                            "password_hash": password_hash,
                                            "password_salt": salt,
                                            "security.password_last_changed": datetime.utcnow(),
                                            "security.password_expires": datetime.utcnow() + timedelta(days=90)
                                        }}
                                    )
                                    
                                    # Log the action
                                    db.activity_log.insert_one({
                                        "user": username,
                                        "action": "password_changed",
                                        "timestamp": datetime.utcnow(),
                                        "details": "Password changed successfully"
                                    })
                                    
                                    st.success(t("change_password_success"))
                                except Exception as e:
                                    st.error(f"Error updating password: {e}")
            
            # Two-factor authentication (placeholder)
            st.markdown("#### 🔐 Two-Factor Authentication")
            
            two_factor_enabled = user.get("security", {}).get("2fa_enabled", False)
            
            if two_factor_enabled:
                st.success("Two-factor authentication is enabled")
                if st.button("Disable 2FA"):
                    # Placeholder for 2FA disabling
                    st.warning("This feature is not implemented yet")
            else:
                st.warning("Two-factor authentication is not enabled")
                if st.button("Enable 2FA"):
                    # Placeholder for 2FA setup
                    st.warning("This feature is not implemented yet")
                    
            # Security audit log
            st.markdown("#### 🔍 Recent Security Activity")
            security_logs = list(db.activity_log.find(
                {"user": username, "action": {"$in": ["login", "login_failed", "password_changed", "password_reset_requested", "account_locked"]}}
            ).sort("timestamp", -1).limit(5))
            
            if security_logs:
                for log in security_logs:
                    icon = "✅" if log["action"] == "login" else "⚠️"
                    st.markdown(f"""
                    {icon} **{log["action"].replace('_', ' ').title()}** - {log["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}  
                    {log.get("details", "")}
                    """)
            else:
                st.info("No recent security activity")
                
        with profile_tabs[2]:
            # Preferences Tab
            st.markdown("### ⚙️ Preferences")
            
            # Get current preferences
            current_preferences = user.get("profile", {}).get("preferences", {})
            
            # Language selection
            st.markdown("#### 🌐 Language")
            lang_options = {
                "en": "English",
                "fr": "Français",
                "ar": "العربية",
                "darija": "الدارجة المغربية"
            }
            
            current_lang = st.session_state.get("site_language", current_preferences.get("language", "en"))
            
            # Use a more visual language selector
            cols = st.columns(4)
            selected_lang = current_lang
            
            for i, (code, name) in enumerate(lang_options.items()):
                with cols[i]:
                    # Create styled button for each language
                    is_selected = code == current_lang
                    button_style = f"""
                    background: {'var(--primary-color)' if is_selected else 'rgba(255,255,255,0.7)'};
                    color: {'white' if is_selected else 'var(--text-color)'};
                    border: {'none' if is_selected else '1px solid #ccc'};
                    border-radius: 0.5rem;
                    padding: 0.5rem;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.2s;
                    """
                    
                    html = f"""
                    <div class="language-option" style="{button_style}" 
                         onclick="selectLanguage('{code}')">
                        {name}
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
            
            # Add JavaScript to handle language selection
            js_code = """
            <script>
            function selectLanguage(lang) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: lang,
                    dataType: 'str',
                    componentInstance: 'selected_language'
                }, '*');
            }
            </script>
            """
            st.markdown(js_code, unsafe_allow_html=True)
            
            # Create a placeholder for the selected language
            if 'selected_language' not in st.session_state:
                st.session_state.selected_language = None
                
            # Check if a language was selected
            if st.session_state.selected_language:
                selected_lang = st.session_state.selected_language
                st.session_state.selected_language = None  # Reset
                
                # Update language preference
                if selected_lang != current_lang:
                    # Update in session state
                    st.session_state["site_language"] = selected_lang
                    
                    # Update in database
                    db.users.update_one(
                        {"username": username},
                        {"$set": {"profile.preferences.language": selected_lang}}
                    )
                    
                    # Show success message with the new language
                    st.success(f"{t('language_updated')} {lang_options.get(selected_lang, selected_lang)}")
                    st.rerun()
            
            # Theme selection
            st.markdown("#### 🎨 Theme")
            
            current_theme = st.session_state.get("theme", current_preferences.get("theme", DEFAULT_THEME))
            
            # Use color swatches for theme selection
            theme_cols = st.columns(len(AVAILABLE_THEMES))
            
            for i, theme_name in enumerate(AVAILABLE_THEMES):
                with theme_cols[i]:
                    colors = get_theme_colors(theme_name)
                    
                    is_selected = theme_name == current_theme
                    
                    # Create a color swatch with the theme's primary color
                    swatch_html = f"""
                    <div onclick="selectTheme('{theme_name}')" style="
                         background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
                         width: 100%; height: 40px; border-radius: 5px;
                         border: {f'3px solid #333' if is_selected else 'none'};
                         cursor: pointer; margin-bottom: 5px;"></div>
                    <div style="text-align: center; font-size: 0.9rem;
                         font-weight: {700 if is_selected else 400};">{theme_name.capitalize()}</div>
                    """
                    st.markdown(swatch_html, unsafe_allow_html=True)
            
            # Add JavaScript for theme selection
            js_code = """
            <script>
            function selectTheme(theme) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: theme,
                    dataType: 'str',
                    componentInstance: 'selected_theme'
                }, '*');
            }
            </script>
            """
            st.markdown(js_code, unsafe_allow_html=True)
            
            # Create a placeholder for the selected theme
            if 'selected_theme' not in st.session_state:
                st.session_state.selected_theme = None
                
            # Check if a theme was selected
            if st.session_state.selected_theme:
                selected_theme = st.session_state.selected_theme
                st.session_state.selected_theme = None  # Reset
                
                # Update theme preference
                if selected_theme != current_theme:
                    # Update in session state
                    st.session_state["theme"] = selected_theme
                    
                    # Update in database
                    db.users.update_one(
                        {"username": username},
                        {"$set": {"profile.preferences.theme": selected_theme}}
                    )
                    
                    # Apply the new theme
                    set_theme_in_storage(selected_theme)
                    st.success(f"Theme updated to: {selected_theme.capitalize()}")
                    st.rerun()
            
            # Voice feedback preference
            st.markdown("#### 🔊 Accessibility")
            
            voice_feedback = st.checkbox(
                t("voice_feedback"),
                value=st.session_state.get("voice_feedback", current_preferences.get("voice_feedback", False))
            )
            
            if voice_feedback != st.session_state.get("voice_feedback", current_preferences.get("voice_feedback", False)):
                # Update in session state
                st.session_state["voice_feedback"] = voice_feedback
                
                # Update in database
                db.users.update_one(
                    {"username": username},
                    {"$set": {"profile.preferences.voice_feedback": voice_feedback}}
                )
                
                st.success("Voice feedback settings updated!")
                
                # Demonstrate voice feedback
                if voice_feedback:
                    lang = st.session_state.get("site_language", "en")
                    demo_text = {
                        "en": "Voice feedback is now enabled",
                        "fr": "Le retour vocal est maintenant activé",
                        "ar": "تم تفعيل التعليق الصوتي",
                        "darija": "تم تفعيل التعليق الصوتي"
                    }
                    audio_html = text_to_speech(
                        demo_text.get(lang, demo_text["en"]),
                        "en" if lang not in ["ar", "fr"] else lang
                    )
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
        
        with profile_tabs[3]:
            # Sessions Tab
            display_active_sessions(username)
            
    except Exception as e:
        st.error(f"Error loading profile: {e}")

def display_password_reset_form():
    """Enhanced password reset form with animated visuals and feedback."""
    with st.expander(t("forgot_password"), expanded=False):
        # Load a relevant animation
        reset_animation = load_lottie_url(LOTTIE_ANIMATIONS["profile"])
        if reset_animation:
            st_lottie(reset_animation, height=120, key="reset_animation")
            
        # More informative instructions
        st.markdown("""
        ### Reset Your Password
        
        Enter your email address below. If an account exists with this email,
        we'll send instructions on how to reset your password.
        """)
        
        with st.form("password_reset_form"):
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            submit = st.form_submit_button("Send Reset Instructions")
            
            if submit:
                with st.spinner("Processing your request..."):
                    token = generate_password_reset_token(email)
                    if token:
                        # Show success message with token for demo purposes
                        # In production this would be emailed
                        st.success("If an account exists with this email, password reset instructions have been sent.")
                        st.info(f"For demonstration purposes, here's your reset token: {token}")
                        
                        # Generate reset link
                        reset_link = f"?token={token}"
                        st.markdown(f"[Click here to reset your password]({reset_link})")
                    else:
                        # Don't reveal if email exists - security best practice
                        st.success("If an account exists with this email, password reset instructions have been sent.")

# =============================================================================
# REGISTRATION FORM
# =============================================================================
def enhanced_registration_form():
    """Registration form with animations, improved UI, and enhanced security features."""
    with st.expander(t("register_header"), expanded=False):
        # Add a welcoming animation
        welcome_animation = load_lottie_url(LOTTIE_ANIMATIONS["welcome"])
        if welcome_animation:
            st_lottie(welcome_animation, height=150, key="register_animation")
            
        st.markdown("""
        ### Join Civic Catalyst
        
        Create your account to access the full range of Civic Catalyst features. Your participation
        helps improve your community through intelligent citizen engagement.
        """)
        
        with st.form("registration_form", clear_on_submit=True):
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            new_username = st.text_input("Choose a Username", placeholder="Between 3-20 characters")
            
            # Password input with strength meter
            new_password = st.text_input("Create Password", type="password", key="reg_password")
            
            # Show password strength meter
            if new_password:
                is_valid, msg, strength = check_password_strength(new_password)
                
                # Determine strength class and color
                strength_class = ""
                strength_text = ""
                
                if strength <= 1:
                    strength_class = "strength-weak"
                    strength_text = t("weak")
                elif strength <= 2:
                    strength_class = "strength-medium"
                    strength_text = t("medium")
                else:
                    strength_class = "strength-strong"
                    strength_text = t("strong")
                
                # Display password strength meter with percentage
                st.markdown(f"""
                <div class="password-strength {strength_class}" style="width: {25 * strength}%;"></div>
                <div class="strength-text">{t('password_strength')}: {strength_text}</div>
                <div style="font-size: 0.8rem; margin-top: 0.2rem; color: #666;">{msg}</div>
                """, unsafe_allow_html=True)
            
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Additional fields
            col1, col2 = st.columns(2)
            with col1:
                new_cin = st.text_input(t("cin"), placeholder="e.g., A111981")
            with col2:
                role_choice = st.selectbox("Select Role", ["citizen", "moderator", "admin"])
                
            # Terms and privacy policy
            tos_accept = st.checkbox("I accept the Terms of Service and Privacy Policy", value=False)
            
            # Optional API key
            st.markdown("---")
            api_key = st.text_input(
                t("gpt_key_prompt"), 
                placeholder="sk-...", 
                help="This is optional. You can use your own GPT API key for some features."
            )
            
            # Submit button with loading animation
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if not tos_accept:
                    st.error("You must accept the Terms of Service to register.")
                    return
                    
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                    return
                    
                is_valid, msg, _ = check_password_strength(new_password)
                if not is_valid:
                    st.error(msg)
                    return
                
                # Show a spinner while processing
                with st.spinner("Creating your account..."):
                    # Simulate a brief delay for better UX
                    time.sleep(1)
                    
                    # Attempt creation
                    if create_user(new_username, new_password, email, role_choice, new_cin):
                        # Success animation
                        success_animation = load_lottie_url(LOTTIE_ANIMATIONS["success"])
                        if success_animation:
                            st_lottie(success_animation, height=120, key="success_animation")
                            
                        st.success("Registration successful! You can now log in.")
                        
                        # Store API key in session state if provided
                        if api_key and api_key.startswith("sk-"):
                            st.session_state["user_api_key"] = api_key
                            
                        # Add a slight delay for better experience
                        time.sleep(2)
                        st.rerun()

# =============================================================================
# LOGIN FLOW
# =============================================================================
def enhanced_login_flow(cookies):
    """Login form with improved UI, animations, and enhanced persistence."""
    st.subheader(t("login_title"))
    
    # Display login-related animation
    display_login_animation()
    
    # Create a form with enhanced styling
    with st.form("login_form", clear_on_submit=True):
        # Username input
        username = st.text_input(
            t("username"),
            placeholder="Enter your username"
        )
        
        # Password input
        password = st.text_input(
            t("password"),
            type="password",
            placeholder="Enter your password"
        )
        
        # Options row
        col1, col2 = st.columns(2)
        
        with col1:
            remember_me = st.checkbox(
                t("remember_me"), 
                value=True,
                help=f"Stay logged in for {SESSION_EXPIRY} days"
            )
        
        with col2:
            stay_signed_in = st.checkbox(
                t("stay_signed_in"),
                value=True,
                help="Keep your session active even when closing browser"
            )
        
        # Submit button
        login_button = st.form_submit_button(t("login_button"))
        
        if login_button:
            if not username or not password:
                st.error("Please enter both username and password.")
                return
                
            # Show loading spinner while authenticating
            with st.spinner("Authenticating..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                valid, role, message = verify_user(username, password)
                if valid:
                    # Create token with extended expiry if remember_me is checked
                    token = create_jwt_token(username, role, remember_me)
                    
                    if token:
                        # Success animation
                        success_animation = load_lottie_url(LOTTIE_ANIMATIONS["success"])
                        if success_animation:
                            st_lottie(success_animation, height=120, key="login_success")
                            
                        # Log the successful login
                        log_login_event(username)
                        
                        # Create session record in DB with persistence options
                        try:
                            client = MongoClient(MONGO_URI)
                            db = client["CivicCatalyst"]
                            
                            # Decode token to get session ID
                            payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
                            session_id = payload.get("jti")
                            
                            # Update session with persistence preference
                            db.sessions.update_one(
                                {"session_id": session_id},
                                {"$set": {"stay_signed_in": stay_signed_in}}
                            )
                        except Exception as e:
                            st.error(f"Error updating session record: {e}")
                        
                        # Save user preferences in session state
                        try:
                            user = db.users.find_one({"username": username})
                            if user and "profile" in user and "preferences" in user["profile"]:
                                prefs = user["profile"]["preferences"]
                                if "language" in prefs:
                                    st.session_state["site_language"] = prefs["language"]
                                if "theme" in prefs:
                                    st.session_state["theme"] = prefs["theme"]
                                if "voice_feedback" in prefs:
                                    st.session_state["voice_feedback"] = prefs["voice_feedback"]
                        except Exception as e:
                            # Non-critical error, continue with defaults
                            pass
                        
                        # Compute expiration date as an ISO formatted string
                        expires_date = (datetime.utcnow() + timedelta(days=SESSION_EXPIRY if remember_me else 1)).isoformat()
                        
                        # Save token in session_state
                        st.session_state["jwt_token"] = token
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        
                        # Save token, username, role, and expiration in cookies
                        cookies["jwt_token"] = token
                        cookies["username"] = username
                        cookies["role"] = role
                        cookies["expires_at"] = expires_date
                        cookies.save()
                        
                        # Also save in localStorage for better persistence
                        if stay_signed_in:
                            js_code = f"""
                            <script>
                                localStorage.setItem('civic_jwt_token', '{token}');
                                localStorage.setItem('civic_username', '{username}');
                                localStorage.setItem('civic_role', '{role}');
                                localStorage.setItem('civic_expires', '{expires_date}');
                                console.log('Session saved in localStorage');
                            </script>
                            """
                            st.markdown(js_code, unsafe_allow_html=True)

                        # Display welcome message with voice if enabled
                        welcome_text = f"{t('welcome_back')}, {username}!"
                        st.success(welcome_text)
                        
                        if st.session_state.get("voice_feedback", False):
                            lang = st.session_state.get("site_language", "en")
                            audio_html = text_to_speech(
                                welcome_text,
                                "en" if lang not in ["ar", "fr"] else lang
                            )
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Add a slight delay for better experience
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Error creating session token.")
                else:
                    # Error animation
                    error_animation = load_lottie_url(LOTTIE_ANIMATIONS["error"])
                    if error_animation:
                        st_lottie(error_animation, height=100, key="login_error")
                        
                    st.error(message or "Invalid credentials.")

def check_for_stored_session():
    """Check localStorage for saved session data and restore if valid."""
    js_code = """
    <script>
        // Check if we have a stored session
        const token = localStorage.getItem('civic_jwt_token');
        const username = localStorage.getItem('civic_username');
        const role = localStorage.getItem('civic_role');
        const expires = localStorage.getItem('civic_expires');
        
        if (token && username && role && expires) {
            // Check if the token is still valid (not expired)
            const expiryDate = new Date(expires);
            const now = new Date();
            
            if (expiryDate > now) {
                // Send the session data back to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: JSON.stringify({token, username, role}),
                    dataType: 'str',
                    componentInstance: 'stored_session'
                }, '*');
            } else {
                // Clear expired session
                localStorage.removeItem('civic_jwt_token');
                localStorage.removeItem('civic_username');
                localStorage.removeItem('civic_role');
                localStorage.removeItem('civic_expires');
            }
        }
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    
    # Create a placeholder for the session data
    if 'stored_session' not in st.session_state:
        st.session_state.stored_session = None
        
    # Check if we received session data from localStorage
    if st.session_state.stored_session:
        try:
            session_data = json.loads(st.session_state.stored_session)
            token = session_data.get('token')
            username = session_data.get('username')
            role = session_data.get('role')
            
            # Verify the token
            valid, username_jwt, role_jwt = verify_jwt_token(token)
            
            # If token is valid and matches the stored username/role
            if valid and username_jwt == username and role_jwt == role:
                # Restore the session
                st.session_state["jwt_token"] = token
                st.session_state["username"] = username
                st.session_state["role"] = role
                
                # Also update cookies for consistency
                cookies = get_cookies()
                if cookies and cookies.ready():
                    cookies["jwt_token"] = token
                    cookies["username"] = username
                    cookies["role"] = role
                    cookies.save()
                
                return True
        except Exception as e:
            # If there's any error, clear the stored session
            js_clear = """
            <script>
                localStorage.removeItem('civic_jwt_token');
                localStorage.removeItem('civic_username');
                localStorage.removeItem('civic_role');
                localStorage.removeItem('civic_expires');
            </script>
            """
            st.markdown(js_clear, unsafe_allow_html=True)
            
    return False

# Replace the get_cookies function with this simpler version
def get_cookies():
    """Get auth manager instance"""
    return SessionAuth(prefix="civic_")

def handle_existing_session(cookies):
    """
    Check for existing sessions in session state.
    """
    # Check session state first (already authenticated in this session)
    if "username" in st.session_state and "role" in st.session_state:
        username = st.session_state["username"]
        role = st.session_state["role"]
        
        # Show the authenticated interface
        show_authenticated_interface(username, role, cookies)
        return True
    
    # No valid session found
    return False

# =============================================================================
# AUTHENTICATED INTERFACE
# =============================================================================
def show_authenticated_interface(username: str, role: str, cookies):
    """Display enhanced interface for logged-in users with animations and theme support."""
    # Apply the selected theme
    apply_theme()
    
    # Display logout button in sidebar
    show_logout_button(cookies)
    
    # Welcome header with user info
    st.markdown(f"""
    <div class="profile-header">
        <h2>👋 {t('welcome_back')}, {username}!</h2>
        <span class="badge">{role.upper() if role else "GUEST"}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation using tabs
    tabs = st.tabs(["📋 Dashboard", "👤 Profile", "🔧 Tools", "⚙️ Settings"])
    
    with tabs[0]:
        # Dashboard Tab
        st.markdown("### 📊 Your Dashboard")
        
        # Placeholder dashboard content
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Recent Activity
            - Last login: Recently
            - Completed actions: 0
            - Pending tasks: 0
            """)
            
        with col2:
            st.markdown("""
            #### System Status
            - All systems operational ✅
            - Database connected ✅
            - API services available ✅
            """)
        
        # Demo data visualization
        st.markdown("### 📈 Participation Metrics")
        
        # Sample chart data
        chart_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='ME'),
            'contributions': [3, 5, 2, 7, 10, 8, 12, 15, 10, 9],
            'feedback': [5, 8, 7, 9, 12, 15, 17, 18, 14, 12]
        })
        
        try:
            import plotly.express as px
            fig = px.line(
                chart_data, 
                x='date', 
                y=['contributions', 'feedback'],
                title='Community Participation Over Time',
                labels={'value': 'Count', 'variable': 'Metric', 'date': 'Month'},
                color_discrete_sequence=[
                    get_theme_colors(st.session_state.get("theme", DEFAULT_THEME))["primary"],
                    get_theme_colors(st.session_state.get("theme", DEFAULT_THEME))["secondary"]
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            # Fallback if Plotly is not available
            st.line_chart(chart_data.set_index('date'))
    
    with tabs[1]:
        # Profile Tab
        enhanced_user_profile(username, role)
    
    with tabs[2]:
        # Tools Tab
        st.markdown("### 🔧 Available Tools")
        
        tool_cols = st.columns(3)
        
        with tool_cols[0]:
            st.markdown("""
            #### 📝 Feedback Submission
            Submit feedback on civic projects and initiatives.
            """)
            if st.button("Open Feedback Tool"):
                st.session_state["current_tool"] = "feedback"
                st.rerun()
                
        with tool_cols[1]:
            st.markdown("""
            #### 🔍 Project Explorer
            Browse and search municipal projects.
            """)
            if st.button("Open Project Explorer"):
                st.session_state["current_tool"] = "projects"
                st.rerun()
                
        with tool_cols[2]:
            st.markdown("""
            #### 📊 Data Analyzer
            Analyze civic data and trends.
            """)
            if st.button("Open Data Analyzer"):
                st.session_state["current_tool"] = "analyzer"
                st.rerun()
    
    with tabs[3]:
        # Settings Tab
        st.markdown("### ⚙️ System Settings")
        
        # Theme settings
        st.subheader(t("theme_settings"))
        
        current_theme = st.session_state.get("theme", DEFAULT_THEME)
        
        # Use color swatches for theme selection
        theme_cols = st.columns(len(AVAILABLE_THEMES))
        
        for i, theme_name in enumerate(AVAILABLE_THEMES):
            with theme_cols[i]:
                colors = get_theme_colors(theme_name)
                
                is_selected = theme_name == current_theme
                
                # Create a color swatch with the theme's primary color
                swatch_html = f"""
                <div onclick="selectTheme('{theme_name}')" style="
                     background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
                     width: 100%; height: 40px; border-radius: 5px;
                     border: {f'3px solid #333' if is_selected else 'none'};
                     cursor: pointer; margin-bottom: 5px;"></div>
                <div style="text-align: center; font-size: 0.9rem;
                     font-weight: {700 if is_selected else 400};">{theme_name.capitalize()}</div>
                """
                st.markdown(swatch_html, unsafe_allow_html=True)
        
        # Add JavaScript for theme selection
        js_code = """
        <script>
        function selectTheme(theme) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: theme,
                dataType: 'str',
                componentInstance: 'selected_theme'
            }, '*');
        }
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)
        
        # Check if a theme was selected
        if st.session_state.get('selected_theme'):
            selected_theme = st.session_state.selected_theme
            st.session_state.selected_theme = None  # Reset
            
            # Update theme preference
            if selected_theme != current_theme:
                # Update in session state
                st.session_state["theme"] = selected_theme
                
                # Save in database
                try:
                    client = MongoClient(MONGO_URI)
                    db = client["CivicCatalyst"]
                    db.users.update_one(
                        {"username": username},
                        {"$set": {"profile.preferences.theme": selected_theme}}
                    )
                except Exception as e:
                    st.error(f"Error saving theme preference: {e}")
                
                # Apply the new theme
                set_theme_in_storage(selected_theme)
                st.success(f"Theme updated to: {selected_theme.capitalize()}")
                st.rerun()
                
        # API Integration settings
        st.subheader("API Integration")
        
        current_api_key = st.session_state.get("user_api_key", "")
        
        # Mask the API key for display
        masked_key = "••••••••" + current_api_key[-4:] if current_api_key else ""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_api_key = st.text_input(
                "API Key",
                value=masked_key if masked_key else "",
                type="password" if masked_key else "default",
                placeholder="Enter your API key (sk-...)",
                help="Your OpenAI API key for enhanced features"
            )
        
        with col2:
            if st.button("Save API Key"):
                if new_api_key and new_api_key != masked_key:
                    # Save in session state
                    st.session_state["user_api_key"] = new_api_key
                    
                    # Store in database (securely)
                    try:
                        client = MongoClient(MONGO_URI)
                        db = client["CivicCatalyst"]
                        db.users.update_one(
                            {"username": username},
                            {"$set": {"api_key": new_api_key}}
                        )
                        
                        # Log the action (without the actual key)
                        db.activity_log.insert_one({
                            "user": username,
                            "action": "api_key_updated",
                            "timestamp": datetime.utcnow(),
                            "details": "API key updated"
                        })
                        
                        st.success("API key saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving API key: {e}")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        # Export user data
        if st.button("Export My Data"):
            try:
                client = MongoClient(MONGO_URI)
                db = client["CivicCatalyst"]
                
                # Get user data (excluding sensitive fields)
                user_data = db.users.find_one(
                    {"username": username},
                    {"password_hash": 0, "password_salt": 0, "api_key": 0}
                )
                
                if user_data:
                    # Convert ObjectId to string for JSON serialization
                    user_data["_id"] = str(user_data["_id"])
                    
                    # Get user's activity log
                    activity_log = list(db.activity_log.find(
                        {"user": username},
                        {"_id": 0}
                    ))
                    
                    # Format timestamps for JSON serialization
                    for entry in activity_log:
                        if "timestamp" in entry:
                            entry["timestamp"] = entry["timestamp"].isoformat()
                    
                    # Combine data
                    export_data = {
                        "user_profile": user_data,
                        "activity_log": activity_log
                    }
                    
                    # Convert to JSON
                    json_data = json.dumps(export_data, indent=2)
                    
                    # Offer download
                    st.download_button(
                        label="Download Data (JSON)",
                        data=json_data,
                        file_name=f"civic_catalyst_data_{username}.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Error exporting data: {e}")
        
        # Account deletion option (with strong warning)
        st.subheader("⚠️ Danger Zone")
        
        with st.expander("Delete My Account"):
            st.warning("""
            **Warning**: Account deletion is permanent and cannot be undone.
            All your data will be permanently removed from our systems.
            """)
            
            confirm_username = st.text_input(
                "Confirm your username",
                placeholder="Type your username to confirm deletion"
            )
            
            delete_confirmed = st.checkbox("I understand this action is permanent")
            
            if st.button("Delete My Account", type="primary", disabled=not delete_confirmed):
                if confirm_username == username and delete_confirmed:
                    try:
                        client = MongoClient(MONGO_URI)
                        db = client["CivicCatalyst"]
                        
                        # Delete user data
                        db.users.delete_one({"username": username})
                        
                        # Delete sessions
                        db.sessions.delete_many({"username": username})
                        
                        # Log the action (separate from user account)
                        db.deleted_accounts.insert_one({
                            "username": username,
                            "deleted_at": datetime.utcnow(),
                            "reason": "User requested account deletion"
                        })
                        
                        # Clear session and logout
                        logout_user(cookies)
                        
                    except Exception as e:
                        st.error(f"Error deleting account: {e}")
                else:
                    st.error("Username doesn't match or confirmation not checked")

    # Footer with app information
    st.markdown("""
    <div class="footer">
        <p>Civic Catalyst Platform © 2025 | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
        <p>Version 2.0.1 | Powered by Streamlit and MongoDB</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PUBLIC (UNAUTHENTICATED) INTERFACE
# =============================================================================
def show_login_interface(cookies):
    """Display enhanced login/registration/password-reset interface with animations."""
    # Apply the selected theme
    apply_theme()
    
    # Display main header with animation
    display_main_header_with_animation()
    
    # Try to load language from cookie
    get_language_from_cookie(cookies)
    
    # Language selector at the top
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # Language options
    lang_options = {
        "en": "🇬🇧 English",
        "fr": "🇫🇷 Français",
        "ar": "🇲🇦 العربية",
        "darija": "🇲🇦 الدارجة"
    }
    
    current_lang = st.session_state.get("site_language", "en")
    
    # Create a horizontal language selector
    cols = st.columns(len(lang_options))
    
    for i, (code, name) in enumerate(lang_options.items()):
        with cols[i]:
            if st.button(name, key=f"lang_{code}"):
                set_language_in_cookie(cookies, code)
                st.success(f"{t('language_updated')} {name}")
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content with elevated UI
    with st.container():
        # Create tabs for different authentication options
        auth_tabs = st.tabs(["🔑 Login", "📝 Register", "🔒 Reset Password"])
        
        with auth_tabs[0]:
            # Login tab
            enhanced_login_flow(cookies)
            
        with auth_tabs[1]:
            # Registration tab
            enhanced_registration_form()
            
        with auth_tabs[2]:
            # Password reset tab
            display_password_reset_form()
    
    # Footer with app information
    st.markdown("""
    <div class="footer">
        <p>Civic Catalyst Platform © 2023 | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
        <p>Version 2.0.1 | Help: contact@civiccatalyst.org</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TOKEN RESET INTERFACE
# =============================================================================
def show_reset_password_interface(token, cookies):
    """Display interface for resetting password with a valid token."""
    # Apply theme
    apply_theme()
    
    st.markdown(f"""
    <div class="main-login-container">
        <h1 class="login-title">{t('reset_password_prompt')}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load a relevant animation
    reset_animation = load_lottie_url(LOTTIE_ANIMATIONS["profile"])
    if reset_animation:
        st_lottie(reset_animation, height=150, key="reset_page_animation")
    
    with st.form("reset_password_page_form"):
        st.subheader("Create a New Password")
        
        # Password with strength meter
        new_pass = st.text_input("New Password", type="password", key="reset_new_pass")
        
        # Show password strength meter
        if new_pass:
            is_valid, msg, strength = check_password_strength(new_pass)
            
            # Determine strength class and color
            strength_class = ""
            strength_text = ""
            
            if strength <= 1:
                strength_class = "strength-weak"
                strength_text = t("weak")
            elif strength <= 2:
                strength_class = "strength-medium"
                strength_text = t("medium")
            else:
                strength_class = "strength-strong"
                strength_text = t("strong")
            
            # Display password strength meter
            st.markdown(f"""
            <div class="password-strength {strength_class}" style="width: {25 * strength}%;"></div>
            <div class="strength-text">{t('password_strength')}: {strength_text}</div>
            """, unsafe_allow_html=True)
            
        confirm_pass = st.text_input("Confirm Password", type="password")
        
        if st.form_submit_button("Reset My Password"):
            if new_pass != confirm_pass:
                st.error("Passwords do not match")
            else:
                is_valid, msg, _ = check_password_strength(new_pass)
                if not is_valid:
                    st.error(msg)
                else:
                    with st.spinner("Resetting your password..."):
                        if reset_password(token, new_pass):
                            # Success animation
                            success_animation = load_lottie_url(LOTTIE_ANIMATIONS["success"])
                            if success_animation:
                                st_lottie(success_animation, height=120, key="reset_success")
                                
                            st.success(t("password_reset_success"))
                            
                            # Redirect to login page after a delay
                            time.sleep(2)
                            st.markdown("""
                            <script>
                                window.location.href = './';
                            </script>
                            """, unsafe_allow_html=True)
                        else:
                            # Error animation
                            error_animation = load_lottie_url(LOTTIE_ANIMATIONS["error"])
                            if error_animation:
                                st_lottie(error_animation, height=100, key="reset_error")
                                
                            st.error(t("password_reset_failed"))

    # Link to go back to login
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="./" style="text-decoration: none; color: var(--primary-color);">
            ← Back to Login
        </a>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Set page config
    check_required_packages()
    
    # Initialize theme default if not set
    if "theme" not in st.session_state:
        st.session_state["theme"] = DEFAULT_THEME
    
    # Get theme from storage if available
    get_theme_from_storage()
    
    # Apply the selected theme
    apply_theme()
    
    # Check MongoDB connection and initialize collections
    db_connected = init_auth()
    if not db_connected:
        st.error("Failed to connect to the database. Some features may not work correctly.")
    
    # Import streamlit_components for advanced UI (only needed at runtime)
    import streamlit.components.v1 as components
    
    # 1) Initialize cookie manager
    cookies = get_cookies()
    

    
    # Check query parameters for token (password reset flow)
    params = st.query_params
    reset_token = None
    
    if "token" in params:
        token_val = params["token"]
        reset_token = token_val[0] if isinstance(token_val, list) else token_val
        
        if reset_token:
            # Show password reset interface
            show_reset_password_interface(reset_token, cookies)
            return
    
    # Check if logged out parameter exists
    if "logged_out" in params:
        show_login_interface(cookies)
        return
    
    # Check for existing session (in any storage mechanism)
    if not handle_existing_session(cookies):
        # If no valid session found, show login interface
        show_login_interface(cookies)

if __name__ == "__main__":
    main()
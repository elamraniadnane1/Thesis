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
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

# =============================================================================
# GLOBAL CONFIG & CONSTANTS
# =============================================================================
SECRET_KEY = os.environ.get("SECRET_KEY", "mysecretkey")
JWT_ALGORITHM = "HS256"
COOKIE_PASSWORD = os.environ.get("COOKIE_PASSWORD", "YOUR_STRONG_PASSWORD")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

MAX_LOGIN_ATTEMPTS = 3  # Lock out user after 3 failed attempts
LOCKOUT_TIME = 15       # Lockout duration (minutes)

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
    }
}


def t(key):
    """Translate text based on current session language."""
    lang = st.session_state.get("site_language", "en")
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("en", key))

def set_language_in_cookie(cookies, language):
    """Store chosen language in cookie and session state."""
    st.session_state["site_language"] = language
    cookies["site_language"] = language
    cookies.save()

def get_language_from_cookie(cookies):
    """Retrieve language setting from cookies, if available."""
    if cookies.ready():
        lang = cookies.get("site_language")
        if lang:
            st.session_state["site_language"] = lang

def display_main_header():
    """Display the main animated header with gradient background and title."""
    st.markdown(
        f"""
        <style>
        body {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(-45deg, #F0F4F8, #D9E4F5, #ACB9D7, #E4ECF7);
            background-size: 300% 300%;
            animation: backgroundGradient 15s ease infinite;
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
            background: conic-gradient(from 180deg, #00b09b, #96c93d, #96c93d, #00b09b);
            animation: rotateNeon 8s linear infinite;
            transform: translate(-50%, -50%);
            z-index: -1;
        }}
        .moroccan-flag {{
            width: 80px;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
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
            background: linear-gradient(90deg, #00b09b, #96c93d);
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
            color: #2B3E50;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        .stButton>button {{
            background: linear-gradient(135deg, #3AAFA9, #2B7A78);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 1rem;
            height: 3rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 3px 8px rgba(58, 175, 169, 0.3);
        }}
        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(43, 122, 120, 0.3);
        }}
        input[type="text"], input[type="password"] {{
            border: 1px solid #ccc;
            padding: 0.6rem;
            border-radius: 0.4rem;
            font-size: 1rem;
            width: 100%;
            transition: box-shadow 0.3s ease;
        }}
        input[type="text"]:focus, input[type="password"]:focus {{
            outline: none;
            box-shadow: 0 0 0 2px rgba(58,175,169,0.2);
        }}
        </style>
        <div class="main-login-container">
            <img class="moroccan-flag" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Flag_of_Morocco.svg/320px-Flag_of_Morocco.svg.png" alt="Moroccan Flag">
            <h1 class="login-title">{t('welcome_title')}</h1>
            <h2 class="login-message">🌟 {t('welcome_message')} 🚀</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# ENHANCED SECURITY & DB FUNCTIONS
# =============================================================================
def init_auth():
    """Check MongoDB connection on startup."""
    try:
        client = MongoClient(MONGO_URI)
        client.server_info()  # Will throw exception if connection fails
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
    finally:
        client.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_password_strength(password):
    """Check password strength and return (is_valid, message)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    checks = {
        'uppercase': re.search(r'[A-Z]', password),
        'lowercase': re.search(r'[a-z]', password),
        'digit': re.search(r'\d', password),
        'special': re.search(r'[!@#$%^&*(),.?\":{}|<>]', password)
    }
    
    strength = sum(1 for check in checks.values() if check)
    missing = []
    if not checks['uppercase']: missing.append("uppercase letters")
    if not checks['lowercase']: missing.append("lowercase letters")
    if not checks['digit']:     missing.append("numbers")
    if not checks['special']:   missing.append("special characters")
    
    if strength < 3:
        joined = ", ".join(missing)
        return False, f"Weak password. Please include: {joined}"
    return True, "Strong password"

def create_user(username: str, password: str, email: str, role: str = 'citizen', cin: str = None) -> bool:
    """Create user with enhanced security checks (unique email, lockout fields, etc.)."""
    # Validate email format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        st.error("Invalid email format")
        return False

    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        
        # Check if username or email is already used
        if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):
            st.error("Username or email already exists")
            return False
        
        is_valid, msg = check_password_strength(password)
        if not is_valid:
            st.error(msg)
            return False
        
        new_user = {
            "username": username,
            "password_hash": hash_password(password),
            "email": email,
            "role": role,
            "cin": cin,
            "failed_attempts": 0,
            "locked_until": None,
            "created_at": datetime.utcnow()
        }
        db.users.insert_one(new_user)
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        client.close()

def verify_user(username: str, password: str):
    """
    Enhanced verification with account lockout.
    Returns (is_valid, role, error_message).
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        user = db.users.find_one({"username": username})
        
        if not user:
            return False, None, "User not found"
        
        # Check lockout
        if user.get('locked_until') and user['locked_until'] > datetime.utcnow():
            remaining_secs = (user['locked_until'] - datetime.utcnow()).total_seconds()
            remaining_mins = int(remaining_secs // 60) + 1
            return False, None, f"Account locked. Try again in {remaining_mins} minutes"
        
        # Check password
        if user["password_hash"] == hash_password(password):
            # Reset lockout counters
            db.users.update_one({"_id": user["_id"]}, {"$set": {
                "failed_attempts": 0,
                "locked_until": None
            }})
            return True, user["role"], None
        else:
            # Increase failed attempts
            new_attempts = user.get("failed_attempts", 0) + 1
            db.users.update_one({"_id": user["_id"]}, {"$set": {
                "failed_attempts": new_attempts
            }})
            
            if new_attempts >= MAX_LOGIN_ATTEMPTS:
                lock_time = datetime.utcnow() + timedelta(minutes=LOCKOUT_TIME)
                db.users.update_one({"_id": user["_id"]}, {"$set": {
                    "locked_until": lock_time
                }})
                return False, None, f"Account locked for {LOCKOUT_TIME} minutes"
            
            attempts_left = MAX_LOGIN_ATTEMPTS - new_attempts
            return False, None, f"Invalid password. {attempts_left} attempts remaining"
    except Exception as e:
        return False, None, f"Error verifying user: {e}"
    finally:
        client.close()

def create_jwt_token(username: str, role: str, remember_me: bool = False) -> str:
    """
    Create a JWT token containing username and role.
    Validity is 1 day by default or 7 days if 'remember_me' is True.
    """
    try:
        expiration_days = 7 if remember_me else 1
        payload = {
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(days=expiration_days)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    except Exception as e:
        st.error(f"Error creating JWT token: {e}")
        return None

def verify_jwt_token(token: str) -> tuple:
    """
    Verify a JWT token and return (is_valid, username, role).
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return True, payload.get("username"), payload.get("role")
    except jwt.ExpiredSignatureError:
        st.error("Session expired. Please log in again.")
        return False, None, None
    except jwt.InvalidTokenError:
        st.error("Invalid token. Please log in again.")
        return False, None, None
    except Exception as e:
        st.error(f"Error verifying JWT token: {e}")
        return False, None, None

def log_login_event(username: str):
    """Optional: Log a successful login event to MongoDB."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["CivicCatalyst"]
        login_history = db["login_history"]
        login_record = {"username": username, "timestamp": datetime.utcnow()}
        login_history.insert_one(login_record)
    except Exception as e:
        st.error(f"Error logging login event: {e}")
    finally:
        client.close()

# =============================================================================
# PASSWORD RESET WORKFLOW
# =============================================================================
def generate_password_reset_token(email: str) -> str:
    """Generate JWT token for password reset."""
    try:
        client = MongoClient(MONGO_URI)
        user = client["CivicCatalyst"].users.find_one({"email": email})
        if not user:
            return None
        payload = {
            "sub": "password_reset",
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        st.error(f"Error generating reset token: {e}")
        return None
    finally:
        client.close()

def reset_password(token: str, new_password: str) -> bool:
    """Reset password using valid JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("sub") != "password_reset":
            return False
        
        client = MongoClient(MONGO_URI)
        result = client["CivicCatalyst"].users.update_one(
            {"email": payload["email"]},
            {"$set": {
                "password_hash": hash_password(new_password),
                "failed_attempts": 0,
                "locked_until": None
            }}
        )
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
# =============================================================================
def logout_user(cookies):
    """Clear session state and expire cookies, then reload the page."""
    st.session_state.clear()
    try:
        # Expire each cookie by setting an expiration date in the past.
        past_date = (datetime.utcnow() - timedelta(days=1)).isoformat()
        for key in ["jwt_token", "username", "role", "site_language"]:
            cookies[key] = {"value": "", "expires_at": past_date}
        cookies.save()
    except Exception as e:
        st.error(f"Error deleting cookies: {e}")
    st.query_params = {"logged_out": "true"}
    st.rerun()


def show_logout_button(cookies):
    """Logout button in sidebar."""
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        logout_user(cookies)

# =============================================================================
# ACTIVE SESSIONS MANAGEMENT (Optional)
# =============================================================================
def display_active_sessions(username: str):
    """Display user's active sessions and allow revoking them."""
    try:
        client = MongoClient(MONGO_URI)
        sessions = list(client["CivicCatalyst"].sessions.find({"username": username}))
        
        st.subheader("Active Sessions")
        if not sessions:
            st.info("No active sessions found.")
            return
        for s in sessions:
            col1, col2 = st.columns([3,1])
            col1.write(f"""
                **Logged In:** {s.get('login_time', 'N/A')}  
                **Expires:** {s.get('expires', 'N/A')}
            """)
            if col2.button("Revoke", key=str(s["_id"])):
                client["CivicCatalyst"].sessions.delete_one({"_id": s["_id"]})
                st.rerun()
                st.stop()
    except Exception as e:
        st.error(f"Error retrieving sessions: {e}")

# =============================================================================
# PROFILE & FORGOT PASSWORD
# =============================================================================
def enhanced_user_profile(username: str, role: str):
    """Expanded profile with password change and session management."""
    with st.expander("👤 Your Enhanced Profile", expanded=True):
        st.write(f"**Username:** {username}")
        st.write(f"**Role:** {role}")

        # --- Optional: Profile Image Upload ---
        with st.form("profile_image_form"):
            st.write("**Profile Picture (Optional)**")
            uploaded_file = st.file_uploader("Upload a profile image (JPEG/PNG)", type=["png", "jpg", "jpeg"])
            if st.form_submit_button("Save Image") and uploaded_file is not None:
                try:
                    client = MongoClient(MONGO_URI)
                    db = client["CivicCatalyst"]
                    # Store image bytes or a URL reference
                    image_bytes = uploaded_file.read()
                    db.users.update_one(
                        {"username": username},
                        {"$set": {"profile_image": image_bytes}}
                    )
                    st.success("Profile image saved successfully!")
                except Exception as e:
                    st.error(f"Error saving profile image: {e}")

        # --- Let user change password ---
        with st.form("change_password_form"):
            st.write("**Change Your Password**")
            old_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("Change Password"):
                valid_login, _, msg = verify_user(username, old_pass)
                if not valid_login:
                    st.error(msg or "Invalid current password")
                else:
                    if new_pass != confirm_pass:
                        st.error("New passwords do not match")
                    else:
                        is_strong, strength_msg = check_password_strength(new_pass)
                        if is_strong:
                            try:
                                client = MongoClient(MONGO_URI)
                                client["CivicCatalyst"].users.update_one(
                                    {"username": username},
                                    {"$set": {"password_hash": hash_password(new_pass)}}
                                )
                                st.success("Password updated successfully!")
                            except Exception as e:
                                st.error(f"Error updating password: {e}")
                        else:
                            st.error(strength_msg)

        # --- Show active sessions for user ---
        display_active_sessions(username)

def display_password_reset_form():
    """Allow user to request a password reset link."""
    with st.expander("🔒 Forgot Password?", expanded=False):
        email = st.text_input("Enter your registered email")
        if st.button("Send Reset Link"):
            token = generate_password_reset_token(email)
            if token:
                # In production, you'd email this link to the user
                st.success(f"Reset link generated. Example usage: ?token={token}")
            else:
                st.error("Error generating reset link. Check your email address.")

# =============================================================================
# REGISTRATION FORM
# =============================================================================
def enhanced_registration_form():
    """Registration form with new security features: email, password strength, etc."""
    with st.expander(t("register_header"), expanded=False):
        st.write("**Create a new account to explore the Civic Catalyst platform.**")
        with st.form("registration_form", clear_on_submit=True):
            email = st.text_input("Email (required for password recovery)")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            new_cin = st.text_input(t("cin"), placeholder="e.g., A111981")
            role_choice = st.selectbox("Select Role", ["citizen", "moderator", "admin"])
            tos_accept = st.checkbox("I accept the Terms of Service", value=False)
            
            if st.form_submit_button("Register"):
                if not tos_accept:
                    st.error("You must accept the Terms of Service to register.")
                    return
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                    return
                is_valid, msg = check_password_strength(new_password)
                if not is_valid:
                    st.error(msg)
                    return
                # Attempt creation
                if create_user(new_username, new_password, email, role_choice, new_cin):
                    st.success("Registration successful! You can now log in.")
                    st.rerun()
                    st.stop()

# =============================================================================
# LOGIN FLOW
# =============================================================================
def enhanced_login_flow(cookies):
    """Login form with 'Remember Me' and lockout mechanism."""
    st.subheader(t("login_title"))
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input(t("username"))
        password = st.text_input(t("password"), type="password")
        remember_me = st.checkbox("Remember Me", value=False, help="Extend session for 7 days")
        
        if st.form_submit_button(t("login_button")):
            with st.spinner("Authenticating..."):
                valid, role, message = verify_user(username, password)
                if valid:
                    # Create token
                    token = create_jwt_token(username, role, remember_me)
                    if token:
                        # Log the successful login
                        log_login_event(username)
                        
                        # Create a session record in DB with a unique session_id for persistence
                        try:
                            client = MongoClient(MONGO_URI)
                            session_record = {
                                "session_id": str(uuid.uuid4()),
                                "username": username,
                                "login_time": datetime.utcnow(),
                                "expires": datetime.utcnow() + timedelta(days=7 if remember_me else 1)
                            }
                            client["CivicCatalyst"].sessions.insert_one(session_record)
                        except Exception as e:
                            st.error(f"Error creating session record: {e}")
                        
                        # Compute expiration date as an ISO formatted string
                        expires_date = (datetime.utcnow() + timedelta(days=7 if remember_me else 1)).isoformat()
                        
                        # Save token in session_state
                        st.session_state["jwt_token"] = token
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        
                        # Save token, username, role, and expiration in cookies as plain values
                        cookies["jwt_token"] = token
                        cookies["username"] = username
                        cookies["role"] = role
                        cookies["expires_at"] = expires_date
                        cookies.save()

                        st.success("Login successful!")
                        time.sleep(1)  # Brief pause to show success message
                        st.rerun()
                    else:
                        st.error("Error creating session token.")
                else:
                    st.error(message or "Invalid credentials.")



def handle_existing_session(cookies):
    """
    If user has a valid JWT cookie, show the authenticated interface.
    Also verify that the cookie's username/role match the JWT token contents.
    If there's a mismatch, log out the user.
    """
    token_in_cookie = cookies.get("jwt_token")
    if token_in_cookie:
        valid, username_jwt, role_jwt = verify_jwt_token(token_in_cookie)
        cookie_username = cookies.get("username", "")
        cookie_role = cookies.get("role", "")
        
        # Ensure JWT is valid AND matches the info in cookies
        if valid and username_jwt and (username_jwt == cookie_username) and (role_jwt == cookie_role):
            # Store in session_state for easy re-use
            st.session_state["jwt_token"] = token_in_cookie
            st.session_state["username"] = username_jwt
            st.session_state["role"] = role_jwt
            show_authenticated_interface(username_jwt, role_jwt, cookies)
            return
        else:
            # Mismatch or invalid token -> Force logout
            logout_user(cookies)

    # Otherwise, show login
    show_login_interface(cookies)

# =============================================================================
# AUTHENTICATED INTERFACE
# =============================================================================
def show_authenticated_interface(username: str, role: str, cookies):
    """Display interface for already logged-in users."""
    # Display logout button and welcome message.
    show_logout_button(cookies)
    st.success(f"Welcome, {username}! (Role: {role})")
    
    # Dark Mode Toggle
    dark_mode = st.sidebar.checkbox("Enable Dark Mode?", value=False)
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #2c2c2c !important;
                color: #f0f0f0 !important;
            }
            .stButton>button {
                background: #444 !important;
                color: #f0f0f0 !important;
            }
            .stTextInput>div>div>input {
                background: #555 !important;
                color: #f0f0f0 !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #FFF !important;
                color: #000 !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
    
    # Language selection
    current_lang = st.session_state.get("site_language", "en")
    lang_options = ["en", "fr", "ar", "darija"]
    chosen_language = st.selectbox("Select Language", lang_options, index=lang_options.index(current_lang))
    if st.button("Apply Language"):
        set_language_in_cookie(cookies, chosen_language)
        st.success(f"Language updated to: {chosen_language}")
        st.rerun()
    
    # Show enhanced user profile (password change, active sessions, etc.)
    enhanced_user_profile(username, role)
    
    # Handle password reset if a token is provided in the URL
    params = st.query_params
    token = None
    if "token" in params:
        token_val = params["token"]
        token = token_val[0] if isinstance(token_val, list) else token_val
    if token:
        with st.expander("Reset Password"):
            new_pass = st.text_input("New Password", type="password")
            if st.button("Confirm Reset"):
                if reset_password(token, new_pass):
                    st.success("Password reset successful! Please log in again.")
                    logout_user(cookies)
                else:
                    st.error("Password reset failed. Link may be invalid or expired.")

# =============================================================================
# PUBLIC (UNAUTHENTICATED) INTERFACE
# =============================================================================
def show_login_interface(cookies):
    """Display the main login/registration/password-reset forms."""
    display_main_header()
    # Try to load language from cookie
    get_language_from_cookie(cookies)
    enhanced_login_flow(cookies)
    display_password_reset_form()
    enhanced_registration_form()

# =============================================================================
# MAIN
# =============================================================================
def main():
    init_auth()

    # 1) Initialize cookie manager
    cookies = EncryptedCookieManager(prefix="civic_", password=COOKIE_PASSWORD)
    
    # Check once if cookies are ready; if not, show a warning and stop.
    if not cookies.ready():
        st.warning("Loading cookies... please refresh the page.")
        st.stop()

    # If already authenticated (session state set), show authenticated interface.
    if "jwt_token" in st.session_state:
        show_authenticated_interface(
            st.session_state["username"],
            st.session_state["role"],
            cookies
        )
        return

    # Check if "logged_out" exists; if so, show the login interface.
    params = st.query_params
    if "logged_out" in params:
        show_login_interface(cookies)
        st.stop()
        return

    # Otherwise, check for existing cookie-based session.
    handle_existing_session(cookies)

if __name__ == "__main__":
    main()


import streamlit as st
import openai
import os
import tempfile
import pandas as pd
from gtts import gTTS  # For text-to-speech conversion
from pymongo import MongoClient
import hashlib
import jwt  # PyJWT
import re
from datetime import datetime, timedelta
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu

from streamlit.runtime.scriptrunner import get_script_run_ctx


try:
    # Attempt to import from the old location (for older Streamlit versions)
    from streamlit.source_util import get_pages
except ImportError:
    # For Streamlit v1.44.0+ where get_pages is removed, define it using PagesManager.
    from streamlit.runtime.pages_manager import PagesManager

    def get_pages(dummy_arg=""):
        ctx = get_script_run_ctx()
        if ctx is None:
            raise RuntimeError("Couldn't get script context")
        # Create a PagesManager instance using the main script path from the context.
        pages_manager = PagesManager(main_script_path=ctx.main_script_path)
        return pages_manager.get_pages()



# ----------------------------------------------------------------
# Constants and Configuration
# ----------------------------------------------------------------
SECRET_KEY = "mysecretkey"  # For production, load this from an environment variable!
JWT_ALGORITHM = "HS256"

# ----------------------------------------------------------------
# AUTHENTICATION FUNCTIONS USING MONGODB
# ----------------------------------------------------------------
def init_auth():
    """Initialize authentication by checking connection to MongoDB."""
    try:
        client = MongoClient("mongodb://localhost:27017")
        client.server_info()  # Throws exception if connection fails
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
    finally:
        client.close()

def hash_password(password: str) -> str:
    """Hash the password using SHA-256 (for demonstration; consider bcrypt for production)."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str, role: str = 'user', cin: str = None) -> bool:
    """
    Create a new user in the MongoDB 'users' collection.
    A valid CIN (one letter followed by six digits, e.g., D922986) is required.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        users_collection = db["users"]

        # Check if the username already exists
        if users_collection.find_one({"username": username}):
            st.error("Username already exists.")
            return False


        password_hash = hash_password(password)
        new_user = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "cin": cin
        }
        users_collection.insert_one(new_user)
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False
    finally:
        client.close()

def delete_user(username: str) -> bool:
    """
    Delete an existing user from the 'users' collection by username.
    Returns True if the user was successfully deleted.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        result = db["users"].delete_one({"username": username})
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting user: {e}")
        return False
    finally:
        client.close()

def verify_user(username: str, password: str):
    """Verify the user's credentials against MongoDB.
       Returns (True, role) if valid; otherwise, (False, None)."""
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        user = users_collection.find_one({"username": username})
        if user and user["password_hash"] == hash_password(password):
            return True, user["role"]
        else:
            return False, None
    except Exception as e:
        st.error(f"Error verifying user: {str(e)}")
        return False, None
    finally:
        client.close()

def create_jwt_token(username: str, role: str) -> str:
    """Create a JWT token containing username and role, valid for 1 day."""
    try:
        payload = {
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(days=1)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    except Exception as e:
        st.error(f"Error creating JWT token: {str(e)}")
        return None

def verify_jwt_token(token: str) -> tuple:
    """Verify a JWT token and return (is_valid, username, role) if valid; otherwise, return (False, None, None)."""
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
        st.error(f"Error verifying JWT token: {str(e)}")
        return False, None, None

# ----------------------------------------------------------------
# TRANSLATION DICTIONARY (minimal sample; extend as needed)
# ----------------------------------------------------------------
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
    }
}

def t(key):
    """Return the translated string based on session language."""
    lang = st.session_state.get("site_language", "en")
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("en", key))

# ----------------------------------------------------------------
# LOGIN HISTORY FUNCTION
# ----------------------------------------------------------------
def log_login_event(username: str):
    """
    Log a successful login to MongoDB in the 'login_history' collection.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        login_history = db["login_history"]
        login_record = {
            "username": username,
            "timestamp": datetime.utcnow()
        }
        login_history.insert_one(login_record)
    except Exception as e:
        st.error(f"Error logging login event: {e}")
    finally:
        client.close()

# ----------------------------------------------------------------
# HELPER FUNCTION TO LOAD AND DISPLAY CURRENT USERS FROM MONGODB
# ----------------------------------------------------------------
def show_current_users():
    """
    Queries the 'users' collection in the 'CivicCatalyst' DB and displays them in a table.
    Accessible to admin users only.
    """
    st.subheader("Current Users in the Database")
    client = MongoClient("mongodb://localhost:27017")
    db = client["CivicCatalyst"]
    users_collection = db["users"]

    # Fetch all users, omitting _id for cleaner display
    user_list = list(users_collection.find({}, {"_id": 0, "username": 1, "role": 1}))
    if user_list:
        st.dataframe(user_list)
    else:
        st.info("No users found in the DB.")
    client.close()

# ----------------------------------------------------------------
# LOGIN PAGE WITH ASSISTANT CHATBOT & CACHED Q&A
# ----------------------------------------------------------------
def login_page():
    """
    Displays a Streamlit-based login/registration page with an assistant chatbot (voice interface)
    for users with low literacy in Arabic/Darija.
    Incorporates multilingual UI, role-based session management, and logs login history.
    """
    # ---------------------------
    # FANCY CSS & ANIMATED UI WITH MOROCCAN FLAG
    # ---------------------------
    st.markdown(
        f"""
        <style>
        
        /* Animated background gradient */
        @keyframes backgroundGradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        body {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(-45deg, #F0F4F8, #D9E4F5, #ACB9D7, #E4ECF7);
            background-size: 300% 300%;
            animation: backgroundGradient 15s ease infinite;
            margin: 0;
            padding: 0;
        }}
        /* Main container with neon glow & transparency */
        .main-login-container {{
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
        @keyframes rotateNeon {{
            0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
            100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
        }}
        /* Moroccan flag styling with pulse animation */
        .moroccan-flag {{
            width: 80px;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        /* Title with animated text gradient */
        .login-title {{
            font-size: 2.3rem;
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
        /* Additional styling for messages, labels, buttons, etc. */
        .login-message {{
            font-size: 22px;
            font-weight: 600;
            color: #2B3E50;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        label {{
            font-weight: 600 !important;
            text-shadow: 0 1px 1px rgba(0,0,0,0.06);
        }}
        .stButton>button {{
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
        }}
        .stButton>button:hover {{
            background: linear-gradient(135deg, #2B7A78 0%, #3AAFA9 100%);
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(43, 122, 120, 0.3);
        }}
        .stButton>button:active {{
            transform: scale(0.96);
            box-shadow: 0 2px 5px rgba(43,122,120,0.2);
        }}
        input[type="text"], input[type="password"] {{
            border: 1px solid #ccc !important;
            padding: 0.6rem !important;
            border-radius: 0.4rem;
            font-size: 1rem !important;
            width: 100% !important;
            transition: box-shadow 0.3s ease;
        }}
        input[type="text"]:focus, input[type="password"]:focus {{
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(58,175,169,0.2);
        }}
        </style>

        <div class="main-login-container">
            <!-- Moroccan Flag Image with Pulse Animation -->
            <img class="moroccan-flag" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Flag_of_Morocco.svg/320px-Flag_of_Morocco.svg.png" alt="Moroccan Flag">
            <h1 class="login-title">{t('welcome_title')}</h1>
            <h2 class="login-message">🌟 {t('welcome_message')} 🚀</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # SESSION INITIALIZATION
    # ---------------------------
    if "jwt_token" not in st.session_state:
        st.session_state["jwt_token"] = None

    # ---------------------------
    # IF USER IS ALREADY LOGGED IN
    # ---------------------------
    if st.session_state["jwt_token"] is not None:
        is_valid, username, role = verify_jwt_token(st.session_state["jwt_token"])
        if is_valid:
            st.success(f"You are already logged in as **{username}** (Role: **{role}**).")
            st.session_state["role"] = role  # Store role for later role-based UI rendering

            # Language selection if not set
            if "site_language" not in st.session_state:
                st.write("Please choose your preferred site language:")
                chosen_language = st.selectbox("Select Language", ["Arabic", "French", "English", "Darija"])
                if st.button("Apply Language"):
                    st.session_state["site_language"] = chosen_language
                    st.success(f"Site language set to: {chosen_language}")
                    st.rerun()
                st.stop()
            else:
                st.info(f"Your chosen language is: {st.session_state.site_language}")

            # Logout mechanism clears session state
            if st.button("Logout"):
                st.session_state["jwt_token"] = None
                role="default"
                if "site_language" in st.session_state:
                    del st.session_state["site_language"]
                if "role" in st.session_state:
                    del st.session_state["role"]
                st.rerun()

            # ---------------------------
            # IF ROLE IS ADMIN => SHOW CURRENT USERS
            # ---------------------------
            if role == "admin":
                show_current_users()

            # ---------------------------
            # ASSISTANT CHATBOT & VOICE INTERFACE (for Arabic/Darija users)
            # ---------------------------
            if st.session_state.site_language in ["Arabic", "Darija"]:
                st.write("### 🗣️ Assistant Chatbot & Voice Interface")
                st.info("Ask your questions in Arabic or Darija to learn how to use the interface.")
                user_query = st.text_input("Your question:")

                # Load chatbot cache from CSV file (stored locally)
                cache_file = "chatbot_cache.csv"
                if os.path.exists(cache_file):
                    cache_df = pd.read_csv(cache_file)
                else:
                    cache_df = pd.DataFrame(columns=["question", "answer"])

                if st.button("Get Answer"):
                    query_norm = user_query.strip().lower()
                    cached = cache_df[cache_df["question"].str.lower() == query_norm]
                    if not cached.empty:
                        answer = cached["answer"].iloc[0]
                        st.info("Retrieved answer from cache.")
                    else:
                        with st.spinner("Generating answer..."):
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a helpful assistant who explains the Civic Catalyst interface "
                                            "in simple language for users with low literacy. Provide clear instructions "
                                            "on how to navigate and use the system, focusing on features such as project "
                                            "viewing, feedback submission, and data visualization. Respond in Arabic if the "
                                            "user's language is Arabic, or in Darija if the user's language is Darija."
                                        )
                                    },
                                    {"role": "user", "content": user_query}
                                ],
                                max_tokens=600,
                                temperature=0.5,
                            )
                            answer = response["choices"][0]["message"]["content"].strip()
                        # Cache the new Q&A pair
                        new_row = {"question": user_query, "answer": answer}
                        cache_df = cache_df.append(new_row, ignore_index=True)
                        cache_df.to_csv(cache_file, index=False)
                        st.info("Answer generated and cached.")
                    st.write("**Answer:**", answer)

                    # Convert answer to speech using gTTS (Arabic language used for both Arabic and Darija)
                    tts = gTTS(answer, lang="ar")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name)
            st.stop()
        else:
            st.session_state["jwt_token"] = None


    # ---------------------------
    # LOGIN FORM
    # ---------------------------
    st.subheader(t("login_title"))
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input(t("username"), key="login_username")
        password = st.text_input(t("password"), type="password", key="login_password")
        login_submitted = st.form_submit_button(t("login_button"))

    if login_submitted:
        success, user_role = verify_user(username, password)
        if success:
            token = create_jwt_token(username, user_role)
            if token:
                st.session_state["jwt_token"] = token
                st.session_state["username"] = username 
                st.session_state["role"] = user_role
                st.success("Login successful!")
                # Log the login event to MongoDB
                log_login_event(username)
                # Prompt for language selection after login if not already set
                chosen_language = st.selectbox("Select Language", ["Arabic", "French", "English", "Darija"], key="language_after_login")
                if st.button("Apply Language", key="apply_language_button"):
                    st.session_state["site_language"] = chosen_language
                    st.success(f"Site language set to {chosen_language}. Reloading...")
                    st.rerun()
                st.stop()
            else:
                st.error("Error creating session token.")
        else:
            st.error("Invalid username or password.")
    # ---------------------------
    # GPT API KEY OVERRIDE (OPTIONAL)
    # ---------------------------
    st.subheader(t("gpt_key_prompt"))
    st.write("""
        If you'd like to override the default GPT API key, enter it below.
        This key will be stored **only in the current session** (not saved to disk).
    """)
    new_gpt_key = st.text_input("OpenAI GPT API Key", type="password", placeholder="sk-...")
    if st.button("Use This GPT Key"):
        if new_gpt_key.strip():
            openai.api_key = new_gpt_key
            st.success("OpenAI API key set for this session!")
        else:
            st.warning("Please enter a valid OpenAI key.")
    st.markdown("---")
    
    
    # ---------------------------
    # REGISTRATION FORM
    # ---------------------------
    with st.expander(t("register_header"), expanded=False):
        st.write("Create a new account to explore the Civic Catalyst platform.")
        with st.form("registration_form", clear_on_submit=True):
            new_username = st.text_input("New Username", key="reg_username")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            new_cin = st.text_input(t("CIN Maroc"), key="reg_cin", placeholder="e.g., A111981")
            role_choice = st.selectbox("Select Role", ["citizen", "moderator", "admin"], index=0)
            register_submitted = st.form_submit_button("Register")
        if register_submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                created = create_user(new_username, new_password, role=role_choice, cin=new_cin)
                if created:
                    st.success(f"Registration successful! (Role = {role_choice}). You can now log in.")
                    st.rerun()
                else:
                    st.error("Username already exists or registration failed.")

    
    return False

   

def main():
    """
    Run this file as:
      python -m streamlit run Login.py

    Or integrate login_page() into your multi-page Streamlit app.
    """
    st.markdown("""
        <style>
            /* Hide Streamlit default sidebar navigation */
            [data-testid="stSidebarNav"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 2) Create a custom "fancy" sidebar menu
    with st.sidebar:
        st.markdown(
            """
            <style>
            /* Customize the sidebar background, text, and hover effects */
            .css-1cypcdb-NavbarWrapper, .css-qri22k, .css-1544g2n {
                background: linear-gradient(135deg, #3AAFA9 0%, #2B7A78 100%);
                color: white;
            }
            /* Menu items style */
            .nav-link {
                font-size: 16px !important;
                color: #ffffff !important;
                margin: 5px 0 !important;
            }
            .nav-link:hover {
                background-color: #2B7A78 !important;
            }
            .nav-link-selected {
                background-color: #17252A !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 3) Build the menu with labels and icons matching your files
        selected = option_menu(
            menu_title="Navigation",  # Title of the sidebar menu
            options=[
                "Login",
                "Admin Panel",
                "Chatbot",
                "Citizen Space",
                "Contact Us",
                "Cybersecurity Reports (DGSSI)",
                "Evaluation Panel",
                "General Configuration",
                "Help",
                "Jurisbot",
                "NLP Management",
                "Our Partners",
                "Public Markets & Fundraising",
                "Scaling Up & Deploy",
                "Your Privacy"
            ],
            icons=[
                "gear",
                "gear",             # Admin Panel
                "robot",            # Chatbot
                "people",           # Citizen Space
                "envelope",         # Contact Us
                "shield-lock",      # Cybersecurity Reports
                "check2-circle",    # Evaluation Panel
                "tools",            # General Configuration
                "question-circle",  # Help
                "robot",            # Jurisbot
                "cpu",              # NLP Management
                "people",           # Our Partners
                "coin",             # Public Markets & Fundraising
                "rocket",           # Scaling Up & Deploy
                "lock"              # Your Privacy
            ],
            menu_icon="cast",       # Icon for the menu title
            default_index=0,        # Which menu item is selected by default
            orientation="vertical",
            styles={
                "container": {"padding": "5px"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#2B7A78",
                },
                "nav-link-selected": {"background-color": "#17252A"},
            }
        )

    # 4) Switch pages based on the user’s selection
    #    Make sure each .py in /pages has a matching name (or filename).
    if selected == "Login":
        st.switch_page("pages\Login.py")
    elif selected == "Admin Panel":
        st.switch_page("pages\Admin_Panel.py")
    elif selected == "Chatbot":
        st.switch_page("pages\Chatbot.py")
    elif selected == "Citizen Space":
        st.switch_page("pages\Citizen_Space.py")
    elif selected == "Contact Us":
        st.switch_page("pages\Contact_Us.py")
    elif selected == "Cybersecurity Reports_(DGSSI)":
        st.switch_page("pages\Cybersecurity_Reports_(DGSSI).py")
    elif selected == "Evaluation Panel":
        st.switch_page("pages\Evaluation Panel.py")
    elif selected == "General Configuration":
        st.switch_page("pages\\General Configuration.py")
    elif selected == "Help":
        st.switch_page("pages\Help.py")
    elif selected == "Jurisbot":
        st.switch_page("pages\Jurisbot.py")
    elif selected == "NLP Management":
        st.switch_page("pages\\NLP Management.py")
    elif selected == "Our Partners":
        st.switch_page("pages\\Our Partners.py")
    elif selected == "Public Markets & Fundraising":
        st.switch_page("pages\\Public_Markets & Fundraising.py")
    elif selected == "Scaling Up & Deploy":
        st.switch_page("pages\\Scaling Up & Deploy.py")
    elif selected == "Your Privacy":
        st.switch_page("pages\\Your Privacy.py")

    st.title("Welcome to Civic Catalyst")
    st.write("Use the form below to log in or register.")
    init_auth()  # Possibly sets up DB connections, etc.
    login_page()

if __name__ == "__main__":
    main()

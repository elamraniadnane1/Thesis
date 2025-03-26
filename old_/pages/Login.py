# File: login.py
import streamlit as st
import openai
import os
import tempfile
import pandas as pd
from gtts import gTTS  # For text-to-speech conversion

# Your authentication methods from auth_system.py
from auth_system import (
    init_auth,
    verify_user,
    create_jwt_token,
    create_user,
    verify_jwt_token,
)

# ---------------------------
# LOGIN PAGE WITH ASSISTANT CHATBOT & CACHED Q&A
# ---------------------------
def login_page():
    """
    Displays a Streamlit-based login/registration page with an assistant chatbot
    (voice interface) for illiterate/low-literacy users in Arabic and Darija.
    The chatbot caches previously asked questions in a CSV file.
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

        <div class="main-login-container">
            <h1 class="login-title">Civic Catalyst</h1>
            <h2 class="login-message">🌟 Welcome to Civic Catalyst AI Toolkit! 🚀</h2>
            <p>Your gateway to intelligent citizen participation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize jwt_token if not present
    if "jwt_token" not in st.session_state:
        st.session_state["jwt_token"] = None

    # Check if user is already logged in
    if st.session_state["jwt_token"] is not None:
        is_valid, username, role = verify_jwt_token(st.session_state["jwt_token"])
        if is_valid:
            st.success(f"You are already logged in as **{username}** (Role: **{role}**).")

            # Language selection if not set
            if "site_language" not in st.session_state:
                st.write("Please choose your preferred site language:")
                chosen_language = st.selectbox("Select Language", ["Arabic", "French", "English", "Darija"])
                if st.button("Apply Language"):
                    st.session_state["site_language"] = chosen_language
                    st.success(f"Site language set to: {chosen_language}")
                    st.experimental_rerun()
                st.stop()
            else:
                st.info(f"Your chosen language is: {st.session_state.site_language}")

            # Logout button
            if st.button("Logout"):
                st.session_state["jwt_token"] = None
                if "site_language" in st.session_state:
                    del st.session_state["site_language"]
                st.experimental_rerun()

            # ---------------------------
            # NEW FEATURE: Assistant Chatbot & Voice Interface with Cached Q&A
            # ---------------------------
            # Only enable for Arabic and Darija users
            if st.session_state.site_language in ["Arabic", "Darija"]:
                st.write("### 🗣️ Assistant Chatbot & Voice Interface")
                st.info("Ask your questions in Arabic or Darija to learn how to use the interface.")
                user_query = st.text_input("Your question:")

                # Load chatbot cache from CSV
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
                        # Append new Q&A pair to cache
                        new_row = {"question": user_query, "answer": answer}
                        cache_df = cache_df.append(new_row, ignore_index=True)
                        cache_df.to_csv(cache_file, index=False)
                        st.info("Answer generated and cached.")
                    st.write("**Answer:**", answer)
                    
                    # Convert answer to speech using gTTS (using Arabic for both languages)
                    tts = gTTS(answer, lang="ar")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name)
            st.stop()
        else:
            st.session_state["jwt_token"] = None

    # ============= GPT KEY OVERRIDE (Optional) =============
    st.subheader("Optional: Provide Your GPT API Key")
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

    # ============= LOGIN FORM =============
    st.subheader("Login to Your Account")
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_submitted = st.form_submit_button("Login")

    if login_submitted:
        success, user_role = verify_user(username, password)
        if success:
            token = create_jwt_token(username, user_role)
            if token:
                st.session_state["jwt_token"] = token
                st.success("Login successful! Please select your site language below.")
                chosen_language = st.selectbox("Select Language", ["Arabic", "French", "English", "Darija"], key="language_after_login")
                if st.button("Apply Language", key="apply_language_button"):
                    st.session_state["site_language"] = chosen_language
                    st.success(f"Site language set to {chosen_language}. Reloading...")
                    st.experimental_rerun()
                st.stop()
            else:
                st.error("Error creating session token.")
        else:
            st.error("Invalid username or password.")

    with st.expander("New User? Register Here", expanded=False):
        st.write("Create a new account to explore the Civic Catalyst platform.")
        with st.form("registration_form", clear_on_submit=True):
            new_username = st.text_input("New Username", key="reg_username")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            role_choice = st.selectbox("Select Role", ["user", "admin"], index=0)
            register_submitted = st.form_submit_button("Register")
        if register_submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                created = create_user(new_username, new_password, role=role_choice)
                if created:
                    st.success(f"Registration successful! (Role = {role_choice}). You can now log in.")
                else:
                    st.error("Username already exists or registration failed.")
    return False


def main():
    """
    Run this file as:
      python -m streamlit run login.py

    Or use the login_page() function in your multi-page app.
    """
    st.title("Welcome to Civic Catalyst")
    st.write("Use the form below to log in or register.")
    init_auth()
    login_page()


if __name__ == "__main__":
    main()

# File: user_dashboard.py

import streamlit as st
import pandas as pd
import openai
from datetime import datetime

# Import your auth decorators and verification methods
from auth_system import require_auth, verify_jwt_token

##############################################################################
# 1) Initialize the GPT API
##############################################################################
def init_gpt():
    """
    Initialize OpenAI GPT with the key from Streamlit secrets.
    Ensure you have your openai api_key in secrets.toml:
      [openai]
      api_key = "<YOUR-OPENAI-API-KEY>"
    """
    openai.api_key = st.secrets["openai"]["api_key"]

##############################################################################
# 2) Helper Functions to Call GPT for Sentiment & Summaries
##############################################################################

def gpt_arabic_sentiment(text: str) -> str:
    """
    A very simple GPT-based sentiment classifier for short Arabic text.
    We'll prompt GPT with instructions to return "POS", "NEG", or "NEU".
    This is a minimal example; you can refine the system/user instructions further.
    """
    if not text.strip():
        return "NEU"

    prompt = f"""
    أنت محلل مختص في فهم شعور التعليقات بالعربية. صنّف النص التالي لأحد التصنيفات الثلاثة:
    - POS (إيجابي)
    - NEG (سلبي)
    - NEU (محايد)

    النص: {text}

    أجب فقط برمز التصنيف: POS أو NEG أو NEU
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Arabic sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        # Extract the text
        classification = response["choices"][0]["message"]["content"].strip()
        # We expect classification to be something like "POS", "NEG", or "NEU"
        # If it returns something else, handle fallback
        if classification not in ["POS", "NEG", "NEU"]:
            classification = "NEU"
        return classification
    except Exception as e:
        st.warning(f"GPT Sentiment Error: {e}")
        return "NEU"


def gpt_arabic_summary(text: str) -> str:
    """
    Use GPT to produce a brief Arabic summary of a comment or challenge/solution statement.
    Adjust the prompt for your domain or style. 
    """
    if not text.strip():
        return "لا يوجد نص للخلاصة."

    prompt = f"""
    لخص النص التالي باللغة العربية في جملة واحدة أو اثنتين:

    النص: 
    {text}

    الرجاء تقديم خلاصة موجزة وواضحة.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in Arabic summarization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.0
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        st.warning(f"GPT Summary Error: {e}")
        return "تعذّر توليد خلاصة."


##############################################################################
# 3) Load CSV Data (REMACTO Comments & REMACTO Projects)
##############################################################################
def load_remacto_comments(csv_path: str) -> pd.DataFrame:
    """
    Load the REMACTO Comments CSV. 
    Format (as given):
      رقم الفكرة,القناة,المحور,ما هي التحديات / الإشكاليات المطروحة ؟,ما هو الحل المقترح ؟
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            "idea_id",
            "channel",
            "axis",
            "challenge",
            "proposed_solution"
        ]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Comments CSV: {e}")
        return pd.DataFrame()


def load_remacto_projects(csv_path: str) -> pd.DataFrame:
    """
    Load the REMACTO Projects CSV. 
    Format (as given):
      titles,CT,Collectivité territorial,المواضيع
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            "title",
            "CT",
            "collectivite_territoriale",
            "themes"
        ]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Projects CSV: {e}")
        return pd.DataFrame()

##############################################################################
# 4) Main User Dashboard
##############################################################################
@require_auth
def main():
    """
    Enhanced user view that:
    - Displays REMACTO Comments and Projects data
    - Uses GPT-based sentiment analysis for Arabic text
    - Summarizes each challenge or solution with GPT
    - Allows user to see or filter by sentiment or axis
    - Demonstrates how GPT can be leveraged for required tasks
    """

    # Step 1: Initialize GPT with your API key
    init_gpt()

    st.title("👤 User View (My Dashboard)")
    st.write("Welcome to your personal dashboard! Engage with ongoing proposals, get GPT-based insights, and more.")

    # Retrieve the current JWT token from session_state
    token = st.session_state.get("jwt_token", None)
    
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")

            # ----------------------------------------------------------------------------
            # 4A) Load the REMACTO CSV data
            # ----------------------------------------------------------------------------
            # TODO: Adjust the paths to your local files as needed:
            comments_csv_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Comments.csv"
            projects_csv_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Projects.csv"

            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            if df_comments.empty:
                st.warning("No REMACTO Comments found or CSV not loaded.")
            if df_projects.empty:
                st.warning("No REMACTO Projects found or CSV not loaded.")
            
            # ----------------------------------------------------------------------------
            # 4B) Explore & Summarize REMACTO Comments
            # ----------------------------------------------------------------------------
            with st.expander("🔎 Explore & Summarize Citizen Comments", expanded=True):
                """
                Below are the recorded ideas/challenges from the REMACTO system. 
                We'll let GPT classify the sentiment of the proposed solutions 
                or challenges, and also provide short summaries in Arabic.
                """
                if not df_comments.empty:
                    st.write("### Raw Comments Data (First 10 Rows)")
                    st.dataframe(df_comments.head(10))

                    st.write("### GPT-Based Sentiment & Summaries")
                    # We'll let user choose how many rows to process (for performance)
                    num_rows = st.slider("Number of Comments to Analyze", 1, min(50, len(df_comments)), 5)
                    
                    # We'll create placeholders for results
                    analysis_results = []
                    for idx in range(num_rows):
                        row = df_comments.iloc[idx]
                        challenge_text = str(row["challenge"])
                        solution_text  = str(row["proposed_solution"])

                        # 1) Sentiment for the challenge
                        challenge_sentiment = gpt_arabic_sentiment(challenge_text)
                        # 2) Summaries
                        challenge_summary = gpt_arabic_summary(challenge_text)
                        solution_summary  = gpt_arabic_summary(solution_text)

                        analysis_results.append({
                            "idea_id": row["idea_id"],
                            "axis": row["axis"],
                            "challenge": challenge_text[:80] + ("..." if len(challenge_text) > 80 else ""),
                            "challenge_sentiment": challenge_sentiment,
                            "challenge_summary": challenge_summary,
                            "proposed_solution_summary": solution_summary
                        })
                    
                    df_analysis = pd.DataFrame(analysis_results)
                    st.write("### Analysis Results")
                    st.dataframe(df_analysis)

                    # Possibly let user filter by sentiment
                    selected_sentiment = st.selectbox("Filter by Challenge Sentiment", ["All", "POS", "NEG", "NEU"])
                    if selected_sentiment != "All":
                        df_filtered = df_analysis[df_analysis["challenge_sentiment"] == selected_sentiment]
                    else:
                        df_filtered = df_analysis
                    st.write(f"Showing {len(df_filtered)} row(s) matching sentiment '{selected_sentiment}'")
                    st.dataframe(df_filtered)

                else:
                    st.info("No comments to display/analysis.")

            # ----------------------------------------------------------------------------
            # 4C) Explore & Summarize REMACTO Projects
            # ----------------------------------------------------------------------------
            with st.expander("🏗️ Explore Municipal Projects", expanded=False):
                """
                These are the known projects from REMACTO Projects CSV. 
                We can also run GPT summarization for 'title' or 'themes' 
                for demonstration.
                """
                if not df_projects.empty:
                    st.write("### Projects Data")
                    st.dataframe(df_projects)

                    # Provide GPT summarization for the 'themes' column
                    st.write("### Summaries of 'themes' via GPT")
                    project_summaries = []
                    max_rows = st.slider("Number of Projects to Summarize", 1, len(df_projects), 5)
                    for idx in range(max_rows):
                        proj_row = df_projects.iloc[idx]
                        theme_text = str(proj_row["themes"])
                        summary = gpt_arabic_summary(theme_text)
                        project_summaries.append({
                            "title": proj_row["title"],
                            "themes": theme_text,
                            "themes_summary": summary
                        })
                    st.dataframe(pd.DataFrame(project_summaries))

                else:
                    st.info("No projects to display.")
            
            # ----------------------------------------------------------------------------
            # Additional user-level features could go here
            # e.g., personal timeline, proposals, comments, etc. 
            # This is just a minimal demonstration focusing on using GPT 
            # for analyzing the loaded CSV data.
            # ----------------------------------------------------------------------------

            # Provide a logout button
            st.write("---")
            if st.button("Logout Now"):
                st.session_state.jwt_token = None
                st.experimental_rerun()

        else:
            st.warning("Token is invalid or expired. Please log in again.")
    else:
        st.info("No token found in session. Please go back and log in.")


if __name__ == "__main__":
    main()

# File: user_dashboard.py

import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from datetime import datetime

# Auth system (your own code or from 'auth_system.py')
from auth_system import require_auth, verify_jwt_token

##############################################################################
# 1) GPT Initialization
##############################################################################
def init_gpt():
    """Initialize OpenAI GPT with API key from Streamlit secrets."""
    openai.api_key = st.secrets["openai"]["api_key"]

##############################################################################
# 2) GPT-based Helper Functions
##############################################################################

def gpt_arabic_sentiment(text: str) -> str:
    """
    Classify Arabic text as POS (إيجابي), NEG (سلبي), or NEU (محايد) using GPT.
    Minimal prompt – you can refine further if you like.
    """
    if not text.strip():
        return "NEU"

    system_msg = "You are a helpful assistant for Arabic sentiment analysis."
    user_msg = f"""
    أنت محلل شعوري مختص. صنف النص التالي باعتباره إيجابياً (POS)،
    أو سلبياً (NEG)، أو محايداً (NEU).
    النص: {text}
    أجب حصراً برمز واحد من بين: POS, NEG, NEU
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        classification = response["choices"][0]["message"]["content"].strip()
        if classification not in ["POS", "NEG", "NEU"]:
            classification = "NEU"
        return classification
    except Exception as e:
        st.warning(f"GPT Sentiment Error: {e}")
        return "NEU"


def gpt_arabic_summary(text: str, brief: bool = True) -> str:
    """
    Summarize Arabic text. If brief=True, produce a short (1-2 sentences) summary.
    """
    if not text.strip():
        return "لا يوجد نص للخلاصة."

    if brief:
        user_msg = f"""
        لخص النص التالي في جملة أو جملتين:
        {text}
        """
    else:
        user_msg = f"""
        لخص النص التالي بشكل عام، وأذكر النقاط الإيجابية والسلبية إن وجدت:
        {text}
        """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in Arabic summarization.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        st.warning(f"GPT Summary Error: {e}")
        return "تعذّر توليد خلاصة."


def gpt_extract_pros_cons(text: str) -> dict:
    """
    Attempt to extract top pros and cons from a text using GPT. 
    Return a dict with {'pros': [...], 'cons': [...]}
    If none found, return empty arrays.
    """
    if not text.strip():
        return {"pros": [], "cons": []}

    user_msg = f"""
    اقرأ النص التالي بالعربية، واستخرج النقاط الإيجابية (Pros) والنقاط السلبية (Cons).
    النص:
    {text}

    صيغة الجواب المقترحة:
    Pros:
    - ...
    - ...
    Cons:
    - ...
    - ...
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts pros and cons from Arabic text.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        # We'll do a basic parse: look for lines after 'Pros:' and 'Cons:'
        pros = []
        cons = []
        lines = content.splitlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if "Pros:" in line:
                current_section = "pros"
                continue
            elif "Cons:" in line:
                current_section = "cons"
                continue
            elif line.startswith("-"):
                if current_section == "pros":
                    pros.append(line.lstrip("-").strip())
                elif current_section == "cons":
                    cons.append(line.lstrip("-").strip())

        return {"pros": pros, "cons": cons}
    except Exception as e:
        st.warning(f"GPT Pros/Cons Error: {e}")
        return {"pros": [], "cons": []}


##############################################################################
# 3) Load CSV Data
##############################################################################
def load_remacto_comments(csv_path: str) -> pd.DataFrame:
    """
    REMACTO Comments CSV:
      رقم الفكرة,القناة,المحور,ما هي التحديات / الإشكاليات المطروحة ؟,ما هو الحل المقترح ؟
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            "idea_id",
            "channel",
            "axis",
            "challenge",
            "proposed_solution",
        ]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Comments CSV: {e}")
        return pd.DataFrame()


def load_remacto_projects(csv_path: str) -> pd.DataFrame:
    """
    REMACTO Projects CSV:
      titles,CT,Collectivité territorial,المواضيع
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            "title",
            "CT",
            "collectivite_territoriale",
            "themes",
        ]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Projects CSV: {e}")
        return pd.DataFrame()

##############################################################################
# 4) Utility: Wordcloud
##############################################################################
def plot_wordcloud(texts: list, title: str = "Word Cloud"):
    """
    Generate a word cloud from a list of Arabic strings.
    We'll use a simple approach - you might want 
    to remove stopwords, etc., in production.
    """
    joined_text = " ".join(texts)
    # Adjust font_path to a local .ttf if you want Arabic wordcloud shaping
    # For demonstration, we do a basic approach
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False,
    ).generate(joined_text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    st.pyplot(fig)

##############################################################################
# 5) Store Citizen Inputs
##############################################################################
def store_user_input_in_csv(username: str, input_type: str, content: str):
    """
    Append a row to 'user_inputs.csv' with columns:
      [timestamp, username, input_type, content]
    This ensures we keep a record of each user's input.
    """
    timestamp = datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "username": username,
        "input_type": input_type,
        "content": content,
    }
    csv_file = "user_inputs.csv"

    # If the file doesn't exist, create with headers
    file_exists = os.path.exists(csv_file)
    df_new = pd.DataFrame([row])

    if not file_exists:
        df_new.to_csv(csv_file, index=False)
    else:
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)

@require_auth
def main():
    """
    A comprehensive user dashboard implementing 
    multiple required features with maximum visualizations 
    to foster citizen participation.
    """
    # 1) Initialize GPT
    init_gpt()

    st.title("👤 User View (REMACTO Dashboard)")
    st.write("Welcome to your personal dashboard! Engage with projects, share feedback, see analytics, etc.")

    # Retrieve the current JWT token from session
    token = st.session_state.get("jwt_token", None)
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")

            # Let the user logout if desired
            if st.button("Logout"):
                st.session_state.jwt_token = None
                st.experimental_rerun()

            # Paths to your CSV
            comments_csv_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Comments.csv"
            projects_csv_path = r"C:\Users\DELL\OneDrive\Desktop\Thesis\REMACTO Projects.csv"

            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            # Tabs for organization
            tabs = st.tabs(
                [
                    "📊 Comments Analysis",
                    "🏗️ Projects",
                    "⚙️ Proposals & Feedback",
                    "📈 Additional Visualizations",
                    "🗃️ View User Inputs",
                ]
            )

            # -----------------------------------------------------------------
            # TAB 1: Comments Analysis
            # -----------------------------------------------------------------
            with tabs[0]:
                st.header("Citizen Comments (REMACTO)")

                if df_comments.empty:
                    st.warning("No REMACTO Comments available.")
                else:
                    st.dataframe(df_comments.head(10))

                    # Filter by axis
                    unique_axes = df_comments["axis"].unique()
                    selected_axis = st.selectbox(
                        "Filter by Axis", 
                        options=["All"] + list(unique_axes)
                    )
                    if selected_axis != "All":
                        filtered_comments = df_comments[df_comments["axis"] == selected_axis]
                    else:
                        filtered_comments = df_comments

                    st.write(f"Total {len(filtered_comments)} comments after filtering by axis: {selected_axis}")

                    # GPT-based Analysis
                    st.write("### GPT-Based Sentiment & Summaries")
                    num_rows = st.slider("Number of Rows to Analyze", 1, min(50, len(filtered_comments)), 5)

                    # We'll store results to display and do some aggregated stats
                    analysis_data = []
                    with st.spinner("Analyzing comments with GPT..."):
                        for i in range(num_rows):
                            row = filtered_comments.iloc[i]
                            challenge_text = row["challenge"]
                            sol_text = row["proposed_solution"]

                            sent = gpt_arabic_sentiment(challenge_text)
                            summary_challenge = gpt_arabic_summary(challenge_text, brief=True)
                            summary_solution  = gpt_arabic_summary(sol_text, brief=True)

                            # Optionally extract pros/cons from the proposed solution
                            pc = gpt_extract_pros_cons(sol_text)

                            analysis_data.append({
                                "idea_id": row["idea_id"],
                                "axis": row["axis"],
                                "challenge_sentiment": sent,
                                "challenge_summary": summary_challenge,
                                "solution_summary": summary_solution,
                                "solution_pros": "; ".join(pc["pros"]) if pc["pros"] else "",
                                "solution_cons": "; ".join(pc["cons"]) if pc["cons"] else "",
                            })
                    
                    df_analysis = pd.DataFrame(analysis_data)
                    st.dataframe(df_analysis)

                    # Visualization: Pie chart of sentiments
                    st.write("#### Sentiment Distribution (Challenge)")
                    sentiment_counts = df_analysis["challenge_sentiment"].value_counts()
                    fig1, ax1 = plt.subplots()
                    ax1.pie(
                        sentiment_counts.values,
                        labels=sentiment_counts.index,
                        autopct="%1.1f%%",
                        startangle=140
                    )
                    ax1.axis("equal")
                    st.pyplot(fig1)

                    # Store the "comments" or "solution" texts for a word cloud
                    st.write("#### Word Cloud of Challenges")
                    plot_wordcloud(filtered_comments["challenge"].astype(str).tolist(), "Challenges Word Cloud")

            # -----------------------------------------------------------------
            # TAB 2: Projects
            # -----------------------------------------------------------------
            with tabs[1]:
                st.header("Municipal Projects (REMACTO)")

                if df_projects.empty:
                    st.warning("No REMACTO Projects available.")
                else:
                    st.dataframe(df_projects)

                    # Summarize the 'themes' for each project
                    st.write("### Summaries of Project Themes")
                    max_rows_proj = st.slider("Number of Projects to Summarize", 1, len(df_projects), 5)
                    project_summaries = []
                    with st.spinner("Summarizing project themes..."):
                        for idx in range(max_rows_proj):
                            row = df_projects.iloc[idx]
                            theme_text = row["themes"]
                            summary = gpt_arabic_summary(theme_text, brief=False)
                            project_summaries.append({
                                "title": row["title"],
                                "themes": theme_text,
                                "themes_summary": summary,
                            })
                    st.write(pd.DataFrame(project_summaries))

                    # Visualization: Bar chart of how many times each CT or axis appears
                    st.write("### Projects by Collectivité Territoriale (CT)")
                    ct_counts = df_projects["CT"].value_counts()
                    st.bar_chart(ct_counts)

            # -----------------------------------------------------------------
            # TAB 3: Proposals & Feedback
            # -----------------------------------------------------------------
            with tabs[2]:
                st.header("Submit a New Proposal or Feedback")

                st.write("Use the forms below to propose new ideas or provide feedback about existing projects. Your input is stored for analysis.")

                st.subheader("➕ Submit a Proposal")
                proposal_title = st.text_input("Proposal Title", placeholder="e.g. مسارات خاصة للدراجات في المدينة")
                proposal_description = st.text_area("Proposal Description", placeholder="Describe your idea in detail...")

                if st.button("Submit Proposal"):
                    if proposal_title.strip() and proposal_description.strip():
                        # store in a local CSV
                        store_user_input_in_csv(username, "proposal", f"Title: {proposal_title}\nDesc: {proposal_description}")
                        st.success("Your proposal has been submitted successfully!")
                    else:
                        st.warning("Please provide both title and description.")

                st.subheader("💬 Provide Feedback")
                feedback_text = st.text_area("Your Feedback", placeholder="Any feedback or concerns about the city projects?")
                if st.button("Send Feedback"):
                    if feedback_text.strip():
                        store_user_input_in_csv(username, "feedback", feedback_text)
                        st.success("Thank you! Your feedback has been recorded.")
                    else:
                        st.warning("Please enter some feedback.")

            # -----------------------------------------------------------------
            # TAB 4: Additional Visualizations
            # -----------------------------------------------------------------
            with tabs[3]:
                st.header("Additional Visualizations & Analysis")

                st.write("""
                    Here we demonstrate further visualizations that can help 
                    stakeholders see data at a glance and ensure maximum 
                    transparency and citizen engagement.
                """)

                if not df_comments.empty:
                    # 1) Axis distribution from comments
                    axis_counts = df_comments["axis"].value_counts()
                    st.write("### Axis Distribution (Bar Chart)")
                    st.bar_chart(axis_counts)

                    # 2) Channel distribution (e.g., 'اللقاء', etc.)
                    channel_counts = df_comments["channel"].value_counts()
                    st.write("### Channels Used (Pie Chart)")
                    fig2, ax2 = plt.subplots()
                    ax2.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    # 3) Time-based approach? 
                    # If there's no real date, we can't do timeline. 
                    # But let's imagine we had a date column.
                    st.write("*(No date column in the CSV, skipping timeline charts...)*")

                    # 4) Another word cloud for solutions
                    st.write("### Word Cloud of Proposed Solutions")
                    plot_wordcloud(df_comments["proposed_solution"].astype(str).tolist(), "Proposed Solutions Word Cloud")
                else:
                    st.info("No comments to visualize here...")

            # -----------------------------------------------------------------
            # TAB 5: View User Inputs
            # -----------------------------------------------------------------
            with tabs[4]:
                st.header("🗃️ All Stored Inputs from Citizens")

                # We'll load from user_inputs.csv if it exists
                if not os.path.exists("user_inputs.csv"):
                    st.info("No user inputs stored yet. Interact with the proposals/feedback forms first.")
                else:
                    df_user_inputs = pd.read_csv("user_inputs.csv")
                    st.dataframe(df_user_inputs)

                    # Filter by current user or show all if admin
                    if role != "admin":
                        # Show only the current user's inputs
                        df_user_specific = df_user_inputs[df_user_inputs["username"] == username]
                        st.write(f"Displaying inputs only for you, **{username}**:")
                        st.dataframe(df_user_specific)

                    # Optionally, we can export these inputs as CSV
                    st.write("### Export Citizen Inputs as CSV")
                    csv_data = df_user_inputs.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="user_inputs_all.csv",
                        mime="text/csv"
                    )

        else:
            st.warning("Token is invalid or expired. Please log in again.")
    else:
        st.info("No token found in session. Please go back and log in.")


# If running as a standalone
if __name__ == "__main__":
    main()

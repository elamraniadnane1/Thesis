# File: user_dashboard_enhanced.py

import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from datetime import datetime
import re


# Auth system (your own code from 'auth_system.py')
from auth_system import require_auth, verify_jwt_token

##############################################################################
# 1) GPT Initialization
##############################################################################
def init_gpt():
    """Initialize OpenAI GPT with API key from Streamlit secrets."""
    openai.api_key = st.secrets["openai"]["api_key"]
    


##############################################################################
# 2) Arabic Text Normalization
##############################################################################
def normalize_arabic(text: str) -> str:
    """
    Simple normalization: remove diacritics, extra spaces, etc.
    For demonstration; you can expand with more robust steps if needed.
    """
    # 1) Remove diacritics
    # Arabic diacritics pattern
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(arabic_diacritics, '', text)

    # 2) Remove tatweel / kashida
    text = re.sub(r'ـ+', '', text)

    # 3) Remove punctuation (basic approach)
    text = re.sub(r'[^\w\s]', '', text)

    # 4) Trim extra spaces
    text = ' '.join(text.split())

    return text.strip()


##############################################################################
# 3) GPT-based Helper Functions
##############################################################################

def gpt_arabic_sentiment_with_polarity(text: str) -> tuple:
    """
    Classify Arabic text with a sentiment label (POS/NEG/NEU) 
    and return also a numeric polarity from -1.0 (very negative) to +1.0 (very positive).

    We'll prompt GPT for a short JSON-like reply:
    {
      "sentiment": "POS" or "NEG" or "NEU",
      "score": 0.7
    }
    (score is in range [-1, +1], with negative for negative sentiment).
    """
    text = text.strip()
    if not text:
        return ("NEU", 0.0)

    system_msg = "You are a helpful assistant for Arabic sentiment analysis."
    user_msg = f"""
    حلل الشعور في النص أدناه وأعطِ استنتاجاً من فضلك:
    1) التصنيف: اختر من بين 'POS' إيجابي، 'NEG' سلبي، أو 'NEU' محايد
    2) درجة رقمية تعبر عن قطبية الشعور بين -1.0 (سلبي جدًا) إلى +1.0 (إيجابي جدًا).

    أجب في شكل JSON فقط بدون توضيح إضافي:
    {{
      "sentiment": "POS" / "NEG" / "NEU",
      "score": float
    }}

    النص:
    {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()

        # Attempt to parse JSON
        import json
        parsed = {}
        try:
            parsed = json.loads(content)
        except:
            # fallback if GPT doesn't return perfect JSON
            # let's do some quick string searching
            # Or default to NEU, 0.0
            pass

        sentiment = parsed.get("sentiment", "NEU")
        score = parsed.get("score", 0.0)
        # Validate the sentiment
        if sentiment not in ["POS", "NEG", "NEU"]:
            sentiment = "NEU"
        # Ensure score is in [-1, +1]
        score = max(-1.0, min(1.0, float(score)))

        return (sentiment, score)
    except Exception as e:
        st.warning(f"GPT Sentiment Error: {e}")
        return ("NEU", 0.0)


def gpt_bullet_summary(text: str) -> str:
    """
    Generate bullet-point summary in Arabic for text.
    """
    if not text.strip():
        return "لا يوجد نص للخلاصة."

    prompt = f"""
    لخص النقاط الأساسية في النص التالي باللغة العربية عبر نقاط مختصرة (bullet points):
    النص:
    {text}

    الرجاء أن تكون الإجابة عبارة عن نقاط (– ...)
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant specialized in summarizing Arabic text into bullet points.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0.0,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        st.warning(f"GPT Bullet Summary Error: {e}")
        return "تعذّر توليد الملخص على شكل نقاط."


def gpt_extract_pros_cons(text: str) -> dict:
    """
    Attempt to extract top pros and cons from a text using GPT. 
    Return a dict with {'pros': [...], 'cons': [...]}
    """
    if not text.strip():
        return {"pros": [], "cons": []}

    user_msg = f"""
    اقرأ النص التالي بالعربية، واستخرج أهم النقاط الإيجابية (Pros) والنقاط السلبية (Cons) في شكل قائمة:
    النص:
    {text}

    الصيغة المطلوبة:
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
                    "content": "You extract pros and cons from Arabic text.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        
        pros = []
        cons = []
        lines = content.splitlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if line.lower().startswith("pros"):
                current_section = "pros"
                continue
            elif line.lower().startswith("cons"):
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


def gpt_extract_topics(text: str) -> list:
    """
    Use GPT to do basic "topic modeling" (similarity). 
    We'll ask GPT to produce a short list of key topics from the text.
    """
    if not text.strip():
        return []

    user_msg = f"""
    استخرج المواضيع (Topics) الأساسية المذكورة أو المشار إليها في النص أدناه.
    النص:
    {text}

    أجب بقائمة من المواضيع (كلمات أساسية أو عبارات مختصرة).
    مثال:
    1) البيئة
    2) النقل
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You extract key topics from Arabic text.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()

        # We'll parse line by line for bullet or numeric prefixes
        topics = []
        for line in content.splitlines():
            line = line.strip("-•123456789). ").strip()
            if line:
                topics.append(line)
        # remove duplicates
        topics = list(dict.fromkeys(topics))
        return topics
    except Exception as e:
        st.warning(f"GPT Topic Modeling Error: {e}")
        return []


##############################################################################
# 4) Load CSV Data
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
# 5) Wordcloud
##############################################################################
def plot_wordcloud(texts: list, title: str = "Word Cloud"):
    """
    Generate a word cloud from a list of Arabic strings.
    We'll do minimal preprocessing here. For advanced usage, 
    consider removing Arabic stopwords or diacritics first.
    """
    joined_text = " ".join(texts)
    if not joined_text.strip():
        st.warning("No text to generate wordcloud.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False,
        # For advanced usage, you can specify a font_path for Arabic shaping:
        # font_path="path/to/arabic_font.ttf",
    ).generate(joined_text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    st.pyplot(fig)


##############################################################################
# 6) Store Citizen Inputs
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


##############################################################################
# 7) Main Dashboard
##############################################################################
@require_auth
def main():
    # 1) Initialize GPT
    init_gpt()

    st.title("👤 User View (REMACTO Dashboard) - Enhanced")
    st.write("Welcome to your personal dashboard! Engage with projects, share feedback, see analytics, etc.")

    # Retrieve the current JWT token from session
    token = st.session_state.get("jwt_token", None)
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")

            # Logout
            if st.button("Logout"):
                st.session_state.jwt_token = None
                st.experimental_rerun()

            # CSV paths (adjust if needed)
            comments_csv_path = "REMACTO Comments.csv"
            projects_csv_path = "REMACTO Projects.csv"

            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            # Main tabs
            tabs = st.tabs(
                [
                    "📊 Comments Analysis",
                    "🏗️ Projects",
                    "⚙️ Proposals & Feedback",
                    "📈 Extra Visualizations",
                    "🗃️ All User Inputs",
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
                    st.write("### Original Data (first 10)")
                    st.dataframe(df_comments.head(10))

                    # Optional: Normalization
                    st.write("#### Apply Basic Arabic Normalization (Optional)")
                    do_normalize = st.checkbox("Normalize Text?", value=False)
                    df_comments_proc = df_comments.copy()

                    if do_normalize:
                        df_comments_proc["challenge"] = df_comments_proc["challenge"].apply(normalize_arabic)
                        df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].apply(normalize_arabic)
                        st.success("Text normalization applied!")

                    # Filter by axis
                    unique_axes = df_comments_proc["axis"].unique()
                    selected_axis = st.selectbox("Filter by Axis", ["All"] + list(unique_axes))
                    if selected_axis != "All":
                        filtered_comments = df_comments_proc[df_comments_proc["axis"] == selected_axis]
                    else:
                        filtered_comments = df_comments_proc

                    st.write(f"Total {len(filtered_comments)} comments after filtering by axis: {selected_axis}")

                    # GPT-based Analysis
                    st.write("### GPT-Based Sentiment & Summaries + Polarity")
                    num_rows = st.slider("Number of Rows to Analyze", 1, min(50, len(filtered_comments)), 5)

                    analysis_data = []
                    with st.spinner("Analyzing comments with GPT..."):
                        for i in range(num_rows):
                            row = filtered_comments.iloc[i]
                            challenge_text = row["challenge"]
                            solution_text = row["proposed_solution"]

                            # 1) Sentiment + Polarity
                            sentiment, polarity_score = gpt_arabic_sentiment_with_polarity(challenge_text)

                            # 2) Bullet Summary for challenge
                            bullet_challenge = gpt_bullet_summary(challenge_text)
                            # 3) Pros & Cons for solution
                            pros_cons = gpt_extract_pros_cons(solution_text)
                            pros_join = "; ".join(pros_cons["pros"]) if pros_cons["pros"] else ""
                            cons_join = "; ".join(pros_cons["cons"]) if pros_cons["cons"] else ""

                            # 4) Basic topic extraction from the challenge
                            topics = gpt_extract_topics(challenge_text)
                            topics_join = "; ".join(topics)

                            analysis_data.append({
                                "idea_id": row["idea_id"],
                                "axis": row["axis"],
                                "channel": row["channel"],
                                "challenge_sentiment": sentiment,
                                "polarity_score": polarity_score,
                                "challenge_summary_bullets": bullet_challenge,
                                "solution_pros": pros_join,
                                "solution_cons": cons_join,
                                "extracted_topics": topics_join,
                            })

                    df_analysis = pd.DataFrame(analysis_data)
                    st.dataframe(df_analysis)

                    # Polarity Distribution
                    st.write("#### Polarity Distribution (Histogram)")
                    fig_pol, ax_pol = plt.subplots()
                    ax_pol.hist(df_analysis["polarity_score"], bins=10, color="skyblue")
                    ax_pol.set_title("Polarity Score Distribution")
                    ax_pol.set_xlabel("Polarity Score (-1 = negative, +1 = positive)")
                    ax_pol.set_ylabel("Count")
                    st.pyplot(fig_pol)

                    # Sentiment Pie
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

                    # Wordcloud
                    st.write("#### Word Cloud of All Challenges")
                    plot_wordcloud(filtered_comments["challenge"].astype(str).tolist(), "Challenges Word Cloud")

            # -----------------------------------------------------------------
            # TAB 2: Projects
            # -----------------------------------------------------------------
            with tabs[1]:
                st.header("Municipal Projects (REMACTO)")
                if df_projects.empty:
                    st.warning("No REMACTO Projects available.")
                else:
                    st.write("### Projects Data (Preview)")
                    st.dataframe(df_projects.head(10))

                    st.write("### Summaries of Project Themes")
                    max_rows_proj = st.slider("Number of Projects to Summarize", 1, len(df_projects), 5)
                    project_summaries = []
                    with st.spinner("Summarizing project themes..."):
                        for idx in range(max_rows_proj):
                            row = df_projects.iloc[idx]
                            theme_text = row["themes"]
                            bullet_sum = gpt_bullet_summary(theme_text)
                            project_summaries.append({
                                "title": row["title"],
                                "themes": theme_text,
                                "bullet_summary": bullet_sum,
                            })

                    st.write(pd.DataFrame(project_summaries))

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
                        content = f"Title: {proposal_title}\nDescription: {proposal_description}"
                        store_user_input_in_csv(username, "proposal", content)
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
            # TAB 4: Extra Visualizations
            # -----------------------------------------------------------------
            with tabs[3]:
                st.header("Extra Visualizations & Analysis")
                st.write("""
                    More plots or deeper analysis can be included here to
                    help stakeholders glean insights from the REMACTO data.
                """)

                if not df_comments.empty:
                    # Axis distribution
                    axis_counts = df_comments["axis"].value_counts()
                    st.write("### Axis Distribution (Bar Chart)")
                    st.bar_chart(axis_counts)

                    # Channel distribution
                    channel_counts = df_comments["channel"].value_counts()
                    st.write("### Channels Used (Pie Chart)")
                    fig2, ax2 = plt.subplots()
                    ax2.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    # Word Cloud of solutions
                    st.write("### Word Cloud of Proposed Solutions")
                    plot_wordcloud(df_comments["proposed_solution"].astype(str).tolist(), "Proposed Solutions Word Cloud")

                else:
                    st.info("No comments data available for extra visualization.")

            # -----------------------------------------------------------------
            # TAB 5: All User Inputs
            # -----------------------------------------------------------------
            with tabs[4]:
                st.header("🗃️ All Stored Inputs from Citizens")
                csv_file = "user_inputs.csv"
                if not os.path.exists(csv_file):
                    st.info("No user inputs stored yet. Interact with the proposals/feedback forms first.")
                else:
                    df_user_inputs = pd.read_csv(csv_file)
                    st.dataframe(df_user_inputs)

                    if role != "admin":
                        df_user_specific = df_user_inputs[df_user_inputs["username"] == username]
                        st.write(f"Showing inputs for your user: **{username}**")
                        st.dataframe(df_user_specific)

                    # Export
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


# If running standalone
if __name__ == "__main__":
    main()

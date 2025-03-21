# File: user_dashboard_enhanced.py

import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
import base64
from datetime import datetime
import re
import json

# For better Arabic font handling in Matplotlib:
import matplotlib
# Use a font that supports Arabic glyphs. DejaVu Sans is often a fallback.
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.unicode_minus'] = False

# Auth system (your own code from 'auth_system.py')
from auth_system import require_auth, verify_jwt_token

##############################################################################
# 1) GPT Initialization
##############################################################################
def init_gpt():
    """
    Initialize OpenAI GPT with the key stored in st.secrets.
    If you allow user overrides at runtime, you might conditionally set
    openai.api_key = st.secrets["openai"]["api_key"] only if openai.api_key isn't set.
    """
    if not openai.api_key:
        openai.api_key = st.secrets["openai"]["api_key"]


##############################################################################
# 2) Arabic Text Normalization
##############################################################################
def normalize_arabic(text: str) -> str:
    """
    Simple normalization: remove diacritics, extra spaces, etc.
    You can expand with more robust steps if needed.
    """
    # Remove Arabic diacritics
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(diacritics_pattern, '', text)

    # Remove tatweel / kashida
    text = re.sub(r'ـ+', '', text)

    # Remove punctuation (basic approach)
    text = re.sub(r'[^\w\s]', '', text)

    # Trim extra spaces
    text = ' '.join(text.split())

    return text.strip()


##############################################################################
# 3) GPT-based Helper Functions
##############################################################################
def gpt_arabic_sentiment_with_polarity(text: str) -> tuple:
    """
    Classify Arabic text with a sentiment label (POS/NEG/NEU)
    plus a numeric polarity from -1.0 to +1.0.
    Returns (sentiment_label, polarity_score).
    """
    text = text.strip()
    if not text:
        return ("NEU", 0.0)

    system_msg = "You are a helpful assistant for Arabic sentiment analysis."
    user_msg = f"""
    حلل الشعور في النص أدناه وأعطِ استنتاجاً من فضلك:
    1) التصنيف: اختر من بين 'POS' إيجابي، 'NEG' سلبي، أو 'NEU' محايد
    2) درجة رقمية بين -1.0 إلى +1.0

    أجب بصيغة JSON:
    {{
      "sentiment": "POS"/"NEG"/"NEU",
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

        parsed = {}
        try:
            parsed = json.loads(content)
        except:
            pass

        sentiment = parsed.get("sentiment", "NEU")
        score = float(parsed.get("score", 0.0))
        if sentiment not in ["POS", "NEG", "NEU"]:
            sentiment = "NEU"
        score = max(-1.0, min(1.0, score))
        return (sentiment, score)
    except Exception as e:
        st.warning(f"GPT Sentiment Error: {e}")
        return ("NEU", 0.0)


def gpt_bullet_summary(text: str) -> str:
    """
    Generate bullet-point summary in Arabic for the given text.
    """
    if not text.strip():
        return "لا يوجد نص للخلاصة."

    prompt = f"""
    لخص النقاط الأساسية في النص التالي باللغة العربية عبر نقاط (bullet points):
    النص:
    {text}
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
        return "تعذّر توليد الملخص."


def gpt_extract_pros_cons(text: str) -> dict:
    """
    Attempt to extract top pros and cons from a text using GPT.
    Returns {'pros': [...], 'cons': [...]}
    """
    if not text.strip():
        return {"pros": [], "cons": []}

    user_msg = f"""
    اقرأ النص التالي بالعربية، واستخرج أهم النقاط الإيجابية (Pros) والنقاط السلبية (Cons):
    النص:
    {text}

    الصيغة:
    Pros:
    - ...
    Cons:
    - ...
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract pros and cons from Arabic text."},
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
            low_line = line.lower().strip()
            if low_line.startswith("pros"):
                current_section = "pros"
                continue
            elif low_line.startswith("cons"):
                current_section = "cons"
                continue
            elif line.strip().startswith("-"):
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
    Use GPT to do basic "topic extraction" from Arabic text.
    Returns a list of discovered topics.
    """
    if not text.strip():
        return []

    user_msg = f"""
    استخرج المواضيع الأساسية المذكورة في النص:
    {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract key topics from Arabic text."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        topics = []
        for line in content.splitlines():
            line = line.strip("-•123456789). ").strip()
            if line:
                topics.append(line)
        topics = list(dict.fromkeys(topics))
        return topics
    except Exception as e:
        st.warning(f"GPT Topic Modeling Error: {e}")
        return []


##############################################################################
# 4) Handling Large Arabic Text for GPT Translation
##############################################################################
def chunk_text(text: str, chunk_size: int = 2000) -> list:
    """
    Break a large string into smaller chunks (each up to chunk_size characters).
    This helps avoid large token usage errors in GPT.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def gpt_translate_arabic(text: str, target_language: str = "English") -> str:
    """
    Translate the given Arabic text to either English or French using GPT,
    chunking if necessary to avoid token limit errors.
    Also optionally limit total length or sample the text if data is extremely large.

    :param text: the Arabic source text
    :param target_language: "English" or "French"
    :return: translated text in the target language
    """

    text = text.strip()
    if not text:
        return ""

    # If text is extremely large, let's limit or sample:
    max_overall_length = 6000
    if len(text) > max_overall_length:
        lines = text.split('\n')
        random.shuffle(lines)
        lines = lines[:200]
        text = "\n".join(lines)[:max_overall_length]

    text_chunks = chunk_text(text, chunk_size=1500)
    translated_chunks = []

    # Prepare system and user prompts
    system_prompt = f"You translate Arabic text to {target_language}."
    for chunk in text_chunks:
        user_msg = f"""
Translate the following Arabic text into {target_language} without additional commentary:
{chunk}
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.0,
            )
            chunk_translation = response["choices"][0]["message"]["content"].strip()
            translated_chunks.append(chunk_translation)
        except Exception as e:
            st.warning(f"GPT Translate Error on chunk: {e}")
            continue

    return " ".join(translated_chunks)


##############################################################################
# 5) Load CSV Data
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
# 6) Wordcloud (with GPT-based translation to chosen language)
##############################################################################
def plot_wordcloud(texts: list, title: str, target_language: str = "English"):
    """
    1) Merge the list of Arabic texts into one string.
    2) Use GPT to translate that text to either English or French in manageable chunks.
    3) Generate a WordCloud from the translated text.
    """
    joined_text_ar = "\n".join(texts).strip()
    if not joined_text_ar:
        st.warning("No text found to generate wordcloud.")
        return

    with st.spinner(f"Translating text to {target_language} for WordCloud (may sample if data is huge)..."):
        translated_text = gpt_translate_arabic(joined_text_ar, target_language)

    if not translated_text.strip():
        st.warning("Translation returned empty. Cannot generate WordCloud.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False
    ).generate(translated_text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    st.pyplot(fig)


##############################################################################
# 7) Store Citizen Inputs
##############################################################################
def store_user_input_in_csv(username: str, input_type: str, content: str):
    """
    Append a row to 'user_inputs.csv' with columns:
      [timestamp, username, input_type, content]
    """
    timestamp = datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "username": username,
        "input_type": input_type,
        "content": content,
    }
    csv_file = "user_inputs.csv"

    file_exists = os.path.exists(csv_file)
    df_new = pd.DataFrame([row])

    if not file_exists:
        df_new.to_csv(csv_file, index=False)
    else:
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)


##############################################################################
# 8) Main Dashboard
##############################################################################
@require_auth
def main():
    # 1) Initialize GPT
    init_gpt()

    # ---------------------------------------------------------------------
    # Option to choose the translation language for WordCloud
    # ---------------------------------------------------------------------
    st.sidebar.header("WordCloud Language")
    chosen_language = st.sidebar.selectbox(
        "Select WordCloud translation language:",
        ["English", "French"]
    )

    # Title & Description
    st.title("👤 User View (REMACTO Dashboard) - Enhanced w/ Language Option")
    st.write("Welcome to your personal dashboard! Engage with projects, share feedback, see analytics, etc.")

    token = st.session_state.get("jwt_token", None)
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")

            # Logout
            if st.button("Logout"):
                st.session_state.jwt_token = None
                st.experimental_rerun()

            # CSV paths
            comments_csv_path = "REMACTO Comments.csv"
            projects_csv_path = "REMACTO Projects.csv"

            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            # Create main tabs
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
                    st.write("### Original Data (first 10 rows):")
                    st.dataframe(df_comments.head(10))

                    # Optional: Normalization
                    st.write("#### Apply Basic Arabic Normalization (Optional)")
                    do_normalize = st.checkbox("Normalize Text?", value=False)
                    df_comments_proc = df_comments.copy()

                    if do_normalize:
                        df_comments_proc["challenge"] = df_comments_proc["challenge"].apply(normalize_arabic)
                        df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].apply(normalize_arabic)
                        st.success("Text normalization applied to challenges/solutions.")

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

                            # 4) Basic topic extraction
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
                    ax_pol.set_title("Polarity Score Distribution", fontproperties=None)
                    ax_pol.set_xlabel("Polarity Score (-1 = negative, +1 = positive)")
                    ax_pol.set_ylabel("Count")
                    st.pyplot(fig_pol)

                    # Sentiment Distribution
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

                    # Sentiment by Axis (Stacked Bar)
                    st.write("#### Sentiment by Axis (Stacked Bar Chart)")
                    cross_tab = pd.crosstab(df_analysis["axis"], df_analysis["challenge_sentiment"])
                    fig_stack, ax_stack = plt.subplots(figsize=(6, 4))
                    cross_tab.plot(kind='bar', stacked=True, ax=ax_stack)
                    ax_stack.set_title("Number of Comments by Axis and Sentiment")
                    ax_stack.set_xlabel("Axis")
                    ax_stack.set_ylabel("Count of Comments")
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig_stack)

                    # WordCloud (in chosen language)
                    st.write(f"#### Word Cloud of All Challenges (Translated to {chosen_language})")
                    plot_wordcloud(
                        filtered_comments["challenge"].astype(str).tolist(),
                        f"Challenges Word Cloud ({chosen_language})",
                        target_language=chosen_language
                    )

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
                    axis_counts = df_comments["axis"].value_counts()
                    st.write("### Axis Distribution (Bar Chart)")
                    st.bar_chart(axis_counts)

                    channel_counts = df_comments["channel"].value_counts()
                    st.write("### Channels Used (Pie Chart)")
                    fig2, ax2 = plt.subplots()
                    ax2.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    st.write(f"### Word Cloud of Proposed Solutions (Translated to {chosen_language})")
                    plot_wordcloud(
                        df_comments["proposed_solution"].astype(str).tolist(),
                        f"Proposed Solutions Word Cloud ({chosen_language})",
                        target_language=chosen_language
                    )
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

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
import hashlib
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct


# For better Arabic font handling in Matplotlib:
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.unicode_minus'] = False

# Import your auth system
from auth_system import require_auth, verify_jwt_token


##############################################################################
# 0) UI TEXT DICTIONARIES FOR INTERFACE ELEMENTS
##############################################################################
# This allows us to translate basic interface labels into Arabic, French, English, Darija.
ui_texts = {
    "English": {
        "title": "📊 Dashboard",
        "header_comments": "💬 Citizen Comments Analysis",
        "label_normalize": "🧹 Normalize Arabic Text (Optional)",
        "analysis_section": "🧠 AI-Powered Sentiment Analysis & Summaries",
        "proposal_header": "📝 Submit Your Proposal or Feedback",
        "proposal_title_label": "📌 Proposal Title",
        "proposal_description_label": "🧾 Proposal Description",
        "proposal_submit_button": "📤 Submit Proposal",
        "feedback_label": "💭 Your Feedback",
        "feedback_button": "📬 Send Feedback",
        "logout_button": "🔓 Logout",
        "no_comments_msg": "⚠️ No comments available at this moment.",
        "original_data_label": "📋 Original Data (first 10 rows):",
        "norm_success": "✅ Text normalization applied successfully.",
        "no_token_msg": "⚠️ No token found in session. Please log in.",
        "token_invalid": "❌ Token is invalid or expired. Please log in again.",
        "logged_in_as": "✅ Logged in as:",
        "role_label": "(Role: ",
        "closing_paren": ")",
        "projects_header": "🏗️ Municipal Projects",
        "no_projects_msg": "⚠️ No projects available.",
        "projects_data_preview": "📂 Projects Data (Preview)",
        "summaries_of_themes": "📝 Project Themes Summaries",
        "proposals_feedback_tab": "🗳️ Submit Proposal or Feedback",
        "extra_visualizations_tab": "📈 Extra Visualizations & Analysis",
        "all_user_inputs_tab": "🗃️ All Citizen Inputs"
    },
    "Arabic": {
        "title": "📊 لوحة المستخدم",
        "header_comments": "💬 تحليل تعليقات المواطنين",
        "label_normalize": "🧹 تنقية النص العربي (اختياري)",
        "analysis_section": "🧠 تحليل المشاعر والتلخيص باستخدام الذكاء الاصطناعي",
        "proposal_header": "📝 إرسال اقتراح جديد أو ملاحظات",
        "proposal_title_label": "📌 عنوان الاقتراح",
        "proposal_description_label": "🧾 وصف الاقتراح",
        "proposal_submit_button": "📤 إرسال الاقتراح",
        "feedback_label": "💭 ملاحظاتك",
        "feedback_button": "📬 إرسال الملاحظات",
        "logout_button": "🔓 تسجيل الخروج",
        "no_comments_msg": "⚠️ لا توجد تعليقات متاحة حالياً.",
        "original_data_label": "📋 البيانات الأصلية (أول 10 صفوف):",
        "norm_success": "✅ تم تطبيق تنقية النص بنجاح.",
        "no_token_msg": "⚠️ لا يوجد رمز في الجلسة. يرجى تسجيل الدخول.",
        "token_invalid": "❌ الرمز غير صالح أو منتهي. يرجى تسجيل الدخول مجدداً.",
        "logged_in_as": "✅ تم تسجيل الدخول باسم:",
        "role_label": "(الدور: ",
        "closing_paren": ")",
        "projects_header": "🏗️ مشاريع البلدية",
        "no_projects_msg": "⚠️ لا توجد مشاريع متاحة.",
        "projects_data_preview": "📂 عرض بيانات المشاريع",
        "summaries_of_themes": "📝 تلخيص مواضيع المشاريع",
        "proposals_feedback_tab": "🗳️ إرسال اقتراح أو ملاحظات",
        "extra_visualizations_tab": "📈 تصورات وتحليلات إضافية",
        "all_user_inputs_tab": "🗃️ جميع المدخلات من المواطنين"
    },
    "French": {
        "title": "📊 Tableau de bord",
        "header_comments": "💬 Analyse des commentaires citoyens",
        "label_normalize": "🧹 Normalisation du texte arabe (optionnel)",
        "analysis_section": "🧠 Analyse de sentiment et résumés par IA",
        "proposal_header": "📝 Soumettre une proposition ou un retour",
        "proposal_title_label": "📌 Titre de la proposition",
        "proposal_description_label": "🧾 Description de la proposition",
        "proposal_submit_button": "📤 Soumettre la proposition",
        "feedback_label": "💭 Vos commentaires",
        "feedback_button": "📬 Envoyer le commentaire",
        "logout_button": "🔓 Se déconnecter",
        "no_comments_msg": "⚠️ Aucun commentaire disponible pour le moment.",
        "original_data_label": "📋 Données d'origine (10 premières lignes):",
        "norm_success": "✅ Normalisation du texte appliquée avec succès.",
        "no_token_msg": "⚠️ Aucun jeton trouvé dans la session. Veuillez vous reconnecter.",
        "token_invalid": "❌ Jeton invalide ou expiré. Veuillez vous reconnecter.",
        "logged_in_as": "✅ Connecté en tant que:",
        "role_label": "(Rôle: ",
        "closing_paren": ")",
        "projects_header": "🏗️ Projets municipaux",
        "no_projects_msg": "⚠️ Aucun projet disponible.",
        "projects_data_preview": "📂 Aperçu des données du projet",
        "summaries_of_themes": "📝 Résumés des thèmes du projet",
        "proposals_feedback_tab": "🗳️ Soumettre une proposition ou un retour",
        "extra_visualizations_tab": "📈 Visualisations supplémentaires",
        "all_user_inputs_tab": "🗃️ Toutes les entrées des citoyens"
    },
    "Darija": {
        "title": "📊 لوحة المستخدم بالدارجة",
        "header_comments": "💬 تحليل تعليقات الناس",
        "label_normalize": "🧹 تنقية النص العربي شوية (اختياري)",
        "analysis_section": "🧠 تحليل المشاعر مع الذكاء الاصطناعي",
        "proposal_header": "📝 صيفط اقتراح جديد ولا ملاحظة",
        "proposal_title_label": "📌 عنوان الاقتراح بالدارجة",
        "proposal_description_label": "🧾 وصف الاقتراح بالتفاصيل",
        "proposal_submit_button": "📤 صيفط الاقتراح",
        "feedback_label": "💭 رأيك",
        "feedback_button": "📬 صيفط رأيك",
        "logout_button": "🔓 خروج",
        "no_comments_msg": "⚠️ ماكايناش تعليقات متاحة دابا.",
        "original_data_label": "📋 البيانات الأصلية (أول 10 صفوف):",
        "norm_success": "✅ تنقية النص تمّت بنجاح.",
        "no_token_msg": "⚠️ ماكاينش التوكن فالسيشن. رجع تسيني.",
        "token_invalid": "❌ التوكن خايب ولا منتهي. خصك تسيني.",
        "logged_in_as": "✅ نتا داخل باسم:",
        "role_label": "(دور: ",
        "closing_paren": ")",
        "projects_header": "🏗️ مشاريع الجماعة",
        "no_projects_msg": "⚠️ ماكاين لا مشاريع دابا.",
        "projects_data_preview": "📂 عرض بيانات المشاريع",
        "summaries_of_themes": "📝 تلخيص مواضيع المشاريع",
        "proposals_feedback_tab": "🗳️ صيفط اقتراح ولا ملاحظة",
        "extra_visualizations_tab": "📈 تصاور وتحليلات إضافية",
        "all_user_inputs_tab": "🗃️ جميع المدخلات ديال الناس"
    }
}



##############################################################################
# 4) GPT Initialization + Language Dictionary
##############################################################################
def init_gpt():
    """
    Initialize OpenAI GPT with the key stored in st.secrets.
    """
    if not openai.api_key:
        openai.api_key = st.secrets["openai"]["api_key"]


##############################################################################
# 5) Utility: Normalizing Arabic, chunking, GPT for data
##############################################################################
def normalize_arabic(text: str) -> str:
    # (same as above)
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(diacritics_pattern, '', text)
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
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
# 9) The Main Dashboard
##############################################################################
@require_auth
def main():
    # 1) Initialize GPT
    init_gpt()

    # 2) Determine the interface language from session_state
    #    (fallback to English if not present)
    lang = st.session_state.get("site_language", "English")
    if lang not in ui_texts:
        lang = "English"

    # Shortcut to the dictionary for the chosen language
    L = ui_texts[lang]

     # ======================== SIDEBAR (ARABIC) ========================
    st.sidebar.title("لوحة جانبية (Side Bar)")
    
    # A big welcome text for illiterate/literate
    st.sidebar.markdown("### 🤝 مرحباً بكم في المنصة!")
    st.sidebar.markdown("""
    هذه اللوحة الجانبية تقدم مجموعة من الأدوات:
    - الإرشادات الصوتية
    - تكبير الخط
    - استخدام الأيقونات البصرية
    """)
    
    # 1) Accessibility Expanders
    with st.sidebar.expander("🦻 إعدادات الوصول"):
        # Option: Increase font size
        font_size = st.selectbox("حجم الخط:", ["صغير", "متوسط", "كبير"])
        st.write("يمكنك اختيار حجم الخط المناسب.")
        # Option: Provide a toggle for "High Contrast Mode"
        high_contrast = st.checkbox("وضع تباين عالٍ")
        st.write("هذا الوضع يرفع من وضوح العناصر للأشخاص ذوي القدرة المحدودة على الرؤية.")
    
    # 2) Audio Guidance / TTS for illiterate users
    with st.sidebar.expander("🔊 مساعدة صوتية (TTS)"):
        st.write("للمستخدمين الذين لا يقرؤون العربية بسهولة، يمكنهم الاستماع إلى النصوص الأساسية.")
        # Placeholder: We can have a button to "Play Audio" or "Stop Audio"
        if st.button("تشغيل المساعدة الصوتية"):
            st.info("🚧 يجري تشغيل المساعدة الصوتية... (نموذج تجريبي)")
        if st.button("إيقاف"):
            st.info("🚧 تم إيقاف المساعدة الصوتية.")
    
    # 3) Icons / Visual Aids
    st.sidebar.markdown("### 🏷️ رموز بصرية مساعدة:")
    st.sidebar.write("يمكنك ملاحظة استخدام الأيقونات لتسهيل التعرف على الأقسام:")
    st.sidebar.write("- 📊 للبيانات العامة")
    st.sidebar.write("- 📝 للإقتراحات")
    st.sidebar.write("- 💭 للملاحظات")
    st.sidebar.write("- 🔓 لتسجيل الخروج")
    
    # 4) Possibly a "Language Switcher" in Arabic (though we have a site_language from login)
    with st.sidebar.expander("🌐 تغيير اللغة"):
        chosen_lang = st.selectbox("اختر لغة العرض", ["Arabic", "English", "French", "Darija"], index=0)
        if st.button("تطبيق اللغة"):
            st.session_state.site_language = chosen_lang
            st.experimental_rerun()

    # Title & Description
    st.title(L["title"])
    st.write("Welcome to your personal dashboard! Engage with projects, share feedback, see analytics, etc.")

    # Check JWT token
    token = st.session_state.get("jwt_token", None)
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            # Show a success message with user's role
            st.success(f"{L['logged_in_as']} **{username}** {L['role_label']}{role}{L['closing_paren']}")

            # Logout button
            if st.button(L["logout_button"]):
                st.session_state.jwt_token = None
                st.experimental_rerun()

            # CSV paths (adjust as needed)
            comments_csv_path = "REMACTO Comments.csv"
            projects_csv_path = "REMACTO Projects.csv"

            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            # Create main tabs
            tabs = st.tabs(
                [
                    L["header_comments"],             # e.g. "Citizen Comments (REMACTO)"
                    L["projects_header"],             # e.g. "Municipal Projects (REMACTO)"
                    L["proposals_feedback_tab"],       # e.g. "Submit a New Proposal or Feedback"
                    L["extra_visualizations_tab"],     # e.g. "Extra Visualizations & Analysis"
                    L["all_user_inputs_tab"],          # e.g. "All Stored Inputs from Citizens"
                ]
            )

            # -----------------------------------------------------------------
            # TAB 1: Comments Analysis
            # -----------------------------------------------------------------
            # ------------------------- TAB 1: Comments Analysis -------------------------
            with tabs[0]:
                st.header("💬 Citizen Comments Analysis")
                if df_comments.empty:
                    st.warning("⚠️ No REMACTO Comments available.")
                else:
                    st.write("### 📋 Original Data (first 10 rows):")
                    st.dataframe(df_comments.head(10))

                    st.write("#### 🧹 Apply Basic Arabic Normalization (Optional)")
                    do_normalize = st.checkbox("Normalize Text?")
                    df_comments_proc = df_comments.copy()

                    # Ensure text columns are strings
                    df_comments_proc["challenge"] = df_comments_proc["challenge"].astype(str)
                    df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].astype(str)

                    if do_normalize:
                        df_comments_proc["challenge"] = df_comments_proc["challenge"].apply(normalize_arabic)
                        df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].apply(normalize_arabic)
                        st.success("✅ Text normalization applied.")

                    # Filter by axis
                    unique_axes = df_comments_proc["axis"].unique()
                    selected_axis = st.selectbox("📍 Filter by Axis:", ["All"] + list(unique_axes))
                    if selected_axis != "All":
                        filtered_comments = df_comments_proc[df_comments_proc["axis"] == selected_axis]
                    else:
                        filtered_comments = df_comments_proc

                    st.write(f"Total {len(filtered_comments)} comments after filtering by axis: **{selected_axis}**")

                    # ---------------------------
                    # Caching mechanism
                    # ---------------------------
                    CACHE_FILE = "cached_gpt_analysis.csv"
                    HASH_FILE = "comments_hash.txt"
                    COMMENTS_CSV = "REMACTO Comments.csv"

                    def generate_file_hash(path):
                        with open(path, "rb") as f:
                            return hashlib.md5(f.read()).hexdigest()

                    def save_current_hash(path, hash_path):
                        with open(hash_path, "w") as f:
                            f.write(generate_file_hash(path))

                    def should_reprocess_csv(path, hash_path):
                        if not os.path.exists(hash_path):
                            return True
                        with open(hash_path, "r") as f:
                            old_hash = f.read().strip()
                        return old_hash != generate_file_hash(path)

                    should_reprocess = should_reprocess_csv(COMMENTS_CSV, HASH_FILE)

                    # ---------------------------
                    # Initial GPT Analysis (first 20 rows)
                    # ---------------------------
                    if should_reprocess or not os.path.exists(CACHE_FILE):
                        st.warning("🧠 New data detected. Running fresh GPT analysis for initial 20 comments...")
                        analysis_data = []
                        with st.spinner("🔍 Analyzing first 20 comments with GPT..."):
                            for i, row in df_comments_proc.head(20).iterrows():
                                challenge = row["challenge"]
                                solution = row["proposed_solution"]
                                sentiment, polarity = gpt_arabic_sentiment_with_polarity(challenge)
                                summary = gpt_bullet_summary(challenge)
                                pros_cons = gpt_extract_pros_cons(solution)
                                topics = gpt_extract_topics(challenge)
                                analysis_data.append({
                                    "idea_id": row["idea_id"],
                                    "axis": row["axis"],
                                    "channel": row["channel"],
                                    "sentiment": sentiment,
                                    "polarity_score": polarity,
                                    "summary": summary,
                                    "pros": "; ".join(pros_cons.get("pros", [])),
                                    "cons": "; ".join(pros_cons.get("cons", [])),
                                    "topics": "; ".join(topics),
                                })
                        df_analysis = pd.DataFrame(analysis_data)
                        df_analysis.to_csv(CACHE_FILE, index=False)
                        save_current_hash(COMMENTS_CSV, HASH_FILE)
                        st.success("✅ GPT Analysis Complete and Cached.")
                    else:
                        st.success("✅ Using cached GPT analysis.")
                        df_analysis = pd.read_csv(CACHE_FILE)

                    # ---------------------------
                    # Feature 1: Process Additional Comments with Prompt for Number of Rows
                    # ---------------------------
                    if st.button("Process Additional Comments"):
                        num_cached = len(df_analysis)
                        total_comments = len(filtered_comments)
                        new_rows_available = total_comments - num_cached
                        if new_rows_available > 0:
                            st.write(f"There are {new_rows_available} unprocessed comments available.")
                            num_new_rows = st.number_input(
                                "How many new rows would you like to process?",
                                min_value=1,
                                max_value=new_rows_available,
                                value=new_rows_available,
                                step=1
                            )
                            new_analysis = []
                            with st.spinner("🔍 Analyzing additional comments with GPT..."):
                                for i, row in filtered_comments.iloc[num_cached:num_cached+num_new_rows].iterrows():
                                    challenge = row["challenge"]
                                    solution = row["proposed_solution"]
                                    sentiment, polarity = gpt_arabic_sentiment_with_polarity(challenge)
                                    summary = gpt_bullet_summary(challenge)
                                    pros_cons = gpt_extract_pros_cons(solution)
                                    topics = gpt_extract_topics(challenge)
                                    new_analysis.append({
                                        "idea_id": row["idea_id"],
                                        "axis": row["axis"],
                                        "channel": row["channel"],
                                        "sentiment": sentiment,
                                        "polarity_score": polarity,
                                        "summary": summary,
                                        "pros": "; ".join(pros_cons.get("pros", [])),
                                        "cons": "; ".join(pros_cons.get("cons", [])),
                                        "topics": "; ".join(topics),
                                    })
                            if new_analysis:
                                df_new = pd.DataFrame(new_analysis)
                                df_analysis = pd.concat([df_analysis, df_new], ignore_index=True)
                                df_analysis.to_csv(CACHE_FILE, index=False)
                                st.success(f"✅ Processed additional {len(df_new)} comments. Total analyzed: {len(df_analysis)}")
                            else:
                                st.info("No new comments to process.")
                        else:
                            st.info("All comments are already processed.")

                    # ---------------------------
                    # Feature 2: Sentiment Filter for Display
                    # ---------------------------
                    selected_sentiment = st.selectbox("Filter by Sentiment:", ["All", "POS", "NEG", "NEU"])
                    if selected_sentiment != "All":
                        df_display = df_analysis[df_analysis["sentiment"] == selected_sentiment]
                    else:
                        df_display = df_analysis

                    num_rows = st.slider("🔢 Number of Rows to Display", 1, min(50, len(df_display)), 5)
                    st.dataframe(df_display.head(num_rows))

                    # ---------------------------
                    # Feature 3: Analysis Summary Metrics
                    # ---------------------------
                    st.write("### Analysis Summary Metrics")
                    avg_polarity = df_analysis["polarity_score"].mean()
                    sentiment_summary = df_analysis["sentiment"].value_counts().to_dict()
                    st.write(f"Average Polarity Score: {avg_polarity:.2f}")
                    st.write("Sentiment Counts:", sentiment_summary)

                    # ---------------------------
                    # Feature 4: Download Full Analysis Report
                    # ---------------------------
                    csv_analysis = df_analysis.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Full Analysis Report", data=csv_analysis, file_name="full_gpt_analysis.csv", mime="text/csv")

                    # ---------------------------
                    # Feature 5: Top Extracted Topics
                    # ---------------------------
                    from collections import Counter
                    all_topics = []
                    for topics_str in df_analysis["topics"]:
                        topics_list = [t.strip() for t in topics_str.split(";") if t.strip()]
                        all_topics.extend(topics_list)
                    topic_counts = Counter(all_topics)
                    top_topics = topic_counts.most_common(5)
                    st.write("### Top Extracted Topics")
                    st.table(pd.DataFrame(top_topics, columns=["Topic", "Count"]))

                    # ---------------------------
                    # NEW FEATURE: Similarity Comparison using OpenAI Embeddings & Qdrant
                    # ---------------------------
                    st.write("### 🔍 Similarity Comparison Feature")
                    st.info("This feature computes embeddings for the 'challenge' texts using OpenAI's embedding model and stores them in Qdrant for similarity search.")

                    # Initialize Qdrant client (assumes Qdrant is running locally on port 6333)
                    qdrant_client = QdrantClient(host="localhost", port=6333)
                    collection_name = "remacto_comments"

                    # Button to update Qdrant embeddings
                    if st.button("Update Qdrant Embeddings"):
                        # Check if collection already exists
                        collections_info = qdrant_client.get_collections()
                        existing_collections = [col.name for col in collections_info.collections]
                        if collection_name in existing_collections:
                            st.info(f"Collection '{collection_name}' already exists. Updating embeddings without recreating the collection.")
                        else:
                            st.info(f"Collection '{collection_name}' does not exist. Creating new collection.")
                            qdrant_client.recreate_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(size=1536, distance="Cosine")
                            )

                        points = []
                        with st.spinner("Uploading embeddings to Qdrant..."):
                            for idx, row in filtered_comments.iterrows():
                                # Compute embedding for the challenge text using OpenAI embeddings
                                response = openai.Embedding.create(
                                    model="text-embedding-ada-002",
                                    input=row["challenge"]
                                )
                                embedding = response["data"][0]["embedding"]
                                point = PointStruct(
                                    id=int(row["idea_id"]),
                                    vector=embedding,
                                    payload={
                                        "challenge": row["challenge"],
                                        "axis": row["axis"],
                                        "channel": row["channel"]
                                    }
                                )
                                points.append(point)
                            qdrant_client.upsert(collection_name=collection_name, points=points)
                        st.success("✅ Embeddings updated in Qdrant.")

                    # Text input for similarity search
                    query_text = st.text_input("Enter a query to find similar comments:")
                    if st.button("Search Similar Comments") and query_text:
                        with st.spinner("Computing query embedding and searching..."):
                            response = openai.Embedding.create(
                                model="text-embedding-ada-002",
                                input=query_text
                            )
                            query_embedding = response["data"][0]["embedding"]
                            results = qdrant_client.search(
                                collection_name=collection_name,
                                query_vector=query_embedding,
                                limit=5
                            )
                            if results:
                                st.write("### Similar Comments Found:")
                                similar_df = pd.DataFrame([{
                                    "idea_id": r.id,
                                    "challenge": r.payload.get("challenge", ""),
                                    "axis": r.payload.get("axis", ""),
                                    "channel": r.payload.get("channel", ""),
                                    "score": r.score
                                } for r in results])
                                st.dataframe(similar_df)

                                # ---------------------------
                                # Visualization 1: Horizontal Bar Chart for Similarity Scores
                                # ---------------------------
                                st.write("#### Similarity Scores Bar Chart")
                                fig, ax = plt.subplots()
                                ax.barh(similar_df['idea_id'].astype(str), similar_df['score'], color='skyblue')
                                ax.set_xlabel("Similarity Score")
                                ax.set_ylabel("Idea ID")
                                st.pyplot(fig)

                                # ---------------------------
                                # Visualization 2: Pie Chart by Axis
                                # ---------------------------
                                st.write("#### Distribution by Axis")
                                axis_counts = similar_df["axis"].value_counts()
                                fig2, ax2 = plt.subplots()
                                ax2.pie(axis_counts, labels=axis_counts.index, autopct="%1.1f%%", startangle=140)
                                ax2.axis("equal")
                                st.pyplot(fig2)

                                # ---------------------------
                                # Visualization 3: Scatter Plot (Challenge Length vs. Similarity Score)
                                # ---------------------------
                                st.write("#### Scatter Plot: Challenge Length vs. Similarity Score")
                                similar_df["challenge_length"] = similar_df["challenge"].apply(lambda x: len(x))
                                fig3, ax3 = plt.subplots()
                                ax3.scatter(similar_df["challenge_length"], similar_df["score"], color="green")
                                ax3.set_xlabel("Challenge Text Length")
                                ax3.set_ylabel("Similarity Score")
                                st.pyplot(fig3)
                            else:
                                st.info("No similar comments found.")


                    # ---------------------------
                    # Existing Visualizations
                    # ---------------------------
                    st.write("#### 📉 Polarity Score Distribution")
                    fig1, ax1 = plt.subplots()
                    ax1.hist(df_analysis["polarity_score"], bins=10, color="skyblue")
                    ax1.set_title("Polarity Score Distribution")
                    ax1.set_xlabel("Score (-1 = negative, +1 = positive)")
                    ax1.set_ylabel("Count")
                    st.pyplot(fig1)

                    st.write("#### 🥧 Sentiment Distribution")
                    sentiment_counts = df_analysis["sentiment"].value_counts()
                    fig2, ax2 = plt.subplots()
                    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140)
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    st.write(f"#### ☁️ Word Cloud (Challenges) in {lang}")
                    plot_wordcloud(
                        df_comments_proc["challenge"].tolist(),
                        f"Challenges Word Cloud ({lang})",
                        target_language="English" if lang in ["English", "Darija"] else lang
                    )




            # -----------------------------------------------------------------
            # TAB 2: Projects
            # -----------------------------------------------------------------

            with tabs[1]:
                st.header(L["projects_header"])
                
                if df_projects.empty:
                    st.warning(L["no_projects_msg"])
                else:
                    # Display original data preview
                    st.write(f"### {L['projects_data_preview']}")
                    st.dataframe(df_projects.head(10))
                    
                    # ---------------------------
                    # Filter projects by Region (CT)
                    # ---------------------------
                    ct_options = ["All"] + list(df_projects["CT"].dropna().unique())
                    selected_CT = st.selectbox("Filter by Region (CT):", ct_options)
                    if selected_CT != "All":
                        filtered_projects = df_projects[df_projects["CT"] == selected_CT]
                    else:
                        filtered_projects = df_projects.copy()
                    st.write(f"### Projects for Region: {selected_CT}")
                    st.dataframe(filtered_projects)
                    
                    # ---------------------------
                    # Caching strategy for project summaries
                    # ---------------------------
                    CACHE_FILE_PROJECTS = "cached_projects_summaries.csv"
                    HASH_FILE_PROJECTS = "projects_hash.txt"
                    PROJECTS_CSV = "REMACTO Projects.csv"

                    def generate_file_hash_projects(path):
                        with open(path, "rb") as f:
                            return hashlib.md5(f.read()).hexdigest()

                    def save_current_hash_projects(path, hash_path):
                        with open(hash_path, "w") as f:
                            f.write(generate_file_hash_projects(path))

                    def should_reprocess_csv_projects(path, hash_path):
                        if not os.path.exists(hash_path):
                            return True
                        with open(hash_path, "r") as f:
                            old_hash = f.read().strip()
                        return old_hash != generate_file_hash_projects(path)

                    should_reprocess_projects = should_reprocess_csv_projects(PROJECTS_CSV, HASH_FILE_PROJECTS)

                    # ---------------------------
                    # Compute or load project summaries
                    # ---------------------------
                    if should_reprocess_projects or not os.path.exists(CACHE_FILE_PROJECTS):
                        st.warning("🧠 New project data detected. Running fresh GPT analysis for project summaries...")
                        # Allow the citizen to choose how many projects to summarize
                        max_rows_proj = st.slider("Number of Projects to Summarize", 1, len(filtered_projects), 5)
                        project_summaries = []
                        with st.spinner("Summarizing project themes..."):
                            for idx in range(max_rows_proj):
                                row = filtered_projects.iloc[idx]
                                theme_text = row["themes"]
                                bullet_sum = gpt_bullet_summary(theme_text)
                                project_summaries.append({
                                    "title": row["title"],
                                    "themes": theme_text,
                                    "bullet_summary": bullet_sum,
                                })
                        df_proj_summary = pd.DataFrame(project_summaries)
                        df_proj_summary.to_csv(CACHE_FILE_PROJECTS, index=False)
                        save_current_hash_projects(PROJECTS_CSV, HASH_FILE_PROJECTS)
                    else:
                        st.success("✅ Using cached project summaries.")
                        df_proj_summary = pd.read_csv(CACHE_FILE_PROJECTS)

                    st.write(f"### {L['summaries_of_themes']}")
                    st.dataframe(df_proj_summary)

                    # ---------------------------
                    # Quick bar chart: Projects by CT
                    # ---------------------------
                    st.write("### Projects by CT")
                    ct_counts = df_projects["CT"].value_counts()
                    st.bar_chart(ct_counts)

                    # ---------------------------
                    # Download Projects CSV for transparency
                    # ---------------------------
                    csv_data_projects = df_projects.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Projects CSV",
                        data=csv_data_projects,
                        file_name="REMACTO_Projects.csv",
                        mime="text/csv"
                    )

                    # ---------------------------
                    # Citizen Participation: Provide feedback on projects
                    # ---------------------------
                    st.write("### 🗣️ Citizen Participation")
                    with st.expander("💬 Provide your comments or suggestions on projects"):
                        citizen_feedback = st.text_area("Your Comment/Suggestion", placeholder="Share your ideas to improve projects...")
                        if st.button("Submit Comment"):
                            if citizen_feedback.strip():
                                # Save feedback using the provided storage function
                                store_user_input_in_csv(username, "project_feedback", citizen_feedback)
                                st.success("Thank you! Your feedback has been recorded.")
                            else:
                                st.warning("Please enter a comment before submitting.")
            # -----------------------------------------------------------------
            # TAB 3: Proposals & Feedback
            # -----------------------------------------------------------------
            with tabs[2]:
                st.header("📝 Submit a New Proposal or Feedback (Extended)")

                st.write("""
                In this section, you can choose a **Collectivité Territoriale (CT)**, 
                then a specific **commune**, and finally a **project** to give 
                more targeted feedback or a new proposal.
                """)

                # 1) Choose CT
                ct_list = df_projects["CT"].dropna().unique().tolist()
                selected_ct = st.selectbox("Select Region (CT)", ["-- Choose a Region --"] + ct_list)

                if selected_ct != "-- Choose a Region --":
                    # Feature 1: Cache filtered projects by CT
                    @st.cache_data(show_spinner=False)
                    def filter_by_ct(ct):
                        return df_projects[df_projects["CT"] == ct]
                    df_ct = filter_by_ct(selected_ct)

                    # 2) List possible communes for that CT
                    communes = df_ct["collectivite_territoriale"].dropna().unique().tolist()
                    selected_commune = st.selectbox("Select Commune", ["-- Choose a Commune --"] + communes)

                    if selected_commune != "-- Choose a Commune --":
                        # Feature 2: Cache filtered projects by commune
                        @st.cache_data(show_spinner=False)
                        def filter_by_commune(df, commune):
                            return df[df["collectivite_territoriale"] == commune]
                        df_commune = filter_by_commune(df_ct, selected_commune)
                        projects_list = df_commune["title"].dropna().unique().tolist()

                        selected_project = st.selectbox("Select Project to Provide Feedback On:", ["-- Choose a Project --"] + projects_list)

                        if selected_project != "-- Choose a Project --":
                            # Feature 3: Expandable Project Details for transparency
                            project_row = df_commune[df_commune["title"] == selected_project].iloc[0]
                            with st.expander("View Full Project Details"):
                                st.write(project_row.to_dict())

                            st.write(f"**Project Title**: {project_row['title']}")
                            st.write(f"**Themes**: {project_row['themes']}")
                            st.write("Feel free to comment on how to improve or any suggestions you have for this specific project.")

                            # Feature 4: Display Related Comments from REMACTO Comments dataset
                            @st.cache_data(show_spinner=False)
                            def get_related_comments(project_theme):
                                # A simple keyword-based filter using words from the theme
                                keywords = project_theme.split()
                                related = df_comments[df_comments["challenge"].astype(str).str.contains("|".join(keywords), case=False, na=False)]
                                return related.head(10)
                            related_comments = get_related_comments(project_row["themes"])
                            if not related_comments.empty:
                                with st.expander("View Related Citizen Comments"):
                                    st.dataframe(related_comments)
                            else:
                                st.info("No related citizen comments found.")

                            # Feature 5: Proposals & Feedback Submission with caching of user inputs
                            st.subheader("New Proposal (Optional)")
                            proposal_title = st.text_input("Proposal Title", placeholder="e.g. Create more green spaces")
                            proposal_description = st.text_area("Proposal Description", placeholder="Describe your idea in detail...")
                            if st.button("Submit Proposal"):
                                if proposal_title.strip() and proposal_description.strip():
                                    content = (
                                        f"CT: {selected_ct}\n"
                                        f"Commune: {selected_commune}\n"
                                        f"Project: {selected_project}\n"
                                        f"Proposal Title: {proposal_title}\n"
                                        f"Proposal Description: {proposal_description}"
                                    )
                                    store_user_input_in_csv(username, "proposal", content)
                                    st.success("Proposal submitted successfully!")
                                else:
                                    st.warning("Please provide both proposal title and description.")

                            st.subheader("Feedback (Optional)")
                            feedback_text = st.text_area("Any specific feedback about this project?",
                                                        placeholder="Write your feedback here...")
                            if st.button("Send Feedback"):
                                if feedback_text.strip():
                                    feedback_content = (
                                        f"CT: {selected_ct}\n"
                                        f"Commune: {selected_commune}\n"
                                        f"Project: {selected_project}\n"
                                        f"Feedback: {feedback_text}"
                                    )
                                    store_user_input_in_csv(username, "feedback", feedback_content)
                                    st.success("Your feedback has been recorded.")
                                else:
                                    st.warning("Please enter some feedback.")

                            # Feature 6: Show aggregated count of proposals and feedback already submitted for this project
                            @st.cache_data(show_spinner=False)
                            def get_project_submission_counts(project_title):
                                if os.path.exists("user_inputs.csv"):
                                    df_inputs = pd.read_csv("user_inputs.csv")
                                    # Filter rows containing the project title (case insensitive)
                                    df_proj_inputs = df_inputs[df_inputs["content"].str.contains(project_title, case=False, na=False)]
                                    return df_proj_inputs["input_type"].value_counts().to_dict()
                                else:
                                    return {}
                            submission_counts = get_project_submission_counts(selected_project)
                            st.write("### Aggregated Submissions for This Project")
                            st.write(submission_counts if submission_counts else "No proposals/feedback submitted yet.")

                            # Feature 7: Voting mechanism (simulate upvotes/downvotes stored in session state)
                            if "votes" not in st.session_state:
                                st.session_state.votes = {"up": 0, "down": 0}
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("👍 Upvote"):
                                    st.session_state.votes["up"] += 1
                            with col2:
                                st.write("Votes:")
                                st.write(f"👍 {st.session_state.votes['up']} | 👎 {st.session_state.votes['down']}")
                            with col3:
                                if st.button("👎 Downvote"):
                                    st.session_state.votes["down"] += 1

                            # Feature 8: Share Project Feature – generate a simulated shareable link
                            def slugify(text):
                                return text.lower().replace(" ", "-")
                            share_link = f"https://remacto.org/project/{slugify(selected_project)}"
                            st.write("### Share This Project")
                            st.code(share_link, language="plaintext")
                            st.info("Copy the above link to share this project with your network.")

                            # Feature 9: Feedback Summary from Related Comments – aggregate basic sentiment (dummy example)
                            # (Assuming df_comments contains rich text, we simulate sentiment count based on keywords)
                            @st.cache_data(show_spinner=False)
                            def sentiment_summary(related_df):
                                pos = related_df["challenge"].str.contains("جيد|ممتاز|إيجابي", case=False, na=False).sum()
                                neg = related_df["challenge"].str.contains("سيئ|سلبي|متعب", case=False, na=False).sum()
                                neu = len(related_df) - pos - neg
                                return {"Positive": pos, "Negative": neg, "Neutral": neu}
                            sentiment_stats = sentiment_summary(related_comments)
                            st.write("### Sentiment Summary of Related Comments")
                            st.write(sentiment_stats)

                        else:
                            st.info("Please select a specific project to provide feedback or propose an idea.")
                    else:
                        st.info("Please select a commune from this region.")
                else:
                    st.info("Please start by choosing a Collectivité Territoriale (CT).")



            # -----------------------------------------------------------------
            # TAB 4: Extra Visualizations
            # -----------------------------------------------------------------
            with tabs[3]:
                st.header(L["extra_visualizations_tab"])

                if df_comments.empty:
                    st.info(L["no_comments_msg"])
                else:
                    # ---------------------------
                    # Caching helper functions for counts and text lengths
                    # ---------------------------
                    @st.cache_data(show_spinner=False)
                    def get_axis_counts(df):
                        return df["axis"].value_counts()

                    @st.cache_data(show_spinner=False)
                    def get_channel_counts(df):
                        return df["channel"].value_counts()

                    @st.cache_data(show_spinner=False)
                    def compute_text_lengths(text_series):
                        return text_series.apply(lambda x: len(str(x)))

                    # ---------------------------
                    # Visualization 1: Axis Distribution (Bar Chart)
                    # ---------------------------
                    axis_counts = get_axis_counts(df_comments)
                    st.write("### Axis Distribution (Bar Chart)")
                    st.bar_chart(axis_counts)

                    # ---------------------------
                    # Visualization 2: Channel Distribution (Pie Chart)
                    # ---------------------------
                    channel_counts = get_channel_counts(df_comments)
                    st.write("### Channels (Pie Chart)")
                    fig_c, ax_c = plt.subplots()
                    ax_c.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                    ax_c.axis("equal")
                    st.pyplot(fig_c)

                    # ---------------------------
                    # Visualization 3: Word Cloud of Proposed Solutions
                    # ---------------------------
                    st.write(f"### Word Cloud of Proposed Solutions (in {lang})")
                    plot_wordcloud(
                        df_comments["proposed_solution"].astype(str).tolist(),
                        f"Proposed Solutions ({lang})",
                        target_language="English" if lang in ["English", "Darija"] else lang
                    )

                    # ---------------------------
                    # Visualization 4: Distribution of Proposed Solutions Text Length
                    # ---------------------------
                    text_lengths = compute_text_lengths(df_comments["proposed_solution"])
                    st.write("### Distribution of Proposed Solutions Text Length")
                    fig_length, ax_length = plt.subplots()
                    ax_length.hist(text_lengths, bins=20, color="skyblue")
                    ax_length.set_title("Text Length Distribution")
                    ax_length.set_xlabel("Number of Characters")
                    ax_length.set_ylabel("Count")
                    st.pyplot(fig_length)

                    # ---------------------------
                    # Visualization 5: Word Cloud of Challenges
                    # ---------------------------
                    st.write(f"### Word Cloud of Challenges (in {lang})")
                    plot_wordcloud(
                        df_comments["challenge"].astype(str).tolist(),
                        f"Challenges Word Cloud ({lang})",
                        target_language="English" if lang in ["English", "Darija"] else lang
                    )

            # -----------------------------------------------------------------
            # TAB 5: All User Inputs
            # -----------------------------------------------------------------
            with tabs[4]:
                st.header(L["all_user_inputs_tab"])
                csv_file = "user_inputs.csv"
                
                if not os.path.exists(csv_file):
                    st.info("No user inputs stored yet.")
                else:
                    # Feature 1: Cache the CSV reading to avoid reloading data
                    @st.cache_data(show_spinner=False)
                    def load_user_inputs(path):
                        df = pd.read_csv(path)
                        # Ensure proper datetime conversion
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        return df
                    
                    df_user_inputs = load_user_inputs(csv_file)
                    
                    # Feature 2: Admin can filter inputs by username
                    if role == "admin":
                        usernames = df_user_inputs["username"].dropna().unique().tolist()
                        selected_user = st.selectbox("Filter by Username:", ["All"] + usernames)
                        if selected_user != "All":
                            df_filtered = df_user_inputs[df_user_inputs["username"] == selected_user]
                        else:
                            df_filtered = df_user_inputs.copy()
                    else:
                        df_filtered = df_user_inputs[df_user_inputs["username"] == username]
                        st.write(f"Showing inputs for user: **{username}**")
                    
                    st.dataframe(df_filtered)
                    
                    # Feature 3: Search functionality to filter user inputs by keyword
                    search_keyword = st.text_input("Search Inputs:", "")
                    if search_keyword:
                        df_search = df_filtered[df_filtered["content"].str.contains(search_keyword, case=False, na=False)]
                        st.write(f"### Search Results for '{search_keyword}':")
                        st.dataframe(df_search)
                    else:
                        df_search = df_filtered.copy()
                    
                    # Feature 4: Date range filter for inputs
                    if not df_search.empty and df_search["timestamp"].notnull().all():
                        min_date = df_search["timestamp"].min().date()
                        max_date = df_search["timestamp"].max().date()
                        date_range = st.date_input("Select Date Range:", [min_date, max_date])
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            df_date_filtered = df_search[
                                (df_search["timestamp"] >= pd.Timestamp(start_date)) &
                                (df_search["timestamp"] <= pd.Timestamp(end_date))
                            ]
                            st.write(f"### Inputs from {start_date} to {end_date}:")
                            st.dataframe(df_date_filtered)
                        else:
                            df_date_filtered = df_search.copy()
                    else:
                        df_date_filtered = df_search.copy()
                    
                    # Feature 5: Summary statistics – count of inputs by type
                    st.write("### Summary Statistics")
                    input_type_counts = df_date_filtered["input_type"].value_counts()
                    st.write("Count by Input Type:")
                    st.dataframe(input_type_counts.to_frame().reset_index().rename(columns={"index": "Input Type", "input_type": "Count"}))
                    
                    # Feature 6a: Time Series Chart of inputs over time
                    st.write("### Time Series of Inputs")
                    df_time = df_date_filtered.copy().dropna(subset=["timestamp"])
                    if not df_time.empty:
                        df_time.set_index("timestamp", inplace=True)
                        daily_counts = df_time.resample("D").size()
                        st.line_chart(daily_counts)
                    else:
                        st.info("No valid timestamps to plot time series.")
                    
                    # Feature 6b: Pie Chart of Input Types Distribution
                    st.write("### Input Types Distribution")
                    fig, ax = plt.subplots()
                    ax.pie(input_type_counts.values, labels=input_type_counts.index, autopct="%1.1f%%", startangle=140)
                    ax.axis("equal")
                    st.pyplot(fig)
                    
                    # Feature 7: Download full CSV for transparency
                    st.write("### Export Citizen Inputs as CSV")
                    csv_data = df_user_inputs.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="user_inputs_all.csv",
                        mime="text/csv"
                    )


        else:
            st.warning(L["token_invalid"])
    else:
        st.info(L["no_token_msg"])


# If running standalone
if __name__ == "__main__":
    main()
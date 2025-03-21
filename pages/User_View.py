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
        "title": "📊 User Dashboard",
        "header_comments": "💬 Citizen Comments (REMACTO)",
        "label_normalize": "🧹 Apply Basic Arabic Normalization (Optional)",
        "analysis_section": "🧠 GPT-Based Sentiment & Summaries + Polarity",
        "proposal_header": "📝 Submit a New Proposal or Feedback",
        "proposal_title_label": "📌 Proposal Title",
        "proposal_description_label": "🧾 Proposal Description",
        "proposal_submit_button": "📤 Submit Proposal",
        "feedback_label": "💭 Your Feedback",
        "feedback_button": "📬 Send Feedback",
        "logout_button": "🔓 Logout",
        "no_comments_msg": "⚠️ No REMACTO Comments available.",
        "original_data_label": "📋 Original Data (first 10 rows):",
        "norm_success": "✅ Text normalization applied.",
        "no_token_msg": "⚠️ No token found in session. Please go back and log in.",
        "token_invalid": "❌ Token is invalid or expired. Please log in again.",
        "logged_in_as": "✅ You are logged in as:",
        "role_label": "(Role: ",
        "closing_paren": ")",
        "projects_header": "🏗️ Municipal Projects (REMACTO)",
        "no_projects_msg": "⚠️ No REMACTO Projects available.",
        "projects_data_preview": "📂 Projects Data (Preview)",
        "summaries_of_themes": "📝 Summaries of Project Themes",
        "proposals_feedback_tab": "🗳️ Submit a New Proposal or Feedback",
        "extra_visualizations_tab": "📈 Extra Visualizations & Analysis",
        "all_user_inputs_tab": "🗃️ All Stored Inputs from Citizens"
    },
    "Arabic": {
        "title": "📊 لوحة المستخدم",
        "header_comments": "💬 تعليقات المواطنين (ريماكتو)",
        "label_normalize": "🧹 تطبيق تنقيح بسيط للنص العربي (اختياري)",
        "analysis_section": "🧠 تحليل المشاعر والتلخيص باستخدام GPT",
        "proposal_header": "📝 إضافة اقتراح جديد أو ملاحظات",
        "proposal_title_label": "📌 عنوان الاقتراح",
        "proposal_description_label": "🧾 وصف الاقتراح",
        "proposal_submit_button": "📤 إرسال الاقتراح",
        "feedback_label": "💭 ملاحظاتك",
        "feedback_button": "📬 إرسال الملاحظات",
        "logout_button": "🔓 تسجيل الخروج",
        "no_comments_msg": "⚠️ لا توجد تعليقات ريماكتو متاحة.",
        "original_data_label": "📋 البيانات الأصلية (أول 10 صفوف):",
        "norm_success": "✅ تم تطبيق تنقيح النص.",
        "no_token_msg": "⚠️ لا يوجد رمز في الجلسة. يرجى العودة وتسجيل الدخول.",
        "token_invalid": "❌ الرمز غير صالح أو منتهي. يرجى تسجيل الدخول مجددًا.",
        "logged_in_as": "✅ تم تسجيل الدخول باسم:",
        "role_label": "(الدور: ",
        "closing_paren": ")",
        "projects_header": "🏗️ مشاريع البلدية (ريماكتو)",
        "no_projects_msg": "⚠️ لا توجد مشاريع ريماكتو متاحة.",
        "projects_data_preview": "📂 عرض بيانات المشاريع",
        "summaries_of_themes": "📝 تلخيص مواضيع المشاريع",
        "proposals_feedback_tab": "🗳️ إضافة اقتراح أو ملاحظات",
        "extra_visualizations_tab": "📈 تصورات وتحليلات إضافية",
        "all_user_inputs_tab": "🗃️ جميع المدخلات المخزنة من المواطنين"
    },
    "French": {
        "title": "📊 Tableau de bord de l'utilisateur",
        "header_comments": "💬 Commentaires des citoyens (REMACTO)",
        "label_normalize": "🧹 Appliquer une normalisation de l'arabe (optionnel)",
        "analysis_section": "🧠 Analyse de sentiment et résumés GPT + polarité",
        "proposal_header": "📝 Soumettre une nouvelle proposition ou un retour",
        "proposal_title_label": "📌 Titre de la proposition",
        "proposal_description_label": "🧾 Description de la proposition",
        "proposal_submit_button": "📤 Soumettre la proposition",
        "feedback_label": "💭 Vos commentaires",
        "feedback_button": "📬 Envoyer le commentaire",
        "logout_button": "🔓 Se déconnecter",
        "no_comments_msg": "⚠️ Aucun commentaire REMACTO disponible.",
        "original_data_label": "📋 Données d'origine (10 premières lignes):",
        "norm_success": "✅ Normalisation du texte appliquée.",
        "no_token_msg": "⚠️ Aucun jeton trouvé dans la session. Veuillez vous reconnecter.",
        "token_invalid": "❌ Jeton invalide ou expiré. Veuillez vous reconnecter.",
        "logged_in_as": "✅ Connecté en tant que:",
        "role_label": "(Rôle: ",
        "closing_paren": ")",
        "projects_header": "🏗️ Projets Municipaux (REMACTO)",
        "no_projects_msg": "⚠️ Aucun projet REMACTO disponible.",
        "projects_data_preview": "📂 Aperçu des données du projet",
        "summaries_of_themes": "📝 Résumés des thèmes du projet",
        "proposals_feedback_tab": "🗳️ Propositions et retour",
        "extra_visualizations_tab": "📈 Visualisations supplémentaires",
        "all_user_inputs_tab": "🗃️ Toutes les entrées des citoyens"
    },
    "Darija": {
        "title": "📊 لوحة المستخدم بالدارجة",
        "header_comments": "💬 تعليقات الناس (ريماكتو)",
        "label_normalize": "🧹 نقّي النص العربي شوية (اختياري)",
        "analysis_section": "🧠 تحليل المشاعر مع GPT + البولاريتي",
        "proposal_header": "📝 زيد اقتراح جديد ولا شي ملاحظة",
        "proposal_title_label": "📌 عنوان الاقتراح بالدارجة",
        "proposal_description_label": "🧾 وصف الاقتراح بالتفاصيل",
        "proposal_submit_button": "📤 صيفط الاقتراح",
        "feedback_label": "💭 تعطينا رأيك",
        "feedback_button": "📬 صيفط رأيك",
        "logout_button": "🔓 خروج",
        "no_comments_msg": "⚠️ ماكايناش تعليقات ريماكتو دابا.",
        "original_data_label": "📋 البيانات الأصلية (أول 10 صفوف):",
        "norm_success": "✅ تصاوابات تنقية النص.",
        "no_token_msg": "⚠️ ماكينش التوكن فالسيشن. رجع سيني.",
        "token_invalid": "❌ التوكن خايب ولا سالا. خصك تسيني.",
        "logged_in_as": "✅ نتا داخل باسم:",
        "role_label": "(دور: ",
        "closing_paren": ")",
        "projects_header": "🏗️ مشاريع الجماعة (ريماكتو)",
        "no_projects_msg": "⚠️ ماكاين لا مشاريع لا والو.",
        "projects_data_preview": "📂 شوف بيانات المشاريع",
        "summaries_of_themes": "📝 لخص مواضيع المشاريع",
        "proposals_feedback_tab": "🗳️ زيد اقتراح ولا ملاحظة",
        "extra_visualizations_tab": "📈 تصاور وتحليلات زوينة",
        "all_user_inputs_tab": "🗃️ كلشي ديال المدخلات ديال الناس"
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

                    # Check for reprocessing
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

                    # Perform GPT processing or load from cache
                    if should_reprocess or not os.path.exists(CACHE_FILE):
                        st.warning("🧠 New data detected. Running fresh GPT analysis...")

                        analysis_data = []
                        with st.spinner("🔍 Analyzing comments with GPT..."):
                            for i, row in df_comments_proc.iterrows():
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

                    # Slider for row display
                    num_rows = st.slider("🔢 Number of Rows to Display", 1, min(50, len(df_analysis)), 5)
                    st.dataframe(df_analysis.head(num_rows))

                    # Visualization 1: Polarity Histogram
                    st.write("#### 📉 Polarity Score Distribution")
                    fig1, ax1 = plt.subplots()
                    ax1.hist(df_analysis["polarity_score"], bins=10, color="skyblue")
                    ax1.set_title("Polarity Score Distribution")
                    ax1.set_xlabel("Score (-1 = negative, +1 = positive)")
                    ax1.set_ylabel("Count")
                    st.pyplot(fig1)

                    # Visualization 2: Sentiment Pie Chart
                    st.write("#### 🥧 Sentiment Distribution")
                    sentiment_counts = df_analysis["sentiment"].value_counts()
                    fig2, ax2 = plt.subplots()
                    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140)
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    # Visualization 3: Word Cloud
                    st.write(f"#### ☁️ Word Cloud (Challenges) in {lang}")
                    plot_wordcloud(
                        df_comments_proc["challenge"].astype(str).tolist(),
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
                    st.write(f"### {L['projects_data_preview']}")
                    st.dataframe(df_projects.head(10))

                    st.write(f"### {L['summaries_of_themes']}")
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

                    # Quick bar chart
                    st.write("### Projects by CT")
                    ct_counts = df_projects["CT"].value_counts()
                    st.bar_chart(ct_counts)

            # -----------------------------------------------------------------
            # TAB 3: Proposals & Feedback
            # -----------------------------------------------------------------
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
                    # 2) Filter projects by that CT
                    df_ct = df_projects[df_projects["CT"] == selected_ct]

                    # 3) List possible communes for that region
                    communes = df_ct["collectivite_territoriale"].dropna().unique().tolist()
                    selected_commune = st.selectbox("Select Commune", ["-- Choose a Commune --"] + communes)

                    if selected_commune != "-- Choose a Commune --":
                        # 4) Filter to show the projects in that commune
                        df_commune = df_ct[df_ct["collectivite_territoriale"] == selected_commune]
                        projects_list = df_commune["title"].dropna().unique().tolist()

                        selected_project = st.selectbox("Select Project to Provide Feedback On:", ["-- Choose a Project --"] + projects_list)

                        if selected_project != "-- Choose a Project --":
                            # 5) Provide a text area for user to see the details or 'themes'
                            project_row = df_commune[df_commune["title"] == selected_project].iloc[0]
                            st.write(f"**Project Title**: {project_row['title']}")
                            st.write(f"**Themes**: {project_row['themes']}")

                            st.write("Feel free to comment on how to improve or any suggestions you have for this specific project.")
                            
                            st.subheader("New Proposal (Optional)")
                            proposal_title = st.text_input("Proposal Title", placeholder="e.g. Create more green spaces")
                            proposal_description = st.text_area("Proposal Description", placeholder="Describe your idea in detail...")

                            if st.button("Submit Proposal"):
                                if proposal_title.strip() and proposal_description.strip():
                                    # Store the proposal with region/commune/project context
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
                    # Axis distribution
                    axis_counts = df_comments["axis"].value_counts()
                    st.write("### Axis Distribution (Bar Chart)")
                    st.bar_chart(axis_counts)

                    # Channel distribution
                    channel_counts = df_comments["channel"].value_counts()
                    st.write("### Channels (Pie Chart)")
                    fig_c, ax_c = plt.subplots()
                    ax_c.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                    ax_c.axis("equal")
                    st.pyplot(fig_c)

                    st.write(f"### Word Cloud of Proposed Solutions (in {lang})")
                    plot_wordcloud(
                        df_comments["proposed_solution"].astype(str).tolist(),
                        f"Proposed Solutions ({lang})",
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
                    df_user_inputs = pd.read_csv(csv_file)
                    st.dataframe(df_user_inputs)

                    if role != "admin":
                        df_user_specific = df_user_inputs[df_user_inputs["username"] == username]
                        st.write(f"Showing inputs for user: **{username}**")
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
            st.warning(L["token_invalid"])
    else:
        st.info(L["no_token_msg"])


# If running standalone
if __name__ == "__main__":
    main()
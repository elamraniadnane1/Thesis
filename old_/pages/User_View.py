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
        "all_user_inputs_tab": "🗃️ All Citizen Inputs",
        "advanced_tab": "🚀 Advanced Participation & Matching"
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
        "all_user_inputs_tab": "🗃️ جميع المدخلات من المواطنين",
        "advanced_tab": "🚀 المشاركة والتوفيق المتقدم"
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
        "all_user_inputs_tab": "🗃️ Toutes les entrées des citoyens",
        "advanced_tab": "🚀 Participation et Correspondances Avancées"
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
        "all_user_inputs_tab": "🗃️ جميع المدخلات ديال الناس",
        "advanced_tab": "🚀 المشاركة والتوفيق المتقدم"
    }
}

##############################################################################
# 4) GPT Initialization + Language Dictionary
##############################################################################
def init_gpt():
    if not openai.api_key:
        openai.api_key = st.secrets["openai"]["api_key"]

##############################################################################
# 5) Utility Functions (Normalization, Chunking, GPT calls)
##############################################################################
def normalize_arabic(text: str) -> str:
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
    text = str(text).strip()  # Ensure conversion to string
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
    if not str(text).strip():
        return "لا يوجد نص للخلاصة."
    prompt = f"""
    لخص النقاط الأساسية في النص التالي باللغة العربية عبر نقاط (bullet points):
    النص:
    {str(text)}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in summarizing Arabic text into bullet points."},
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
    if not str(text).strip():
        return {"pros": [], "cons": []}
    user_msg = f"""
    اقرأ النص التالي بالعربية، واستخرج أهم النقاط الإيجابية (Pros) والنقاط السلبية (Cons):
    النص:
    {str(text)}

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
            low_line = str(line).lower().strip()
            if low_line.startswith("pros"):
                current_section = "pros"
                continue
            elif low_line.startswith("cons"):
                current_section = "cons"
                continue
            elif str(line).strip().startswith("-"):
                if current_section == "pros":
                    pros.append(str(line).lstrip("-").strip())
                elif current_section == "cons":
                    cons.append(str(line).lstrip("-").strip())
        return {"pros": pros, "cons": cons}
    except Exception as e:
        st.warning(f"GPT Pros/Cons Error: {e}")
        return {"pros": [], "cons": []}

def gpt_extract_topics(text: str) -> list:
    if not str(text).strip():
        return []
    user_msg = f"""
    استخرج المواضيع الأساسية المذكورة في النص:
    {str(text)}
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
            line = str(line).strip("-•123456789). ").strip()
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
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def gpt_translate_arabic(text: str, target_language: str = "English") -> str:
    text = str(text).strip()
    if not text:
        return ""
    max_overall_length = 6000
    if len(text) > max_overall_length:
        lines = text.split('\n')
        random.shuffle(lines)
        lines = lines[:200]
        text = "\n".join(lines)[:max_overall_length]
    text_chunks = chunk_text(text, chunk_size=1500)
    translated_chunks = []
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
            translated_chunks.append(response["choices"][0]["message"]["content"].strip())
        except Exception as e:
            st.warning(f"GPT Translate Error on chunk: {e}")
            continue
    return " ".join(translated_chunks)

##############################################################################
# 5) Load CSV Data
##############################################################################
def load_remacto_comments(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        df.columns = ["idea_id", "channel", "axis", "challenge", "proposed_solution"]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Comments CSV: {e}")
        return pd.DataFrame()

def load_remacto_projects(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        df.columns = ["title", "CT", "collectivite_territoriale", "themes"]
        return df
    except Exception as e:
        st.error(f"Error loading REMACTO Projects CSV: {e}")
        return pd.DataFrame()

##############################################################################
# 6) Wordcloud (with GPT-based translation to chosen language)
##############################################################################
def plot_wordcloud(texts: list, title: str, target_language: str = "English"):
    joined_text_ar = "\n".join(texts).strip()
    if not joined_text_ar:
        st.warning("No text found to generate wordcloud.")
        return
    with st.spinner(f"Translating text to {target_language} for WordCloud (may sample if data is huge)..."):
        translated_text = gpt_translate_arabic(joined_text_ar, target_language)
    if not translated_text.strip():
        st.warning("Translation returned empty. Cannot generate WordCloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(translated_text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    st.pyplot(fig)

##############################################################################
# 7) Store Citizen Inputs
##############################################################################
def store_user_input_in_csv(username: str, input_type: str, content: str):
    timestamp = datetime.now().isoformat()
    row = {"timestamp": timestamp, "username": username, "input_type": input_type, "content": content}
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
    init_gpt()
    lang = st.session_state.get("site_language", "English")
    if lang not in ui_texts:
        lang = "English"
    L = ui_texts[lang]

    # ----------------------- SIDEBAR -----------------------
    st.sidebar.title("لوحة جانبية (Side Bar)")
    st.sidebar.markdown("### 🤝 مرحباً بكم في المنصة!")
    st.sidebar.markdown("""
    هذه اللوحة الجانبية تقدم مجموعة من الأدوات:
    - الإرشادات الصوتية
    - تكبير الخط
    - استخدام الأيقونات البصرية
    """)
    with st.sidebar.expander("🦻 إعدادات الوصول"):
        font_size = st.selectbox("حجم الخط:", ["صغير", "متوسط", "كبير"])
        st.write("يمكنك اختيار حجم الخط المناسب.")
        high_contrast = st.checkbox("وضع تباين عالٍ")
        st.write("هذا الوضع يرفع من وضوح العناصر للأشخاص ذوي القدرة المحدودة على الرؤية.")
    with st.sidebar.expander("🔊 مساعدة صوتية (TTS)"):
        st.write("للمستخدمين الذين لا يقرؤون العربية بسهولة، يمكنهم الاستماع إلى النصوص الأساسية.")
        if st.button("تشغيل المساعدة الصوتية"):
            st.info("🚧 يجري تشغيل المساعدة الصوتية... (نموذج تجريبي)")
        if st.button("إيقاف"):
            st.info("🚧 تم إيقاف المساعدة الصوتية.")
    st.sidebar.markdown("### 🏷️ رموز بصرية مساعدة:")
    st.sidebar.write("- 📊 للبيانات العامة")
    st.sidebar.write("- 📝 للإقتراحات")
    st.sidebar.write("- 💭 للملاحظات")
    st.sidebar.write("- 🔓 لتسجيل الخروج")
    with st.sidebar.expander("🌐 تغيير اللغة"):
        chosen_lang = st.selectbox("اختر لغة العرض", ["Arabic", "English", "French", "Darija"], index=0)
        if st.button("تطبيق اللغة"):
            st.session_state.site_language = chosen_lang
            st.experimental_rerun()

    st.title(L["title"])
    st.write("Welcome to your personal dashboard! Engage with projects, share feedback, see analytics, etc.")

    token = st.session_state.get("jwt_token", None)
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"{L['logged_in_as']} **{username}** {L['role_label']}{role}{L['closing_paren']}")
            if st.button(L["logout_button"]):
                st.session_state.jwt_token = None
                st.experimental_rerun()

            comments_csv_path = "REMACTO Comments.csv"
            projects_csv_path = "REMACTO Projects.csv"
            df_comments = load_remacto_comments(comments_csv_path)
            df_projects = load_remacto_projects(projects_csv_path)

            # Create main tabs including Advanced Participation & Matching
            tabs = st.tabs([
                L["header_comments"],
                L["projects_header"],
                L["proposals_feedback_tab"],
                L["extra_visualizations_tab"],
                L["all_user_inputs_tab"],
                L["advanced_tab"]
            ])

            # --------------------------- TAB 1: Comments Analysis ---------------------------
            with tabs[0]:
                st.header("💬 Citizen Comments Analysis")
                if df_comments.empty:
                    st.warning("⚠️ No REMACTO Comments available.")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("### 📋 Original Data (first 10 rows):")
                        st.dataframe(df_comments.head(10))
                        st.write("#### 🧹 Apply Basic Arabic Normalization (Optional)")
                        do_normalize = st.checkbox("Normalize Text?")
                        df_comments_proc = df_comments.copy()
                        df_comments_proc["challenge"] = df_comments_proc["challenge"].astype(str)
                        df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].astype(str)
                        if do_normalize:
                            df_comments_proc["challenge"] = df_comments_proc["challenge"].apply(normalize_arabic)
                            df_comments_proc["proposed_solution"] = df_comments_proc["proposed_solution"].apply(normalize_arabic)
                            st.success("✅ Text normalization applied.")
                    with col2:
                        unique_axes = df_comments_proc["axis"].unique()
                        selected_axis = st.selectbox("📍 Filter by Axis:", ["All"] + list(unique_axes))
                        if selected_axis != "All":
                            filtered_comments = df_comments_proc[df_comments_proc["axis"] == selected_axis]
                        else:
                            filtered_comments = df_comments_proc
                        st.write(f"Total {len(filtered_comments)} comments after filtering by axis: **{selected_axis}**")
                    with col3:
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
                        selected_sentiment = st.selectbox("Filter by Sentiment:", ["All", "POS", "NEG", "NEU"])
                        if selected_sentiment != "All":
                            df_display = df_analysis[df_analysis["sentiment"] == selected_sentiment]
                        else:
                            df_display = df_analysis
                        num_rows = st.slider("🔢 Number of Rows to Display", 1, min(50, len(df_display)), 5)
                        st.dataframe(df_display.head(num_rows))
                        st.write("### Analysis Summary Metrics")
                        avg_polarity = df_analysis["polarity_score"].mean()
                        sentiment_summary = df_analysis["sentiment"].value_counts().to_dict()
                        st.write(f"Average Polarity Score: {avg_polarity:.2f}")
                        st.write("Sentiment Counts:", sentiment_summary)
                        csv_analysis = df_analysis.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Full Analysis Report", data=csv_analysis, file_name="full_gpt_analysis.csv", mime="text/csv")
                        from collections import Counter
                        all_topics = []
                        for topics_str in df_analysis["topics"]:
                            topics_list = [t.strip() for t in topics_str.split(";") if t.strip()]
                            all_topics.extend(topics_list)
                        topic_counts = Counter(all_topics)
                        top_topics = topic_counts.most_common(5)
                        st.write("### Top Extracted Topics")
                        st.table(pd.DataFrame(top_topics, columns=["Topic", "Count"]))
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
            # --------------------------- TAB 2: Projects ---------------------------
            with tabs[1]:
                st.header(L["projects_header"])
                if df_projects.empty:
                    st.warning(L["no_projects_msg"])
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"### {L['projects_data_preview']}")
                        st.dataframe(df_projects.head(10))
                    with col2:
                        ct_options = ["All"] + list(df_projects["CT"].dropna().unique())
                        selected_CT = st.selectbox("Filter by Region (CT):", ct_options)
                        if selected_CT != "All":
                            filtered_projects = df_projects[df_projects["CT"] == selected_CT]
                        else:
                            filtered_projects = df_projects.copy()
                        st.write(f"### Projects for Region: {selected_CT}")
                        st.dataframe(filtered_projects)
                    with col3:
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
                        if should_reprocess_projects or not os.path.exists(CACHE_FILE_PROJECTS):
                            st.warning("🧠 New project data detected. Running fresh GPT analysis for project summaries...")
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
                        st.write("### Projects by CT")
                        ct_counts = df_projects["CT"].value_counts()
                        st.bar_chart(ct_counts)
                        csv_data_projects = df_projects.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Projects CSV",
                            data=csv_data_projects,
                            file_name="REMACTO_Projects.csv",
                            mime="text/csv"
                        )
                        st.write("### 🗣️ Citizen Participation")
                        with st.expander("💬 Provide your comments or suggestions on projects"):
                            citizen_feedback = st.text_area("Your Comment/Suggestion", placeholder="Share your ideas to improve projects...")
                            if st.button("Submit Comment"):
                                if citizen_feedback.strip():
                                    store_user_input_in_csv(username, "project_feedback", citizen_feedback)
                                    st.success("Thank you! Your feedback has been recorded.")
                                else:
                                    st.warning("Please enter a comment before submitting.")
            # --------------------------- TAB 3: Proposals & Feedback ---------------------------
            with tabs[2]:
                st.header("📝 Submit a New Proposal or Feedback (Extended)")
                st.write("""
                In this section, you can choose a **Collectivité Territoriale (CT)**, 
                then a specific **commune**, and finally a **project** to give 
                more targeted feedback or a new proposal.
                """)
                col1, col2, col3 = st.columns(3)
                with col1:
                    ct_list = df_projects["CT"].dropna().unique().tolist()
                    selected_ct = st.selectbox("Select Region (CT)", ["-- Choose a Region --"] + ct_list)
                with col2:
                    if selected_ct != "-- Choose a Region --":
                        @st.cache_data(show_spinner=False)
                        def filter_by_ct(ct):
                            return df_projects[df_projects["CT"] == ct]
                        df_ct = filter_by_ct(selected_ct)
                        communes = df_ct["collectivite_territoriale"].dropna().unique().tolist()
                        selected_commune = st.selectbox("Select Commune", ["-- Choose a Commune --"] + communes)
                    else:
                        selected_commune = None
                with col3:
                    if selected_commune and selected_commune != "-- Choose a Commune --":
                        @st.cache_data(show_spinner=False)
                        def filter_by_commune(df, commune):
                            return df[df["collectivite_territoriale"] == commune]
                        df_commune = filter_by_commune(df_ct, selected_commune)
                        projects_list = df_commune["title"].dropna().unique().tolist()
                        selected_project = st.selectbox("Select Project to Provide Feedback On:", ["-- Choose a Project --"] + projects_list)
                    else:
                        selected_project = None
                if selected_ct != "-- Choose a Region --" and selected_commune and selected_project and selected_project != "-- Choose a Project --":
                    project_row = df_commune[df_commune["title"] == selected_project].iloc[0]
                    with st.expander("View Full Project Details"):
                        st.write(project_row.to_dict())
                    st.write(f"**Project Title**: {str(project_row['title'])}")
                    st.write(f"**Themes**: {str(project_row['themes'])}")
                    st.write("Feel free to comment on how to improve or any suggestions you have for this specific project.")
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
                    feedback_text = st.text_area("Any specific feedback about this project?", placeholder="Write your feedback here...")
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
                    @st.cache_data(show_spinner=False)
                    def get_project_submission_counts(project_title):
                        if os.path.exists("user_inputs.csv"):
                            df_inputs = pd.read_csv("user_inputs.csv")
                            df_proj_inputs = df_inputs[df_inputs["content"].str.contains(str(project_title), case=False, na=False)]
                            return df_proj_inputs["input_type"].value_counts().to_dict()
                        else:
                            return {}
                    submission_counts = get_project_submission_counts(selected_project)
                    st.write("### Aggregated Submissions for This Project")
                    st.write(submission_counts if submission_counts else "No proposals/feedback submitted yet.")
                    if "votes" not in st.session_state:
                        st.session_state.votes = {"up": 0, "down": 0}
                    col_v1, col_v2, col_v3 = st.columns(3)
                    with col_v1:
                        if st.button("👍 Upvote"):
                            st.session_state.votes["up"] += 1
                    with col_v2:
                        st.write("Votes:")
                        st.write(f"👍 {st.session_state.votes['up']} | 👎 {st.session_state.votes['down']}")
                    with col_v3:
                        if st.button("👎 Downvote"):
                            st.session_state.votes["down"] += 1
                    def slugify(text):
                        return str(text).lower().replace(" ", "-")
                    share_link = f"https://remacto.org/project/{slugify(selected_project)}"
                    st.write("### Share This Project")
                    st.code(share_link, language="plaintext")
                    st.info("Copy the above link to share this project with your network.")
                    @st.cache_data(show_spinner=False)
                    def sentiment_summary(related_df):
                        pos = related_df["challenge"].str.contains("جيد|ممتاز|إيجابي", case=False, na=False).sum()
                        neg = related_df["challenge"].str.contains("سيئ|سلبي|متعب", case=False, na=False).sum()
                        neu = len(related_df) - pos - neg
                        return {"Positive": pos, "Negative": neg, "Neutral": neu}
                    @st.cache_data(show_spinner=False)
                    def get_related_comments(project_theme):
                        keywords = [str(k) for k in project_theme.split()]
                        related = df_comments[df_comments["challenge"].astype(str).str.contains("|".join(keywords), case=False, na=False)]
                        return related.head(10)
                    related_comments = get_related_comments(project_row["themes"])
                    if not related_comments.empty:
                        with st.expander("View Related Citizen Comments"):
                            st.dataframe(related_comments)
                    else:
                        st.info("No related citizen comments found.")
                    sentiment_stats = sentiment_summary(related_comments)
                    st.write("### Sentiment Summary of Related Comments")
                    st.write(sentiment_stats)
                else:
                    st.info("Please select a specific project to provide feedback or propose an idea.")
            # --------------------------- TAB 4: Extra Visualizations ---------------------------
            with tabs[3]:
                st.header(L["extra_visualizations_tab"])
                if df_comments.empty:
                    st.info(L["no_comments_msg"])
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        @st.cache_data(show_spinner=False)
                        def get_axis_counts(df):
                            return df["axis"].value_counts()
                        axis_counts = get_axis_counts(df_comments)
                        st.write("### Axis Distribution (Bar Chart)")
                        st.bar_chart(axis_counts)
                    with col2:
                        @st.cache_data(show_spinner=False)
                        def get_channel_counts(df):
                            return df["channel"].value_counts()
                        channel_counts = get_channel_counts(df_comments)
                        st.write("### Channels (Pie Chart)")
                        fig_c, ax_c = plt.subplots()
                        ax_c.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%")
                        ax_c.axis("equal")
                        st.pyplot(fig_c)
                    with col3:
                        st.write(f"### Word Cloud of Proposed Solutions (in {lang})")
                        plot_wordcloud(
                            df_comments["proposed_solution"].astype(str).tolist(),
                            f"Proposed Solutions ({lang})",
                            target_language="English" if lang in ["English", "Darija"] else lang
                        )
                        @st.cache_data(show_spinner=False)
                        def compute_text_lengths(text_series):
                            return text_series.apply(lambda x: len(str(x)))
                        text_lengths = compute_text_lengths(df_comments["proposed_solution"])
                        st.write("### Text Length Distribution")
                        fig_length, ax_length = plt.subplots()
                        ax_length.hist(text_lengths, bins=20, color="skyblue")
                        ax_length.set_title("Text Length Distribution")
                        ax_length.set_xlabel("Number of Characters")
                        ax_length.set_ylabel("Count")
                        st.pyplot(fig_length)
                        st.write(f"### Word Cloud of Challenges (in {lang})")
                        plot_wordcloud(
                            df_comments["challenge"].astype(str).tolist(),
                            f"Challenges Word Cloud ({lang})",
                            target_language="English" if lang in ["English", "Darija"] else lang
                        )
            # --------------------------- TAB 6: Advanced Participation & Matching ---------------------------
            with tabs[5]:
                st.header(L["advanced_tab"])
                st.write("This tab provides advanced features to enhance citizen participation and matching.")

                # Initialize Qdrant client (assumes Qdrant is running locally on port 6333)
                qdrant_client = QdrantClient(host="localhost", port=6333)
                collection_name = "citizen_embeddings"
                
                # ---------- Qdrant Collection Setup with CSV Caching ----------
                CSV_EMBEDDINGS = "citizen_embeddings.csv"
                
                # Function to check if the Qdrant collection exists
                @st.cache_data(show_spinner=False)
                def qdrant_collection_exists(name):
                    collections_info = qdrant_client.get_collections()
                    existing = [col.name for col in collections_info.collections]
                    return name in existing

                # Create collection if not exists; if CSV cache exists, use its length as a quick check
                if not qdrant_collection_exists(collection_name):
                    st.info(f"Collection '{collection_name}' not found. Creating new collection in Qdrant.")
                    qdrant_client.recreate_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=1536, distance="Cosine")
                    )
                else:
                    st.info(f"Using existing Qdrant collection '{collection_name}'.")

                # ---------- Feature 1: Smart Matching using Qdrant and CSV Caching ----------
                colA, colB, colC = st.columns(3)
                
                with colA:
                    st.subheader("Smart Matching")
                    st.write("Match citizen suggestions (Challenges+Solutions) with project themes using Qdrant storage.")
                    match_ct = st.selectbox("Filter by CT", ["All"] + list(df_projects["CT"].dropna().unique()))
                    match_commune = st.selectbox("Filter by Commune", ["All"] + list(df_projects["collectivite_territoriale"].dropna().unique()))
                    match_theme = st.text_input("Enter Theme Keyword", "")
                    
                    # Button to update citizen embeddings in Qdrant and CSV cache
                    if st.button("Update Citizen Embeddings"):
                        @st.cache_data(show_spinner=False)
                        def compute_and_cache_citizen_embeddings(df):
                            embeddings = []
                            for index, row in df.iterrows():
                                combined_text = str(row["challenge"]) + " " + str(row["proposed_solution"])
                                # Use caching to avoid redundant API calls
                                response = openai.Embedding.create(
                                    model="text-embedding-ada-002",
                                    input=combined_text
                                )
                                embedding = response["data"][0]["embedding"]
                                embeddings.append((row["idea_id"], embedding))
                            return embeddings
                        citizen_embeddings = compute_and_cache_citizen_embeddings(df_comments)
                        # Save embeddings to CSV for cost optimization
                        df_emb = pd.DataFrame(citizen_embeddings, columns=["idea_id", "embedding"])
                        df_emb.to_csv(CSV_EMBEDDINGS, index=False)
                        # Upsert embeddings to Qdrant
                        points = []
                        for _, row in df_emb.iterrows():
                            point = PointStruct(
                                id=int(row["idea_id"]),
                                vector=row["embedding"],
                                payload={"idea_id": row["idea_id"]}
                            )
                            points.append(point)
                        qdrant_client.upsert(collection_name=collection_name, points=points)
                        st.success("✅ Citizen embeddings updated in Qdrant and cached to CSV.")
                    
                    # If embeddings CSV exists, load them
                    if os.path.exists(CSV_EMBEDDINGS):
                        df_emb = pd.read_csv(CSV_EMBEDDINGS)
                        citizen_embeddings = df_emb.apply(lambda r: (r["idea_id"], json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]), axis=1).tolist()
                    else:
                        citizen_embeddings = []
                    
                    # Now, for each project (filtered by CT, Commune, and Theme) we compute the project embedding,
                    # then use Qdrant to search for similar citizen suggestions.
                    if st.button("Run Smart Matching"):
                        project_matches = []
                        for index, row in df_projects.iterrows():
                            if (match_ct == "All" or row["CT"] == match_ct) and (match_commune == "All" or row["collectivite_territoriale"] == match_commune):
                                theme = str(row["themes"])
                                if match_theme and match_theme.lower() not in theme.lower():
                                    continue
                                # Compute project theme embedding (cache this call per project)
                                @st.cache_data(show_spinner=False)
                                def get_project_embedding(theme_text):
                                    response = openai.Embedding.create(
                                        model="text-embedding-ada-002",
                                        input=theme_text
                                    )
                                    return response["data"][0]["embedding"]
                                project_embedding = get_project_embedding(theme)
                                # Instead of looping over citizen_embeddings and computing dot products in Python,
                                # we can search in Qdrant directly using the project_embedding.
                                search_result = qdrant_client.search(
                                    collection_name=collection_name,
                                    query_vector=project_embedding,
                                    limit=5
                                )
                                for res in search_result:
                                    project_matches.append((res.id, row["title"], res.score))
                        if project_matches:
                            df_matches = pd.DataFrame(project_matches, columns=["Citizen_ID", "Project Title", "Match Score"])
                            st.dataframe(df_matches.sort_values(by="Match Score", ascending=False))
                        else:
                            st.info("No matching projects found.")
                
                # ---------- Feature 2: Suggest Projects Based on Citizen Concerns ----------
                with colB:
                    st.subheader("Project Suggestions")
                    citizen_input = st.text_area("Describe your concern:")
                    if st.button("Suggest Projects"):
                        response = openai.Embedding.create(
                            model="text-embedding-ada-002",
                            input=str(citizen_input)
                        )
                        citizen_emb = response["data"][0]["embedding"]
                        suggestions = []
                        for index, row in df_projects.iterrows():
                            response = openai.Embedding.create(
                                model="text-embedding-ada-002",
                                input=str(row["themes"])
                            )
                            proj_emb = response["data"][0]["embedding"]
                            score = np.dot(citizen_emb, proj_emb)
                            suggestions.append((row["title"], score))
                        if suggestions:
                            df_suggestions = pd.DataFrame(suggestions, columns=["Project Title", "Confidence Score"])
                            st.write("Top Suggested Projects:")
                            st.dataframe(df_suggestions.sort_values(by="Confidence Score", ascending=False).head(5))
                            if st.button("Feedback: Does this address your issue?"):
                                st.success("Thank you for your feedback!")
                        else:
                            st.info("No projects suggested.")
                
                # ---------- Feature 3: Highlight Participation Gaps ----------
                with colC:
                    st.subheader("Participation Gaps")
                    st.write("Identifying topics in citizen suggestions not linked to any project.")
                    all_topics = []
                    for text in df_comments["challenge"]:
                        topics = gpt_extract_topics(str(text))
                        all_topics.extend(topics)
                    from collections import Counter
                    topic_counts = Counter(all_topics)
                    project_themes = []
                    for text in df_projects["themes"]:
                        project_themes.extend(gpt_extract_topics(str(text)))
                    project_topic_counts = Counter(project_themes)
                    gap_topics = {topic: count for topic, count in topic_counts.items() if topic not in project_topic_counts}
                    if gap_topics:
                        st.write("Orphan Topics:")
                        st.table(pd.DataFrame(gap_topics.items(), columns=["Topic", "Count"]))
                    else:
                        st.info("No participation gaps detected.")
                
                st.write("---")
                # ---------- Feature 4: Crowd-vote on Proposed Solutions ----------
                st.subheader("Crowd-Voting")
                if "crowd_votes" not in st.session_state:
                    st.session_state.crowd_votes = {}
                vote_project = st.selectbox("Select a Proposal for Voting", df_projects["title"].dropna().unique())
                if st.button("Vote Up"):
                    st.session_state.crowd_votes[vote_project] = st.session_state.crowd_votes.get(vote_project, 0) + 1
                if st.button("Vote Down"):
                    st.session_state.crowd_votes[vote_project] = st.session_state.crowd_votes.get(vote_project, 0) - 1
                st.write("Current Votes:")
                st.write(st.session_state.crowd_votes)
                
                # ---------- Feature 5: Participation KPIs per Commune/CT ----------
                st.subheader("Participation KPIs")
                kpi_df = pd.DataFrame({
                    "Commune": df_comments["channel"],
                    "Challenges": np.random.randint(5, 50, len(df_comments)),
                    "Solutions": np.random.randint(3, 40, len(df_comments)),
                    "Aligned Projects": np.random.randint(1, 10, len(df_comments)),
                    "Average Polarity": df_analysis["polarity_score"].mean()
                })
                st.dataframe(kpi_df.head(10))
                st.line_chart(kpi_df["Challenges"])
                
                # ---------- Feature 6: Citizen-Informed Project Enrichment ----------
                st.subheader("Project Enrichment")
                enrich_proj = st.selectbox("Select a Project for Enrichment", df_projects["title"].dropna().unique())
                st.write("Top 5 related citizen suggestions:")
                enrichment = []
                for index, row in df_comments.iterrows():
                    response = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=str(row["challenge"])
                    )
                    emb = response["data"][0]["embedding"]
                    similarity = np.random.rand()  # Dummy similarity for demo purposes
                    enrichment.append((row["idea_id"], similarity))
                enrich_df = pd.DataFrame(enrichment, columns=["Citizen_ID", "Similarity"])
                st.dataframe(enrich_df.sort_values(by="Similarity", ascending=False).head(5))
                
                # ---------- Feature 7: Alert System for New Matching Projects ----------
                st.subheader("Alert System")
                alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.80, 0.05)
                new_project_match = np.random.rand()  # Dummy match score for demo
                if new_project_match >= alert_threshold:
                    st.success(f"Alert: A new project has a match score of {new_project_match:.2f} with your issue!")
                else:
                    st.info("No new project matches exceeding the threshold.")
                
                # ---------- Feature 8: Thematic Sentiment Trends ----------
                st.subheader("Thematic Sentiment Trends")
                trend_df = pd.DataFrame({
                    "Theme": df_comments["axis"],
                    "Sentiment": df_analysis["polarity_score"],
                    "Date": pd.date_range(start="2022-01-01", periods=len(df_comments), freq="D")
                })
                trend_df = trend_df.groupby("Theme").agg({"Sentiment": "mean"}).reset_index()
                st.bar_chart(trend_df.set_index("Theme"))
                
                # ---------- Feature 9: Participatory Budgeting Suggestions ----------
                st.subheader("Budgeting Suggestions")
                budget_keywords = ["تهيئة", "إحداث", "ميزانية"]
                budget_suggestions = df_comments[df_comments["challenge"].astype(str).str.contains("|".join(budget_keywords), na=False)]
                st.write("Suggestions with budget-related keywords:")
                st.dataframe(budget_suggestions.head(5))
                
                # ---------- Feature 10: Impact Simulation (Bonus) ----------
                st.subheader("Impact Simulation")
                impact_text = st.text_area("Describe your proposal in detail to simulate its impact:")
                if st.button("Simulate Impact"):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You simulate the impact of a proposal on social, environmental, and economic metrics."},
                            {"role": "user", "content": impact_text}
                        ],
                        max_tokens=150,
                        temperature=0.5
                    )
                    impact_result = response["choices"][0]["message"]["content"].strip()
                    st.write("### Simulated Impact:")
                    st.write(impact_result)

        else:
            st.warning(L["token_invalid"])
    else:
        st.info(L["no_token_msg"])

if __name__ == "__main__":
    main()

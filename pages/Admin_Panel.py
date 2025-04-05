import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import psutil
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
from bson import ObjectId
from CivicCatalyst import create_user
import openai
from CivicCatalyst import delete_user
import hashlib
# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]
# --------------------------
# Load Font Awesome and Custom CSS
# --------------------------
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
    crossorigin="anonymous" referrerpolicy="no-referrer" media="all" />
    <style>
      /* Global Styles */
      body {
          background: linear-gradient(135deg, #e0eafc, #cfdef3);
          font-family: 'Poppins', sans-serif;
          margin: 0;
          padding: 0;
      }
      /* Dashboard Header */
      .dashboard-header {
          color: #2B3E50;
          font-size: 2.5rem;
          font-weight: 700;
          text-align: center;
          margin-bottom: 2rem;
          text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
      }
      /* Section Headers */
      .section-header {
          margin-top: 2rem;
          margin-bottom: 1rem;
          color: #2B3E50;
          font-size: 1.8rem;
          border-bottom: 2px solid #00b09b;
          padding-bottom: 0.5rem;
      }
      /* Icons Styling */
      .icon {
          margin-right: 0.5rem;
          color: #00b09b;
      }
      /* Export Button Styling */
      .export-btn {
          margin-top: 0.5rem;
      }
      /* Tab Header Styling */
      .tab-header {
          font-family: 'Poppins', sans-serif;
          font-weight: 600;
          font-size: 1.2rem;
      }
      /* Custom Data Card Style */
      .data-card {
          background-color: #ffffff;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
          padding: 1.5rem;
          margin-bottom: 1.5rem;
      }
      /* Custom Button Style */
      button {
          border-radius: 5px;
          padding: 0.5rem 1rem;
          font-size: 1rem;
          font-weight: bold;
          background: linear-gradient(90deg, #00b09b, #96c93d);
          color: #ffffff;
          border: none;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
      }
      button:hover {
          transform: translateY(-3px);
          box-shadow: 0 6px 12px rgba(0,0,0,0.2);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------
# Translation Dictionary (minimal sample; extend as needed)
# ----------------------------------------------------------------
translations = {
    "admin_dashboard": {
        "en": "Admin Dashboard",
        "fr": "Tableau de bord administrateur",
        "ar": "لوحة تحكم المسؤول",
        "darija": "لوحة تحكم المسؤول"
    },
    "overview": {
        "en": "Overview & Analytics",
        "fr": "Vue d'ensemble & Analytique",
        "ar": "نظرة عامة وتحليلات",
        "darija": "نظرة عامة وتحليلات"
    },
    "user_management": {
        "en": "User Management",
        "fr": "Gestion des utilisateurs",
        "ar": "إدارة المستخدمين",
        "darija": "تسيير المستخدمين"
    },
    "content_moderation": {
        "en": "Content Moderation",
        "fr": "Modération du contenu",
        "ar": "مراجعة المحتوى",
        "darija": "مراقبة المحتوى"
    },
    "project_management": {
        "en": "Project Management",
        "fr": "Gestion des projets",
        "ar": "إدارة المشاريع",
        "darija": "تسيير المشاريع"
    },
    "news_management": {
        "en": "News Management",
        "fr": "Gestion des actualités",
        "ar": "إدارة الأخبار",
        "darija": "تسيير الأخبار"
    },
    "ai_insights": {
        "en": "AI & NLP Insights",
        "fr": "Aperçus IA & NLP",
        "ar": "رؤى الذكاء الاصطناعي وNLP",
        "darija": "تحليلات الذكاء الاصطناعي"
    },
    "system_health": {
        "en": "System Health & Settings",
        "fr": "Santé du système & Paramètres",
        "ar": "صحة النظام والإعدادات",
        "darija": "صحة النظام والإعدادات"
    },
    "audit_logs": {
        "en": "Audit Logs",
        "fr": "Journaux d'audit",
        "ar": "سجلات التدقيق",
        "darija": "سجلات التدقيق"
    },
    "qdrant_metrics": {
        "en": "Qdrant Metrics",
        "fr": "Métriques Qdrant",
        "ar": "مؤشرات Qdrant",
        "darija": "مؤشرات Qdrant"
    },
    "semantic_search": {
        "en": "Semantic Search",
        "fr": "Recherche sémantique",
        "ar": "البحث الدلالي",
        "darija": "البحث الدلالي"
    },
    "total_users": {
        "en": "Total Users",
        "fr": "Nombre total d'utilisateurs",
        "ar": "إجمالي المستخدمين",
        "darija": "إجمالي المستخدمين"
    },
    "total_comments": {
        "en": "Total Citizen Comments",
        "fr": "Nombre total de commentaires citoyens",
        "ar": "إجمالي تعليقات المواطنين",
        "darija": "إجمالي تعليقات المواطنين"
    },
    "total_projects": {
        "en": "Total Municipal Projects",
        "fr": "Nombre total de projets municipaux",
        "ar": "إجمالي المشاريع البلدية",
        "darija": "إجمالي المشاريع البلدية"
    },
    "select_user": {
        "en": "Select a User",
        "fr": "Sélectionner un utilisateur",
        "ar": "اختر مستخدمًا",
        "darija": "اختار مستعمل"
    },
    "new_role": {
        "en": "New Role",
        "fr": "Nouveau rôle",
        "ar": "دور جديد",
        "darija": "دور جديد"
    },
    "update_role": {
        "en": "Update Role",
        "fr": "Mettre à jour le rôle",
        "ar": "تحديث الدور",
        "darija": "تحديث الدور"
    }
}

def t(key):
    """Return the translated string based on session language."""
    lang = st.session_state.get("site_language", "en")
    return translations.get(key, {}).get(lang, key)

# ----------------------------------------------------------------
# MongoDB (Users & Audit Logs)
# ----------------------------------------------------------------
def get_mongo_client():
    """Return a MongoDB client."""
    return MongoClient("mongodb://localhost:27017")

def load_users():
    """Load all users from the MongoDB 'users' collection."""
    client = get_mongo_client()
    db = client["CivicCatalyst"]
    users_collection = db["users"]
    users = list(users_collection.find({}, {"_id": 0, "username": 1, "role": 1}))
    client.close()
    return users


def compute_text_hash(text: str) -> str:
    """Compute an MD5 hash of the provided text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_stored_summary(region: str) -> dict:
    """
    Retrieve the stored summary document for a given region from the 'comment_summaries' collection.
    Returns a dictionary with keys: 'summary', 'text_hash', and 'last_updated'
    or None if not found.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        summary_doc = db["comment_summaries"].find_one({"region": region})
        return summary_doc
    except Exception as e:
        st.error(f"Error retrieving stored summary for {region}: {e}")
        return None
    finally:
        client.close()

def store_summary(region: str, summary: str, text_hash: str):
    """
    Store or update the summary for a given region in the 'comment_summaries' collection.
    Also stores the computed hash of the combined text and the current timestamp.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        summary_doc = {
            "region": region,
            "summary": summary,
            "text_hash": text_hash,
            "last_updated": datetime.utcnow()
        }
        # Upsert: update if exists, insert otherwise.
        db["comment_summaries"].update_one({"region": region}, {"$set": summary_doc}, upsert=True)
    except Exception as e:
        st.error(f"Error storing summary for {region}: {e}")
    finally:
        client.close()


def update_user_role(username, new_role):
    """Update the role of a user in the MongoDB 'users' collection."""
    client = get_mongo_client()
    db = client["CivicCatalyst"]
    users_collection = db["users"]
    result = users_collection.update_one({"username": username}, {"$set": {"role": new_role}})
    client.close()
    return result.modified_count > 0

def load_audit_logs():
    """Load audit logs from 'AuditLogs' in Mongo. If none found, return some sample logs."""
    client = get_mongo_client()
    db = client["CivicCatalyst"]
    logs = list(db["AuditLogs"].find({}))
    client.close()
    if not logs:
        logs = [
            {"timestamp": str(datetime.utcnow()), "action": "User admin updated role for user 'john' to moderator."},
            {"timestamp": str(datetime.utcnow()), "action": "Flagged comment approved."}
        ]
    return logs

def get_recent_login_history(limit=5):
    """
    Retrieve the most recent login events from the 'login_history' collection.
    Returns a list of dictionaries with 'username' and 'timestamp' fields.
    """
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        login_history = db["login_history"]
        # Get the most recent login events sorted by timestamp descending
        history = list(login_history.find({}).sort("timestamp", -1).limit(limit))
        for rec in history:
            # Format the timestamp for display
            if "timestamp" in rec and isinstance(rec["timestamp"], datetime):
                rec["timestamp"] = rec["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            # Remove _id field for a cleaner table
            rec.pop("_id", None)
        return history
    except Exception as e:
        st.error("Error retrieving login history: " + str(e))
        return []
    finally:
        client.close()
# ----------------------------------------------------------------
# Qdrant Helper Functions
# ----------------------------------------------------------------
def get_qdrant_client():
    """Return a Qdrant client (adjust host/port if necessary)."""
    return QdrantClient(host="localhost", port=6333)

def load_qdrant_documents(collection_name: str, vector_dim: int):
    """
    Retrieve all documents (payloads) from a Qdrant collection using scroll.
    Use 'vector_dim' to match the dimension your collection was created with.
    """
    client = get_qdrant_client()

    all_docs = []
    points, next_page_offset = client.scroll(collection_name=collection_name, limit=100)
    all_docs.extend([pt.payload for pt in points])

    while next_page_offset is not None:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            offset=next_page_offset,
            limit=100
        )
        all_docs.extend([pt.payload for pt in points])

    return all_docs

def perform_semantic_search(collection_name: str, query_vector: list, top: int = 5):
    """Perform semantic search on a Qdrant collection using the given query vector."""
    client = get_qdrant_client()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top,
        with_payload=True
    )
    return results

def load_qdrant_collection_info(collection_name: str):
    """Fetch and return info about a Qdrant collection."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name=collection_name)
    return info

def get_user_comment_stats():
    """
    Retrieve all comments from the 'citizen_comments' Qdrant collection (dimension=384)
    and compute:
      - total_comments: total number of comments per citizen
      - positive_comments: number of comments with sentiment "POS" per citizen
    Returns a dictionary mapping citizen_name to their statistics.
    """
    comments = load_qdrant_documents("citizen_comments", vector_dim=384)
    stats = {}
    for comment in comments:
        # Retrieve the citizen's name from the comment payload
        user = comment.get("citizen_name")
        if not user:
            continue
        if user not in stats:
            stats[user] = {"total_comments": 0, "positive_comments": 0}
        stats[user]["total_comments"] += 1
        # Assume sentiment is stored as "POS" for positive
        sentiment = comment.get("sentiment", "").upper()
        if sentiment == "POS":
            stats[user]["positive_comments"] += 1
    return stats

# ----------------------------------------------------------------
# Helper Function: Get Citizen Statistics from Qdrant
# ----------------------------------------------------------------
def get_citizen_stats(date_range=None, channel_filter=None):
    """
    Retrieve citizen comment statistics from the 'citizen_comments' Qdrant collection (dim=384)
    with optional filtering by date range and/or channel.
    
    Each comment is expected to have:
      - citizen_name
      - comment_text
      - date_submitted in "YYYY-MM-DD" format
      - channel (e.g., "اللقاء", "SNS", etc.)
      - sentiment (e.g., "POS", "NEG", etc.)
      - polarity (float)
      - votes: a dict with "vote_score" (int)
    
    Returns a dictionary mapping citizen_name to statistics:
      {citizen_name: {
         "total_comments": int,
         "positive_comments": int,
         "negative_comments": int,
         "neutral_comments": int,
         "average_polarity": float,
         "average_vote_score": float
      }}
    """
    comments = load_qdrant_documents("citizen_comments", vector_dim=384)
    stats = {}
    for comment in comments:
        # Optional: filter by date range if provided
        if date_range:
            date_str = comment.get("date_submitted", "")
            try:
                comment_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue  # Skip if date parsing fails
            if comment_date < date_range[0] or comment_date > date_range[1]:
                continue
        
        # Optional: filter by channel if provided and not "All"
        if channel_filter and channel_filter != "All":
            if comment.get("channel", "").strip() != channel_filter:
                continue

        citizen = comment.get("citizen_name")
        if not citizen:
            continue
        if citizen not in stats:
            stats[citizen] = {
                "total_comments": 0,
                "positive_comments": 0,
                "negative_comments": 0,
                "neutral_comments": 0,
                "total_polarity": 0.0,
                "total_vote_score": 0,
                "vote_count": 0
            }
        stats[citizen]["total_comments"] += 1
        try:
            polarity = float(comment.get("polarity", 0.0))
        except Exception:
            polarity = 0.0
        stats[citizen]["total_polarity"] += polarity
        
        sentiment = comment.get("sentiment", "").upper()
        if sentiment == "POS":
            stats[citizen]["positive_comments"] += 1
        elif sentiment == "NEG":
            stats[citizen]["negative_comments"] += 1
        else:
            stats[citizen]["neutral_comments"] += 1

        # Process vote data if available
        votes = comment.get("votes", {})
        vote_score = votes.get("vote_score", 0)
        stats[citizen]["total_vote_score"] += vote_score
        stats[citizen]["vote_count"] += 1

    # Compute averages
    for citizen, data in stats.items():
        if data["total_comments"] > 0:
            data["average_polarity"] = data["total_polarity"] / data["total_comments"]
        else:
            data["average_polarity"] = 0.0
        if data["vote_count"] > 0:
            data["average_vote_score"] = data["total_vote_score"] / data["vote_count"]
        else:
            data["average_vote_score"] = 0.0
        # Remove intermediate accumulators
        del data["total_polarity"]
        del data["total_vote_score"]
        del data["vote_count"]
    return stats


# ----------------------------------------------------------------
# Comments, Projects, "News" in Qdrant
# ----------------------------------------------------------------
def load_flagged_comments_qdrant():
    """
    Load flagged comments from 'citizen_comments' (dimension=384) by checking if 'status' == 'flagged'.
    """
    all_docs = load_qdrant_documents("citizen_comments", vector_dim=384)
    flagged = [doc for doc in all_docs if doc.get("status", "").lower() == "flagged"]
    return flagged

def load_projects_qdrant():
    """
    Load projects from 'municipal_projects' (dimension=384) in Qdrant.
    """
    return load_qdrant_documents("municipal_projects", vector_dim=384)

def add_project_qdrant(project_data: dict):
    """
    Upsert a new project into 'municipal_projects'. 
    We generate a random 384-dim vector for demonstration.
    """
    client = get_qdrant_client()
    embedding_dim = 384
    random_vector = np.random.rand(embedding_dim).tolist()

    import random
    doc_id = random.randint(1_000_000, 9_999_999)

    point = PointStruct(
        id=doc_id,
        vector=random_vector,
        payload=project_data
    )
    client.upsert(collection_name="municipal_projects", points=[point])
    return doc_id

def update_project_qdrant(project_id: str, updated_data: dict) -> bool:
    """
    Update an existing project in the 'municipal_projects' Qdrant collection by upserting.
    For demonstration, we generate a new random 384-dim vector.
    In production, you may want to recalculate the vector only if necessary.
    """
    try:
        client = get_qdrant_client()
        embedding_dim = 384
        # For simplicity, we generate a new vector (in production, re-use or recalc based on text)
        random_vector = np.random.rand(embedding_dim).tolist()
        point = PointStruct(
            id=project_id,
            vector=random_vector,
            payload=updated_data
        )
        client.upsert(collection_name="municipal_projects", points=[point])
        return True
    except Exception as e:
        st.error(f"Error updating project: {e}")
        return False
    finally:
        client.close()

def delete_project_qdrant(project_id: str) -> bool:
    """
    Delete a project from the 'municipal_projects' Qdrant collection by project_id.
    """
    try:
        client = get_qdrant_client()
        result = client.delete(collection_name="municipal_projects", point_id=project_id)
        # The result may vary; adjust based on the Qdrant client version.
        # For example, check if result.status is "ok" or result.deleted_count > 0.
        return result.status == "ok"
    except Exception as e:
        st.error(f"Error deleting project: {e}")
        return False
    finally:
        client.close()


def load_hespress_news():
    """
    Load "news" from 'hespress_politics_details' in Qdrant (dimension=1536).
    You can rename or adapt as needed, but dimension=1536 is indicated by your data.
    """
    return load_qdrant_documents("hespress_politics_details", vector_dim=1536)

def add_hespress_news(article_data: dict):
    """
    Add a new article to 'hespress_politics_details' in Qdrant.
    We'll generate a random 1536-dim vector for demonstration.
    """
    client = get_qdrant_client()
    embedding_dim = 1536
    random_vector = np.random.rand(embedding_dim).tolist()

    import random
    doc_id = random.randint(1_000_000, 9_999_999)

    point = PointStruct(
        id=doc_id,
        vector=random_vector,
        payload=article_data
    )
    client.upsert(collection_name="hespress_politics_details", points=[point])
    return doc_id

# ----------------------------------------------------------------
# System Metrics (Combining Mongo + Qdrant)
# ----------------------------------------------------------------
def load_system_metrics():
    """
    Return system metrics: 
    - total_users (MongoDB 'users') 
    - total_comments (from 'citizen_comments' dimension=384)
    - total_projects (from 'municipal_projects' dimension=384)
    """
    # total users from Mongo
    client = get_mongo_client()
    db = client["CivicCatalyst"]
    total_users = db["users"].count_documents({})
    client.close()

    # total_comments from Qdrant
    total_comments = len(load_qdrant_documents("citizen_comments", vector_dim=384))

    # total_projects from Qdrant
    total_projects = len(load_qdrant_documents("municipal_projects", vector_dim=384))

    return total_users, total_comments, total_projects

# ----------------------------------------------------------------
# Word Cloud
# ----------------------------------------------------------------
def generate_word_cloud(keywords: list):
    """Generate a word cloud image from a list of keywords."""
    text = " ".join(keywords)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    temp_file = "wordcloud.png"
    plt.savefig(temp_file, bbox_inches="tight")
    plt.close()
    return temp_file

# ----------------------------------------------------------------
# System Health (psutil)
# ----------------------------------------------------------------
def load_system_health():
    """Return basic system health metrics using psutil."""
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    return cpu_percent, mem_percent

# ----------------------------------------------------------------
# ADMIN DASHBOARD PAGE
# ----------------------------------------------------------------
def dashboard_admin():
    """
    Displays the admin dashboard page with extended features:
      - Overview & Analytics (Mongo + Qdrant)
      - User Management (Mongo)
      - Content Moderation (Flagged from 'citizen_comments')
      - Project Management ('municipal_projects')
      - News Management ('hespress_politics_details')
      - AI & NLP Insights
      - System Health & Settings
      - Audit Logs (Mongo)
      - Qdrant Metrics (includes all known collections)
      - Semantic Search
    Accessible only to admin users.
    """
    if st.session_state.get("role") != "admin":
        st.error("Access Denied. Admins only.")
        return

    st.markdown(
        f"<h1 class='dashboard-header'><i class='fas fa-user-cog icon'></i>{t('CivicCatalyst Admin Dashboard')}</h1>",
        unsafe_allow_html=True
    )
    st.write(
        "Welcome to the Admin Dashboard. Here you can manage users, "
        "review flagged content, monitor system health, perform semantic searches, "
        "manage projects and news articles, and view audit logs and Qdrant metrics."
    )

    tabs = st.tabs([
        f"📊 {t('Overview')}",
        f"👥 {t('User Management')}",
        f"⚠️ {t('Flagged Content')}",
        f"🗂️ {t('Project Management')}",
        f"📰 {t('News Management')}",
        f"🤖 {t('AI Insights')}",
        f"🖥️ {t('System Health')}",
        f"📋 {t('Audit Logs')}",
        f"💾 {t('Qdrant Metrics')}",
        f"🔍 {t('Semantic Search')}"
    ])


    # --- Tab 1: Overview & Analytics ---
    with tabs[0]:
        st.markdown("<h2 class='section-header'><i class='fas fa-chart-line icon'></i>Overview & Analytics</h2>", unsafe_allow_html=True)
        
        # System-wide metrics (users from MongoDB, comments & projects from Qdrant)
        total_users, total_comments, total_projects = load_system_metrics()
        col1, col2, col3 = st.columns(3)
        col1.metric(t("total_users"), total_users)
        col2.metric(t("total_comments"), total_comments)
        col3.metric(t("total_projects"), total_projects)
        
        st.markdown("---")
        
        # Simulate User Activity Trend data
        st.write("### Daily User Activity Trend (Simulated)")
        df_activity = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=50, freq="D"),
            "new_users": np.random.randint(0, 50, 50)
        })
        st.plotly_chart(px.line(df_activity, x="date", y="new_users", title="Daily New Users"), use_container_width=True)
        
        # Compute weekly aggregation and display as a bar chart
        df_activity["week"] = df_activity["date"].dt.strftime("%Y-%W")
        df_weekly = df_activity.groupby("week")["new_users"].sum().reset_index()
        st.plotly_chart(px.bar(df_weekly, x="week", y="new_users", title="Weekly New Users"), use_container_width=True)
        
        st.markdown("---")
        
        # Simulate Citizen Sentiment data
        st.write("### Citizen Sentiment Trend (Simulated)")
        df_sentiment = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=50, freq="D"),
            "positive": np.random.randint(10, 100, 50),
            "negative": np.random.randint(5, 50, 50),
            "neutral": np.random.randint(20, 80, 50)
        })
        st.plotly_chart(px.area(df_sentiment, x="date", y=["positive", "negative", "neutral"], title="Daily Sentiment Trend"), use_container_width=True)
        
        # Display average sentiment metrics
        avg_positive = df_sentiment["positive"].mean()
        avg_negative = df_sentiment["negative"].mean()
        avg_neutral = df_sentiment["neutral"].mean()
        col4, col5, col6 = st.columns(3)
        col4.metric("Avg Positive", f"{avg_positive:.1f}")
        col5.metric("Avg Negative", f"{avg_negative:.1f}")
        col6.metric("Avg Neutral", f"{avg_neutral:.1f}")
        
        st.markdown("---")
        
        # Pie chart for overall sentiment distribution
        total_positive = df_sentiment["positive"].sum()
        total_negative = df_sentiment["negative"].sum()
        total_neutral = df_sentiment["neutral"].sum()
        df_pie = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [total_positive, total_negative, total_neutral]
        })
        st.plotly_chart(px.pie(df_pie, names="Sentiment", values="Count", title="Overall Sentiment Distribution"), use_container_width=True)
        
        st.markdown("---")
        
        # Word Cloud for Top Keywords (simulated sample keywords)
        st.write("### Top Keywords")
        sample_keywords = ["transparency", "participation", "innovation", "development", "citizenship", "open government"]
        wc_path = generate_word_cloud(sample_keywords)
        st.image(wc_path, caption="Word Cloud of Top Keywords", use_column_width=True)
        
        st.markdown("---")
        
        st.write("### Recent Login Activity")
        recent_logins = get_recent_login_history(limit=5)
        if recent_logins:
            df_recent = pd.DataFrame(recent_logins)
            st.table(df_recent)
        else:
            st.info("No login activity found.")
        
    # --- Tab 2: User Management (Mongo) ---

    with tabs[1]:
        st.markdown(
            f"<h2 class='section-header'><i class='fas fa-users icon'></i>{t('User Management')}</h2>",
            unsafe_allow_html=True
        )
        
        # 1) Provide a filter/search box for username or role
        search_query = st.text_input("Search users (by username or role):")
        
        # Load all users from Mongo
        users = load_users()  # returns a list of dicts with keys: username, role

        # Retrieve comment statistics from Qdrant (per citizen)
        user_stats = get_user_comment_stats()
        
        # Merge statistics into the user dictionary
        for user in users:
            username = user.get("username")
            stats = user_stats.get(username, {"total_comments": 0, "positive_comments": 0})
            user["total_comments"] = stats["total_comments"]
            user["positive_comments"] = stats["positive_comments"]

        # If we have users, apply a simple filter on the list
        if users:
            if search_query.strip():
                search_lower = search_query.lower()
                users = [
                    u for u in users 
                    if search_lower in u["username"].lower() 
                    or search_lower in u["role"].lower()
                ]
            # 2) Display the filtered user list in a dataframe
            st.dataframe(users)
            
            # 3) Export the currently *filtered* list as CSV
            if st.button("Export Filtered Users as CSV", key="export_users"):
                df_users = pd.DataFrame(users)
                st.download_button(
                    "Download CSV",
                    df_users.to_csv(index=False),
                    "filtered_users.csv",
                    "text/csv"
                )
            
            # 4) Provide a dropdown to select a user for role updates
            usernames = [user["username"] for user in users]
            if usernames:
                selected_user = st.selectbox(t("select_user"), usernames)
                new_role = st.selectbox(t("new_role"), ["citizen", "moderator", "admin"])
                if st.button(t("update_role")):
                    if update_user_role(selected_user, new_role):
                        st.success(f"Role for {selected_user} updated to {new_role}.")
                        st.rerun()
                    else:
                        st.error("Failed to update role. Please try again.")
            
            # 5) Deletion of users, if current user is admin
            if st.session_state.get("role") == "admin":
                st.subheader("Delete a User")
                user_to_delete = st.selectbox("Select a User to Delete", [""] + usernames)
                if user_to_delete:
                    if st.button("Confirm Delete"):
                        if delete_user(user_to_delete):
                            st.success(f"User '{user_to_delete}' deleted successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to delete user or user not found.")
        
        else:
            st.info("No users found.")
        
        st.markdown("---")
        
        # 6) Add New User Form (Only admins can create new users)
        if st.session_state.get("role") == "admin":
            st.subheader("Add a New User")
            with st.form("create_user_form", clear_on_submit=True):
                new_username = st.text_input("New Username")
                new_user_role = st.selectbox("Role", ["citizen", "moderator", "admin"])
                new_password = st.text_input("Temporary Password (plaintext)", type="password")
                create_user_btn = st.form_submit_button("Create User")
            if create_user_btn:
                if not new_username.strip():
                    st.error("Username cannot be empty.")
                else:
                    success = create_user(new_username, new_password, new_user_role)
                    if success:
                        st.success(f"User '{new_username}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create user (maybe user already exists?).")
            st.markdown("---")
        # -------------------------
        # CITIZEN MANAGEMENT & STATISTICS (from Qdrant)
        # -------------------------
        st.subheader("Citizen Management & Statistics")

        # --- Filtering Options ---
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            # Date range filter: default from Jan 1, 2022 to today
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1).date())
        with col_filter2:
            end_date = st.date_input("End Date", value=datetime.utcnow().date())

        # Channel filter: You could also extract channels from your data dynamically;
        # here we use a static list as an example.
        channel_options = ["All", "اللقاء", "SNS"]
        selected_channel = st.selectbox("Filter by Channel", channel_options)

        # Retrieve citizen statistics with filters applied
        citizen_stats = get_citizen_stats(date_range=(start_date, end_date), channel_filter=selected_channel)
        if citizen_stats:
            df_citizens = pd.DataFrame.from_dict(citizen_stats, orient="index").reset_index().rename(columns={"index": "citizen_name"})
            st.dataframe(df_citizens)
            
            # --- Visualization: Total Comments per Citizen ---
            st.write("#### Total Comments per Citizen")
            st.plotly_chart(px.bar(df_citizens, x="citizen_name", y="total_comments", title="Total Comments per Citizen"), use_container_width=True)
            
            # --- Visualization: Sentiment Distribution per Citizen ---
            st.write("#### Sentiment Distribution per Citizen")
            df_stack = df_citizens.melt(id_vars="citizen_name", value_vars=["positive_comments", "negative_comments", "neutral_comments"], 
                                        var_name="sentiment", value_name="count")
            st.plotly_chart(px.bar(df_stack, x="citizen_name", y="count", color="sentiment", 
                                    title="Sentiment Distribution per Citizen"), use_container_width=True)
            
            # --- Visualization: Total Comments vs. Average Polarity ---
            st.write("#### Total Comments vs. Average Polarity")
            st.plotly_chart(px.scatter(df_citizens, x="total_comments", y="average_polarity", hover_data=["citizen_name"],
                                        title="Total Comments vs. Average Polarity"), use_container_width=True)
            
            # --- Visualization: Average Vote Score per Citizen ---
            st.write("#### Average Vote Score per Citizen")
            st.plotly_chart(px.bar(df_citizens, x="citizen_name", y="average_vote_score", title="Average Vote Score per Citizen"), use_container_width=True)
            
            # --- Aggregated Visualization: Overall Sentiment Distribution ---
            overall_sentiments = df_citizens[["positive_comments", "negative_comments", "neutral_comments"]].sum().reset_index()
            overall_sentiments.columns = ["sentiment", "count"]
            st.write("#### Overall Sentiment Distribution")
            st.plotly_chart(px.pie(overall_sentiments, names="sentiment", values="count", title="Overall Sentiment Distribution"), use_container_width=True)
            
        else:
            st.info("No citizen comment data found.")

    # --- Tab 3: Content Moderation (Qdrant: 'citizen_comments') ---
    with tabs[2]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-exclamation-triangle icon'></i>{t('Content Moderation')}</h2>", unsafe_allow_html=True)
        flagged_comments = load_flagged_comments_qdrant()
        if flagged_comments:
            for i, comment in enumerate(flagged_comments):
                summary = comment.get("content", "N/A")
                summary_label = summary[:50] + "..." if len(summary) > 50 else summary

                with st.expander(f"Flagged Comment #{i+1} - {summary_label}", expanded=False):
                    st.markdown(f"**Full Content:** {comment.get('content', 'N/A')}")
                    st.markdown(f"**Challenge:** {comment.get('challenge', 'N/A')}")
                    st.markdown(f"**Proposed Solution:** {comment.get('solution', 'N/A')}")
                    st.markdown(f"**Current Status:** {comment.get('status', 'N/A')}")

                    col_a, col_b = st.columns(2)
                    if col_a.button("Approve", key=f"approve_{i}"):
                        # For production, do an actual Qdrant update. Here we simulate:
                        comment["status"] = "approved"
                        st.success("Comment approved (simulated).")
                        st.rerun()
                    if col_b.button("Reject", key=f"reject_{i}"):
                        comment["status"] = "rejected"
                        st.success("Comment rejected (simulated).")
                        st.rerun()
        else:
            st.info("No flagged content found.")

   # --- Tab 4: Project Management (Qdrant: 'municipal_projects') ---
    with tabs[3]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-clipboard icon'></i>{t('Project Management')}</h2>", unsafe_allow_html=True)
        
        # Retrieve projects from Qdrant
        projects = load_projects_qdrant()
        if projects:
            df_projects = pd.DataFrame(projects)
            st.dataframe(df_projects)
            if st.button("Export Projects as CSV", key="export_projects"):
                st.download_button("Download CSV", df_projects.to_csv(index=False), "projects.csv", "text/csv")
        else:
            st.info("No projects found in Qdrant.")
        
        st.markdown("---")
        
        # ---- Create New Project ----
        st.markdown("#### Add New Project (Qdrant)")
        with st.form("project_form", clear_on_submit=True):
            proj_title = st.text_input("Project Title")
            proj_desc = st.text_area("Project Description")
            proj_theme = st.text_input("Theme")
            proj_CT = st.text_input("CT / City")
            proj_province = st.text_input("Province")
            proj_budget = st.number_input("Budget", min_value=0, step=1000)
            proj_status = st.selectbox("Status", ["Planned", "In Progress", "Completed"])
            submit_proj = st.form_submit_button("Add Project")
        if submit_proj:
            new_proj = {
                "title": proj_title,
                "description": proj_desc,
                "theme": proj_theme,
                "CT": proj_CT,
                "province": proj_province,
                "budget": proj_budget,
                "status": proj_status,
                "date_added": str(datetime.utcnow())
            }
            pid = add_project_qdrant(new_proj)
            if pid:
                st.success(f"Project added to Qdrant with ID: {pid}")
                st.rerun()
            else:
                st.error("Failed to add project (Qdrant error).")
        
        st.markdown("---")
        
        # ---- Update Existing Project ----
        st.markdown("#### Update Existing Project (Qdrant)")
        if projects:
            # Prepare options for update: display title with id for clarity
            project_options = {f"{p.get('title', 'Untitled')} (ID: {p.get('project_id', p.get('id'))})": p for p in projects}
            selected_proj_label = st.selectbox("Select a Project to Update", list(project_options.keys()))
            selected_proj = project_options[selected_proj_label]
            # Pre-fill the form with existing details
            with st.form("update_project_form", clear_on_submit=True):
                up_title = st.text_input("Project Title", value=selected_proj.get("title", ""))
                up_desc = st.text_area("Project Description", value=selected_proj.get("description", ""))
                up_theme = st.text_input("Theme", value=selected_proj.get("theme", ""))
                up_CT = st.text_input("CT / City", value=selected_proj.get("CT", ""))
                up_province = st.text_input("Province", value=selected_proj.get("province", ""))
                up_budget = st.number_input("Budget", min_value=0, step=1000, value=int(selected_proj.get("budget", 0)))
                up_status = st.selectbox("Status", ["Planned", "In Progress", "Completed"], index=["Planned", "In Progress", "Completed"].index(selected_proj.get("status", "Planned")))
                update_btn = st.form_submit_button("Update Project")
            if update_btn:
                updated_proj = {
                    "title": up_title,
                    "description": up_desc,
                    "theme": up_theme,
                    "CT": up_CT,
                    "province": up_province,
                    "budget": up_budget,
                    "status": up_status,
                    "date_updated": str(datetime.utcnow())
                }
                # Use the project ID from the selected project (assuming it is stored as 'project_id' or 'id')
                proj_id = selected_proj.get("project_id", selected_proj.get("id"))
                if update_project_qdrant(proj_id, updated_proj):
                    st.success(f"Project with ID {proj_id} updated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to update project.")
        else:
            st.info("No projects available to update.")
        
        st.markdown("---")
        
        # ---- Delete a Project ----
        st.markdown("#### Delete a Project (Qdrant)")
        if projects:
            # Let admin select a project by title (with id) for deletion
            delete_options = {f"{p.get('title', 'Untitled')} (ID: {p.get('project_id', p.get('id'))})": p for p in projects}
            proj_to_delete_label = st.selectbox("Select a Project to Delete", [""] + list(delete_options.keys()))
            if proj_to_delete_label:
                proj_to_delete = delete_options[proj_to_delete_label]
                proj_id = proj_to_delete.get("project_id", proj_to_delete.get("id"))
                if st.button("Confirm Delete Project"):
                    if delete_project_qdrant(proj_id):
                        st.success(f"Project with ID {proj_id} deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete project.")
        else:
            st.info("No projects available to delete.")
        # ---- Additional Statistics and Filtering for Projects ----
        st.subheader("Project Statistics & Filtering")
        if projects:
            df_proj = pd.DataFrame(projects)
            
            # Ensure relevant columns exist: 'completion_percentage', 'votes', 'CT', 'province', 'citizen_participation'
            if "completion_percentage" not in df_proj.columns:
                # For demo, simulate random completion percentages if not present
                df_proj["completion_percentage"] = np.random.randint(0, 101, len(df_proj))
            if "votes" in df_proj.columns:
                # Compute total vote_score for each project
                df_proj["vote_score"] = df_proj["votes"].apply(lambda v: v.get("vote_score") if isinstance(v, dict) else 0)
            else:
                df_proj["vote_score"] = np.random.randint(-20, 50, len(df_proj))
            if "citizen_participation" not in df_proj.columns:
                # For demo, simulate citizen involvement (True/False)
                df_proj["citizen_participation"] = np.random.choice([True, False], len(df_proj))
            
            # Filtering controls
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                selected_CT = st.selectbox("Filter by CT / City", ["All"] + sorted(df_proj["CT"].unique().tolist()))
            with col_filter2:
                selected_province = st.selectbox("Filter by Province", ["All"] + sorted(df_proj["province"].unique().tolist()))
            
            if selected_CT != "All":
                df_proj = df_proj[df_proj["CT"] == selected_CT]
            if selected_province != "All":
                df_proj = df_proj[df_proj["province"] == selected_province]
            
            st.markdown("##### Aggregated Metrics")
            avg_completion = df_proj["completion_percentage"].mean() if not df_proj.empty else 0
            total_vote = df_proj["vote_score"].sum() if not df_proj.empty else 0
            citizen_involvement = df_proj["citizen_participation"].sum() if not df_proj.empty else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Completion (%)", f"{avg_completion:.1f}")
            col2.metric("Total Vote Score", total_vote)
            col3.metric("Citizen Involvement (Count)", citizen_involvement)
            
            st.markdown("---")
            st.write("#### Average Completion Percentage per CT")
            comp_CT = df_proj.groupby("CT")["completion_percentage"].mean().reset_index()
            st.plotly_chart(px.bar(comp_CT, x="CT", y="completion_percentage", title="Avg Completion by CT"), use_container_width=True)
            
            st.write("#### Total Vote Score per Province")
            vote_prov = df_proj.groupby("province")["vote_score"].sum().reset_index()
            st.plotly_chart(px.bar(vote_prov, x="province", y="vote_score", title="Total Vote Score by Province"), use_container_width=True)

            
            
        else:
            st.info("No projects found for statistics.")
        
        st.markdown("---")
    
        # ---- Summarize Citizen Comments per Region Using GPT API ----
        st.markdown("---")
        st.subheader("Comments Summarization per Region")

        with st.spinner("Processing comments and checking for updates..."):
            # Retrieve all citizen comments from Qdrant
            comments = load_qdrant_documents("citizen_comments", vector_dim=384)
            # Group comments by region (using 'project_province')
            region_comments = {}
            for comment in comments:
                region = comment.get("project_province", "Unknown")
                text = comment.get("comment_text", "")
                if text:
                    region_comments.setdefault(region, []).append(text)

        summaries = {}
        for region, texts in region_comments.items():
            combined_text = "\n".join(texts)
            current_hash = compute_text_hash(combined_text)
            stored = get_stored_summary(region)
            
            if stored and stored.get("text_hash") == current_hash:
                # Use stored summary if no change
                summaries[region] = stored.get("summary")
            else:
                # If not stored or text has changed, generate a new summary using GPT
                prompt = f"Summarize the following citizen comments from the region '{region}':\n{combined_text}\nSummary:"
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful summarization assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.5,
                    )
                    summary = response["choices"][0]["message"]["content"].strip()
                    summaries[region] = summary
                    # Store the new summary with the computed hash and current timestamp in MongoDB
                    store_summary(region, summary, current_hash)
                except Exception as e:
                    summaries[region] = f"Error summarizing: {e}"

        if summaries:
            for region, summary in summaries.items():
                st.markdown(f"**Region:** {region}")
                st.markdown(f"> {summary}")
                st.markdown("---")
        else:
            st.info("No comments found to summarize.")

    # --- Tab 5: News Article Management (Qdrant: 'hespress_politics_details') ---
    with tabs[4]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-newspaper icon'></i>{t('News Management')}</h2>", unsafe_allow_html=True)

        # Load "news" from hespress_politics_details (dim=1536)
        articles = load_hespress_news()
        if articles:
            st.dataframe(pd.DataFrame(articles))
            if st.button("Export Articles as CSV", key="export_articles"):
                df_articles = pd.DataFrame(articles)
                st.download_button("Download CSV", df_articles.to_csv(index=False), "hespress_news.csv", "text/csv")
        else:
            st.info("No Hespress news articles found.")

        st.markdown("#### Add New News Article (Qdrant, dimension=1536)")
        with st.form("hespress_news_form", clear_on_submit=True):
            article_title = st.text_input("Article Title")
            article_content = st.text_area("Content")
            article_source = st.text_input("Source")
            article_url = st.text_input("URL")
            article_date = st.date_input("Date Published")
            submit_article = st.form_submit_button("Add Article")
        if submit_article:
            new_article = {
                "title": article_title,
                "content": article_content,
                "source": article_source,
                "url": article_url,
                "date_published": str(article_date)
            }
            aid = add_hespress_news(new_article)
            if aid:
                st.success(f"News article added to Qdrant with ID: {aid}")
                st.rerun()
            else:
                st.error("Failed to add Hespress news (Qdrant error).")
        st.markdown("---")
    

    # --- Tab 6: AI & NLP Insights ---
    with tabs[5]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-brain icon'></i>{t('AI Insights')}</h2>", unsafe_allow_html=True)
        st.write("#### Semantic Search Across Content (Qdrant)")

        query_text = st.text_input("Enter query text for semantic search:")
        collection_choice = st.selectbox(
            "Select Qdrant Collection", 
            [
                "citizen_comments",
                "citizen_ideas",
                "hespress_politics_comments",
                "hespress_politics_details",  # dimension=1536
                "municipal_projects",
                "remacto_comments",
                "remacto_projects"
            ]
        )

        if st.button("Search", key="ai_search"):
            if not query_text.strip():
                st.error("Please enter a query text.")
            else:
                # In production, compute the embedding dimension according to the chosen collection.
                # For demonstration, we're defaulting to 384. If you pick "hespress_politics_details"
                # (which is 1536), you must adapt accordingly.
                query_dim = 384
                if collection_choice == "hespress_politics_details":
                    query_dim = 1536
                query_vector = np.random.rand(query_dim).tolist()

                results = perform_semantic_search(collection_choice, query_vector, top=5)
                st.write("Search Results:")
                for point in results:
                    st.write(point.payload)

        # Word Cloud demo
        st.write("#### Keyword Insights (Word Cloud)")
        sample_keywords = ["transparency", "participation", "innovation", "development", "citizenship", "open government"]
        wc_path = generate_word_cloud(sample_keywords)
        st.image(wc_path, caption="Word Cloud of Frequently Occurring Keywords", use_column_width=True)

    # --- Tab 7: System Health & Settings ---
    with tabs[6]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-server icon'></i>{t('System Health')}</h2>", unsafe_allow_html=True)
        cpu_percent, mem_percent = load_system_health()
        col1, col2 = st.columns(2)
        col1.metric("CPU Usage (%)", cpu_percent)
        col2.metric("Memory Usage (%)", mem_percent)
        st.write("**MongoDB Status:** Connected")
        st.write("**Qdrant Status:** Connected")

    # --- Tab 8: Audit Logs (Mongo) ---
    with tabs[7]:
        st.markdown(f"<h2 class='section-header'><i class='fas fa-clipboard-list icon'></i>{t('Audit Logs')}</h2>", unsafe_allow_html=True)
        audit_logs = load_audit_logs()
        if audit_logs:
            st.dataframe(audit_logs)
            df_audit = pd.DataFrame(audit_logs)
            st.download_button("Export Audit Logs", df_audit.to_csv(index=False), "audit_logs.csv", "text/csv")
        else:
            st.info("No audit logs found.")

    # --- Tab 9: Qdrant Metrics ---
    with tabs[8]:
        st.markdown("<h2 class='section-header'><i class='fas fa-database icon'></i>" + t("Qdrant Metrics") + "</h2>", unsafe_allow_html=True)
        # List out all Qdrant collections you use
        qdrant_collections = [
            "citizen_comments",
            "citizen_ideas",
            "hespress_politics_comments",
            "hespress_politics_details",  # your "news"
            "municipal_projects",
            "remacto_comments",
            "remacto_projects"
        ]
        qdrant_info_list = []
        for coll in qdrant_collections:
            try:
                info = load_qdrant_collection_info(coll)
                st.write(f"**{coll}**")
                info_dict = info.dict()
                st.json(info_dict)
                qdrant_info_list.append({"collection": coll, "info": info_dict})
            except Exception as e:
                st.error(f"Error loading Qdrant info for {coll}: {e}")
        if qdrant_info_list:
            st.download_button(
                "Export Qdrant Info",
                pd.DataFrame(qdrant_info_list).to_json(orient="records"),
                "qdrant_info.json",
                "application/json"
            )

    # --- Tab 10: Semantic Search ---
    with tabs[9]:
        st.markdown("<h2 class='section-header'><i class='fas fa-search-plus icon'></i>" + t("Semantic Search") + "</h2>", unsafe_allow_html=True)
        st.write("Input a query text to perform semantic search over a selected Qdrant collection.")

        query_text = st.text_input("Enter query text (Semantic Search):", key="semantic_query")
        collection_choice = st.selectbox(
            "Select Qdrant Collection",
            [
                "citizen_comments",
                "citizen_ideas",
                "hespress_politics_comments",
                "hespress_politics_details",  # dimension=1536
                "municipal_projects",
                "remacto_comments",
                "remacto_projects"
            ],
            key="semantic_collection"
        )

        if st.button("Perform Semantic Search", key="perform_semantic"):
            if not query_text.strip():
                st.error("Please enter a query text.")
            else:
                # Adjust dimension if user picks 'hespress_politics_details'
                query_dim = 384
                if collection_choice == "hespress_politics_details":
                    query_dim = 1536

                query_vector = np.random.rand(query_dim).tolist()
                results = perform_semantic_search(collection_choice, query_vector, top=5)
                st.write("Semantic Search Results:")
                for point in results:
                    st.write(point.payload)

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():

    if st.session_state.get("role") != "admin":
        st.error(" ⚠️ Access Denied. Admins only.")
        return

    dashboard_admin()

if __name__ == "__main__":
    main()

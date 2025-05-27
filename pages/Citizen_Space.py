import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pymongo import MongoClient
import hashlib
import plotly.express as px
import random
import uuid  # New import for generating UUIDs
import hashlib
import json
import random
import hashlib
import json
from datetime import datetime
import pandas as pd
import plotly.express as px

# --- New Helper Function to Load Points (including vectors) from Qdrant ---
def load_qdrant_points(collection_name: str, vector_dim: int):
    client = get_qdrant_client()
    all_points = []
    # Use the correct parameter: with_vectors=True
    points, next_page_offset = client.scroll(
        collection_name=collection_name, 
        limit=100, 
        with_payload=True, 
        with_vectors=True
    )
    all_points.extend([{"payload": pt.payload, "vector": pt.vector} for pt in points])
    while next_page_offset is not None:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            offset=next_page_offset,
            limit=100,
            with_payload=True,
            with_vectors=True
        )
        all_points.extend([{"payload": pt.payload, "vector": pt.vector} for pt in points])
    client.close()
    return all_points




# -----------------------------------------------------------------------------
# SETUP & HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Set OpenAI API key from Streamlit secrets (or your preferred mechanism)
openai.api_key = st.secrets["openai"]["api_key"]

def get_mongo_client():
    return MongoClient("mongodb://ac-aurbbb0-shard-00-01.mvvbpez.mongodb.net:27017")

def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

def load_qdrant_documents(collection_name: str, vector_dim: int):
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
    client.close()
    return all_docs

def search_qdrant(collection_name: str, query_vector: list, top: int = 5):
    client = get_qdrant_client()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top,
        with_payload=True
    )
    client.close()
    return results

def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_answer(prompt: str) -> dict:
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        cache_doc = db["chatbot_cache"].find_one({"prompt_hash": compute_hash(prompt)})
        return cache_doc
    except Exception as e:
        st.error(f"Error retrieving cached answer: {e}")
        return None
    finally:
        client.close()
# Additional helper: Get project details from Qdrant "municipal_projects"
def get_project_details(project_id: str):
    projects = load_qdrant_documents("municipal_projects", vector_dim=384)
    for proj in projects:
        if proj.get("project_id") == project_id:
            return proj
    return None


def update_user_password(username: str, new_password: str) -> bool:
    """Update the password hash for the given username in the 'users' collection."""
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
        result = db["users"].update_one({"username": username}, {"$set": {"password_hash": new_hash}})
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error updating password: {e}")
        return False
    finally:
        client.close()

def verify_user(username: str, password: str):
    """Verify user's credentials. Returns (True, role) if valid, else (False, None)."""
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        user = db["users"].find_one({"username": username})
        if user and user["password_hash"] == hashlib.sha256(password.encode()).hexdigest():
            return True, user.get("role")
        else:
            return False, None
    except Exception as e:
        st.error(f"Error verifying user: {e}")
        return False, None
    finally:
        client.close()

def store_cached_answer(prompt: str, answer: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {
            "prompt_hash": compute_hash(prompt),
            "prompt": prompt,
            "answer": answer,
            "timestamp": datetime.utcnow()
        }
        db["chatbot_cache"].update_one({"prompt_hash": doc["prompt_hash"]}, {"$set": doc}, upsert=True)
    except Exception as e:
        st.error(f"Error storing cached answer: {e}")
    finally:
        client.close()

def store_chat_history(username: str, prompt: str, answer: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {
            "username": username,
            "prompt": prompt,
            "answer": answer,
            "timestamp": datetime.utcnow()
        }
        db["chat_history"].insert_one(doc)
    except Exception as e:
        st.error(f"Error storing chat history: {e}")
    finally:
        client.close()

def get_chat_history(username: str, limit: int = 10):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        docs = list(db["chat_history"].find({"username": username}).sort("timestamp", -1).limit(limit))
        return docs
    except Exception as e:
        st.error(f"Error retrieving chat history: {e}")
        return []
    finally:
        client.close()

def update_citizen_idea_vote(idea_id: str, vote_action: str) -> bool:
    """
    Update the vote count for a citizen idea in the 'citizen_ideas' Qdrant collection.
    vote_action: "Upvote" or "Downvote".
    Returns True if update succeeded, False otherwise.
    """
    try:
        client = get_qdrant_client()
        # Load all ideas from citizen_ideas collection
        ideas = load_qdrant_documents("citizen_ideas", vector_dim=384)
        target_idea = None
        for idea in ideas:
            if idea.get("idea_id") == idea_id:
                target_idea = idea
                break
        if not target_idea:
            st.error("Idea not found.")
            return False
        # Initialize votes if not present
        if "votes" not in target_idea:
            target_idea["votes"] = {"thumb_up": 0, "thumb_down": 0, "vote_score": 0}
        if vote_action == "Upvote":
            target_idea["votes"]["thumb_up"] += 1
        elif vote_action == "Downvote":
            target_idea["votes"]["thumb_down"] += 1
        # Recalculate vote score: (thumb_up - thumb_down)
        target_idea["votes"]["vote_score"] = target_idea["votes"]["thumb_up"] - target_idea["votes"]["thumb_down"]
        # Preserve the original vector if possible; here we simulate with a dummy vector.
        vector = np.random.rand(384).tolist()  # In production, store and re-use the original vector.
        point = PointStruct(
            id=idea_id,
            vector=vector,
            payload=target_idea
        )
        client.upsert(collection_name="citizen_ideas", points=[point])
        return True
    except Exception as e:
        st.error(f"Error updating vote: {e}")
        return False
    finally:
        client.close()
# -----------------------------------------------------------------------------
# CITIZEN IDEA STATISTICS
# -----------------------------------------------------------------------------
# Helper functions for caching GPT summaries in MongoDB
def get_cached_summary(summary_key: str) -> dict:
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        summary_doc = db["gpt_summaries"].find_one({"summary_key": summary_key})
        return summary_doc
    except Exception as e:
        st.error(f"Error retrieving cached summary: {e}")
        return None
    finally:
        client.close()

def store_cached_summary(summary_key: str, summary: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {
            "summary_key": summary_key,
            "summary": summary,
            "timestamp": datetime.utcnow()
        }
        db["gpt_summaries"].update_one({"summary_key": summary_key}, {"$set": doc}, upsert=True)
    except Exception as e:
        st.error(f"Error storing cached summary: {e}")
    finally:
        client.close()
# Here we assume that citizen ideas are stored in the "citizen_ideas" collection.
# For each idea, if its "idea_id" is not present in any project's "linked_project_ids"
# from the "municipal_projects" collection, then it is considered rejected.
def get_citizen_idea_stats(username: str):
    # Load citizen ideas and projects
    ideas = load_qdrant_documents("citizen_ideas", vector_dim=384)
    projects = load_qdrant_documents("municipal_projects", vector_dim=384)
    # Build a set of all linked idea IDs from projects
    linked_ids = set()
    for proj in projects:
        linked = proj.get("linked_idea_ids", [])
        if isinstance(linked, list):
            for idea_id in linked:
                linked_ids.add(idea_id)
    # Process citizen ideas for the given user
    approved = []
    rejected = []
    for idea in ideas:
        if idea.get("citizen_name", "").lower() == username.lower():
            idea_id = idea.get("idea_id")
            # If idea_id is present in linked_ids, consider it approved; else, rejected.
            if idea_id and idea_id in linked_ids:
                approved.append(idea)
            else:
                # If idea_id is missing or not linked, mark it as rejected with a reason.
                idea["rejection_reason"] = "Idea not linked to any project"
                rejected.append(idea)
    total_approved = len(approved)
    pos = sum(1 for a in approved if a.get("sentiment", "").upper() == "POS")
    neg = sum(1 for a in approved if a.get("sentiment", "").upper() == "NEG")
    neu = total_approved - pos - neg
    # Rejected details: include comment text and rejection reason.
    rejected_details = [{"comment": r.get("comment_text"), "reason": r.get("rejection_reason")} for r in rejected]
    return {"total_comments": total_approved, "positive": pos, "negative": neg, "neutral": neu, "rejected": rejected_details}

# -----------------------------------------------------------------------------
# CITIZEN IDEA SUBMISSION
# -----------------------------------------------------------------------------
# Generate a UUID in standard format and add the idea to the "citizen_ideas" collection
def add_citizen_idea(idea_data: dict):
    client = get_qdrant_client()
    embedding_dim = 384
    random_vector = np.random.rand(embedding_dim).tolist()
    new_id = str(uuid.uuid4())
    point = PointStruct(id=new_id, vector=random_vector, payload=idea_data)
    client.upsert(collection_name="citizen_ideas", points=[point])
    client.close()
    return new_id

# GPT-based offensive language detection
def is_offensive_with_gpt(text: str) -> bool:
    try:
        mod_response = openai.Moderation.create(input=text)
        return mod_response["results"][0]["flagged"]
    except Exception as e:
        st.error(f"Error during moderation check: {e}")
        return False

# GPT-based extraction of additional idea details
def extract_idea_details(idea_text: str) -> dict:
    prompt = (
        f"Extract the following details from the citizen idea/comment below:\n\n"
        f"Comment Text: {idea_text}\n\n"
        "Return a JSON object with the following keys (if not applicable, return an empty string or null):\n"
        "- axis: main category (e.g., economic development, environmental improvement)\n"
        "- challenge: the problem stated in the idea\n"
        "- solution: the proposed solution\n"
        "- city: city name\n"
        "- commune: commune name\n"
        "- province: province name\n"
        "- CT: regional unit\n"
        "- channel: how the idea was submitted (e.g., phone, SNS)\n"
        "- sentiment: overall sentiment (POS, NEG, or NEU)\n"
        "- polarity: a float representing the sentiment polarity\n"
        "- topic: the main topic of the idea\n"
        "- keywords: a list of key terms from the idea\n\n"
        "Only return the JSON object."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an NLP extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5,
        )
        output = response["choices"][0]["message"]["content"].strip()
        return json.loads(output)
    except Exception as e:
        st.error(f"Error extracting idea details: {e}")
        return {
            "axis": "",
            "challenge": "",
            "solution": "",
            "city": "",
            "commune": "",
            "province": "",
            "CT": "",
            "channel": "",
            "sentiment": "",
            "polarity": 0.0,
            "topic": "",
            "keywords": []
        }

# -----------------------------------------------------------------------------
# GET PROJECT DETAILS (for linked ideas)
# -----------------------------------------------------------------------------
def get_project_details(project_id: str):
    projects = load_qdrant_documents("municipal_projects", vector_dim=384)
    for proj in projects:
        if proj.get("project_id") == project_id:
            return proj
    return None

# -----------------------------------------------------------------------------
# DEFINE QDRANT COLLECTION NAMES (all required collections)
# -----------------------------------------------------------------------------
qdrant_collections = [
    "citizen_comments",
    "citizen_ideas",
    "hespress_politics_comments",
    "hespress_politics_details",  # expects 1536-dim vectors
    "municipal_projects",
    "remacto_comments",
    "remacto_projects"
]

# -----------------------------------------------------------------------------
# SIDEBAR SETTINGS & EXTRA CHATBOT FEATURES
# -----------------------------------------------------------------------------
if st.session_state.get("role") != "citizen":
    st.error("Access Denied. Only citizens can access this Citizen Page.")
    st.stop()
st.sidebar.header("LLM & Chatbot Settings")
llm_options = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "O1 (Custom Model)": "o1"
}
selected_llm = st.sidebar.selectbox("Select LLM", list(llm_options.keys()))
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=150, step=50)
cost_opt = st.sidebar.checkbox("Enable Cost Optimization", value=True)
# Extra Chatbot Options (5 additional features)
show_history = st.sidebar.checkbox("Show My Chat History", value=True)
export_history = st.sidebar.button("Export Chat History as CSV")
chat_tone = st.sidebar.selectbox("Chat Tone", ["Formal", "Casual", "Friendly"])
extended_context = st.sidebar.checkbox("Enable Extended Context", value=False)
if st.sidebar.button("Refresh Chat History"):
    st.rerun()
if st.sidebar.button("Clear Chat History"):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        db["chat_history"].delete_many({"username": st.session_state.get("username")})
        st.sidebar.success("Chat history cleared!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing chat history: {e}")
    finally:
        client.close()
st.sidebar.markdown("---")
st.sidebar.write("Adjust parameters to balance cost and performance.")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
if st.session_state["openai_api_key"]:
    openai.api_key = st.session_state["openai_api_key"]

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR MODERN DARK CHATBOT STYLE
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# ENFORCE CITIZEN LOGIN (Using same login mechanism as Login.py)
# -----------------------------------------------------------------------------
from streamlit_cookies_manager import EncryptedCookieManager

# Initialize cookie manager
cookies = EncryptedCookieManager(prefix="civic_", password=os.environ.get("COOKIE_PASSWORD", "YOUR_STRONG_PASSWORD"))

if cookies.ready():
    # Fallback: set session state from cookies if not already set
    if "username" not in st.session_state and cookies.get("username"):
        st.session_state["username"] = cookies.get("username")

    if "role" not in st.session_state and cookies.get("role"):
        st.session_state["role"] = cookies.get("role")

# Final check (after fallback)
if not st.session_state.get("username") or not st.session_state.get("role"):
    st.error("Access Denied. Please log in to use the Citizen Space.")

    st.markdown("### Login")
    with st.form("login_form", clear_on_submit=True):
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_submitted = st.form_submit_button("Login")
    if login_submitted:
        try:
            client = get_mongo_client()
            db = client["CivicCatalyst"]
            user = db["users"].find_one({"username": login_username})
            if user and user["password_hash"] == hashlib.sha256(login_password.encode()).hexdigest():
                st.session_state["username"] = login_username
                st.session_state["role"] = user.get("role", "citizen")
                st.success(f"Login successful! Welcome {login_username}.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        except Exception as e:
            st.error(f"Error during login: {e}")
        finally:
            client.close()
    st.stop()

# Ensure the logged-in role is "citizen"
if st.session_state.get("role") != "citizen":
    st.error("Access Denied. This page is for citizens only.")
    st.stop()

# -----------------------------------------------------------------------------
# CITIZEN SPACE: TABS FOR MULTIPLE FEATURES
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Dashboard", 
    "👍👎 Pros & Cons",
    "📝 Submit Idea", 
    "📢 Idea Feed", 
    "🏗 Projects", 
    "📰 News", 
    "👤 Profile", 
    "💬 Chat with Assistant"
])

# -----------------------------------------------------------------------------
# TAB 1: Dashboard
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Citizen Dashboard")
    st.write(f"Welcome, **{st.session_state['username']}**! :sparkles:")

    # -------------------------------------------------------------------------
    # Load All Submissions (Ideas and Comments) for the Logged-In Citizen
    # -------------------------------------------------------------------------
    all_submissions = [
        s for s in load_qdrant_documents("citizen_ideas", vector_dim=384)
        if s.get("citizen_name", "").lower() == st.session_state["username"].lower()
    ]

    # Separate submissions based on their status
    approved_submissions = [s for s in all_submissions if s.get("status", "").lower() == "approved"]
    pending_submissions = [s for s in all_submissions if s.get("status", "").lower() == "pending"]

    # -------------------------------------------------------------------------
    # Check if an Approved Submission is Linked to a Municipal Project
    # -------------------------------------------------------------------------
    def is_submission_linked(idea_id: str) -> bool:
        projects = load_qdrant_documents("municipal_projects", vector_dim=384)
        for proj in projects:
            linked_ids = proj.get("linked_idea_ids", [])
            if idea_id in linked_ids:
                return True
        return False

    # For approved submissions, classify them as linked or rejected based on project association.
    linked_approved = []
    rejected_approved = []
    for sub in approved_submissions:
        idea_id = sub.get("idea_id", "")
        if idea_id and is_submission_linked(idea_id):
            linked_approved.append(sub)
        else:
            rejected_approved.append(sub)

    # -------------------------------------------------------------------------
    # Determine Submission Type: Idea vs. Comment
    # (Idea: no project_id provided; Comment: project_id exists)
    # -------------------------------------------------------------------------
    def submission_type(sub):
        return "Comment" if sub.get("project_id") else "Idea"

    # Split approved submissions by type (only for those already moderated)
    approved_ideas = [s for s in approved_submissions if submission_type(s) == "Idea"]
    approved_comments = [s for s in approved_submissions if submission_type(s) == "Comment"]

    # Also, separate pending submissions by type (linking is not applicable yet)
    pending_ideas = [s for s in pending_submissions if submission_type(s) == "Idea"]
    pending_comments = [s for s in pending_submissions if submission_type(s) == "Comment"]

    total_submissions = len(all_submissions)
    total_approved = len(approved_submissions)
    total_pending = len(pending_submissions)
    total_linked = len(linked_approved)
    total_rejected = len(rejected_approved)

    # -------------------------------------------------------------------------
    # Display Key Metrics
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Submissions", total_submissions)
    col2.metric("Approved", total_approved)
    col3.metric("Pending", total_pending)
    col4.metric("Linked (Approved)", total_linked)
    col5.metric("Rejected (Approved)", total_rejected)

    st.markdown("---")
    st.markdown("### Breakdown by Submission Type")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Approved Submissions")
        st.write(f"Ideas: **{len(approved_ideas)}**")
        st.write(f"Comments: **{len(approved_comments)}**")
    with colB:
        st.subheader("Pending Submissions")
        st.write(f"Ideas: **{len(pending_ideas)}**")
        st.write(f"Comments: **{len(pending_comments)}**")

    # -------------------------------------------------------------------------
    # Sentiment Analysis for Approved Submissions
    # -------------------------------------------------------------------------
    pos = sum(1 for s in approved_submissions if s.get("sentiment", "").upper() == "POS")
    neg = sum(1 for s in approved_submissions if s.get("sentiment", "").upper() == "NEG")
    neu = total_approved - pos - neg

    st.markdown("---")
    st.markdown("### Sentiment Distribution (Approved)")
    df_sent = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [pos, neg, neu]
    })
    st.plotly_chart(
        px.pie(df_sent, names="Sentiment", values="Count", title="Sentiment Distribution"),
        use_container_width=True
    )

    # -------------------------------------------------------------------------
    # Approved Submissions: Linked vs. Rejected Visualization
    # -------------------------------------------------------------------------
    st.markdown("### Approved Submissions: Linked vs. Rejected")
    df_link = pd.DataFrame({
        "Status": ["Linked", "Rejected"],
        "Count": [total_linked, total_rejected]
    })
    st.plotly_chart(
        px.pie(df_link, names="Status", values="Count", title="Linked vs. Rejected Approved Submissions"),
        use_container_width=True
    )

    # -------------------------------------------------------------------------
    # Trend Analysis: Submissions Over Time (All Submissions)
    # -------------------------------------------------------------------------
    dates = []
    for sub in all_submissions:
        try:
            dates.append(datetime.strptime(sub.get("date_submitted", "2020-01-01"), "%Y-%m-%d"))
        except Exception:
            continue
    if dates:
        df_time = pd.DataFrame({"date": dates})
        df_time.sort_values("date", inplace=True)
        df_time["Month"] = df_time["date"].dt.to_period("M").astype(str)
        monthly_counts = df_time.groupby("Month").size().reset_index(name="Submissions")
        st.markdown("---")
        st.markdown("### Submissions Trend Over Time")
        st.plotly_chart(
            px.line(monthly_counts, x="Month", y="Submissions", title="Submissions Over Time", markers=True),
            use_container_width=True
        )
        st.plotly_chart(
            px.bar(monthly_counts, x="Month", y="Submissions", title="Submissions per Month"),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Scatter Plot: Sentiment Polarity Over Time for Approved Submissions
    # -------------------------------------------------------------------------
    polarity_data = []
    for sub in approved_submissions:
        try:
            d = datetime.strptime(sub.get("date_submitted", "2020-01-01"), "%Y-%m-%d")
            p = float(sub.get("polarity", 0.0))
            polarity_data.append({"Date": d, "Polarity": p})
        except Exception:
            continue
    if polarity_data:
        df_polarity = pd.DataFrame(polarity_data)
        st.markdown("---")
        st.markdown("### Sentiment Polarity Over Time")
        st.plotly_chart(
            px.scatter(df_polarity, x="Date", y="Polarity", title="Sentiment Polarity Over Time", trendline="lowess"),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Recent Submissions List with Details
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Recent Submissions")
    if all_submissions:
        sorted_subs = sorted(all_submissions, key=lambda x: x.get("date_submitted", "2020-01-01"), reverse=True)
        for sub in sorted_subs[:5]:
            sub_type = submission_type(sub)
            date_str = sub.get("date_submitted", "Unknown Date")
            status = sub.get("status", "Unknown").capitalize()
            sentiment = sub.get("sentiment", "N/A")
            polarity = sub.get("polarity", 0.0)
            detail = f"**Type:** {sub_type} | **Status:** {status} | **Date:** {date_str} | **Sentiment:** {sentiment} | **Polarity:** {polarity}"
            if sub_type == "Comment" and sub.get("project_id"):
                proj = get_project_details(sub.get("project_id"))
                if proj:
                    detail += f" | **Project:** {proj.get('title', 'N/A')}"
            st.markdown(f"- {detail}")
    else:
        st.info("No submissions found.")

    # -------------------------------------------------------------------------
    # Additional Section: List of Your Project Comments
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Your Project Comments")
    project_comments = [s for s in all_submissions if submission_type(s) == "Comment"]
    if project_comments:
        for comment in sorted(project_comments, key=lambda x: x.get("date_submitted", "2020-01-01"), reverse=True):
            date_str = comment.get("date_submitted", "Unknown Date")
            project_id = comment.get("project_id", "N/A")
            comment_text = comment.get("comment_text", "No text provided")
            # Retrieve project details if available
            proj = get_project_details(project_id)
            project_title = proj.get("title", "Unknown") if proj else "Unknown"
            st.markdown(f"**Date:** {date_str} | **Project:** {project_title} (ID: {project_id})")
            st.markdown(f"**Comment:** {comment_text}")
            st.markdown("---")
    else:
        st.info("No project comments found.")

    #----------------------------------------------------------
    st.header("Project Summary & Citizen Comments")
    st.write(
        "Use the filters below to refine the projects. For each project, view its details and a GPT-generated summary of the citizen comments associated with it.\n\n"
        "**Note:** Summaries are re-generated only if the citizen comments change (as determined by an MD5 hash of the concatenated comments)."
    )

    # Load projects and citizen comments from Qdrant
    projects = load_qdrant_documents("municipal_projects", vector_dim=384)
    comments = load_qdrant_documents("citizen_comments", vector_dim=384)

    # Convert projects to DataFrame for filtering and visualization
    if projects:
        df_projects = pd.DataFrame(projects)
    else:
        st.info("No project data available.")
        df_projects = pd.DataFrame()

    # ---------------------------
    # Filtering Options for Projects
    # ---------------------------
    st.markdown("#### Filter Projects")

    # Filter by Status
    if "status" in df_projects.columns:
        statuses = sorted(df_projects["status"].dropna().unique().tolist())
        selected_status = st.multiselect("Status", options=statuses, default=statuses, key="sts_multiselect_1")
    else:
        selected_status = None

    # Filter by CT / City
    if "CT" in df_projects.columns:
        CTs = sorted(df_projects["CT"].dropna().unique().tolist())
        selected_CT = st.multiselect("CT / City", options=CTs, default=CTs, key="ct_multiselect_1")
    else:
        selected_CT = None

    # Filter by Province
    if "province" in df_projects.columns:
        provinces = sorted(df_projects["province"].dropna().unique().tolist())
        selected_province = st.multiselect("Province", options=provinces, default=provinces, key="pr_multiselect_1")
    else:
        selected_province = None

    # Filter by Commune
    if "commune" in df_projects.columns:
        communes = sorted(df_projects["commune"].dropna().unique().tolist())
        selected_commune = st.multiselect("Commune", options=communes, default=communes, key="cm_multiselect_1")
    else:
        selected_commune = None

    # Filter by Budget Range
    if "budget" in df_projects.columns and not df_projects["budget"].empty:
        min_budget = int(df_projects["budget"].min())
        max_budget = int(df_projects["budget"].max())
        budget_range = st.slider("Budget Range", min_budget, max_budget, (min_budget, max_budget), key="budget_range_slider")
    else:
        budget_range = None

    # Filter by Completion Percentage
    if "completion_percentage" in df_projects.columns and not df_projects["completion_percentage"].empty:
        min_comp = float(df_projects["completion_percentage"].min())
        max_comp = float(df_projects["completion_percentage"].max())
        completion_range = st.slider("Completion % Range", min_comp, max_comp, (min_comp, max_comp), key="completion_range_slider")
    else:
        completion_range = None

    # Apply filters to the DataFrame
    df_filtered = df_projects.copy()
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    if selected_CT:
        df_filtered = df_filtered[df_filtered["CT"].isin(selected_CT)]
    if selected_province:
        df_filtered = df_filtered[df_filtered["province"].isin(selected_province)]
    if selected_commune:
        df_filtered = df_filtered[df_filtered["commune"].isin(selected_commune)]
    if budget_range and "budget" in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered["budget"] >= budget_range[0]) & (df_filtered["budget"] <= budget_range[1])]
    if completion_range and "completion_percentage" in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered["completion_percentage"] >= completion_range[0]) & 
                                (df_filtered["completion_percentage"] <= completion_range[1])]

    st.markdown("#### Filtered Projects")
    st.dataframe(df_filtered)

    # ---------------------------
    # Fancy Visualizations
    # ---------------------------
    st.markdown("#### Projects by Status")
    if "status" in df_filtered.columns:
        status_counts = df_filtered["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.plotly_chart(
            px.pie(status_counts, names="Status", values="Count", title="Projects by Status"),
            use_container_width=True
        )

    st.markdown("#### Budget vs. Completion Percentage")
    if "budget" in df_filtered.columns and "completion_percentage" in df_filtered.columns:
        st.plotly_chart(
            px.scatter(df_filtered, x="budget", y="completion_percentage", hover_data=["title"], 
                    title="Budget vs. Completion Percentage"),
            use_container_width=True
        )

    # ---------------------------
    # Summarization Trigger Button
    # ---------------------------
    if st.button("Show Summaries for Filtered Projects"):
        # Only proceed if filtering is applied (i.e. filtered list is a proper subset)
        if df_filtered.empty or len(df_filtered) == len(df_projects):
            st.warning("Please apply filters to refine the project list before summarizing.")
        else:
            st.markdown("#### Project Details & Citizen Comments Summaries")
            
            
            st.markdown('<div class="scrollable-summary">', unsafe_allow_html=True)
            # For each filtered project, display an expander with details and summarized comments
            for index, proj in df_filtered.iterrows():
                project_id = proj.get("project_id", "")
                project_title = proj.get("title", "Untitled Project")
                with st.expander(f"{project_title} (ID: {project_id})", expanded=False):
                    # Display key project properties
                    st.markdown(f"**Themes:** {proj.get('themes', 'N/A')}")
                    st.markdown(f"**CT:** {proj.get('CT', 'N/A')} | **Province:** {proj.get('province', 'N/A')} | **Commune:** {proj.get('commune', 'N/A')}")
                    st.markdown(f"**Status:** {proj.get('status', 'N/A')} | **Completion:** {proj.get('completion_percentage', 'N/A')}%")
                    st.markdown(f"**Budget:** {proj.get('budget', 'N/A')} | **Funders:** {proj.get('funders', 'N/A')}")
                    
                    # Retrieve all citizen comments for this project from the citizen_comments collection
                    proj_comments = [c for c in comments if c.get("project_id") == project_id]
                    st.markdown(f"**Total Citizen Comments:** {len(proj_comments)}")
                    if proj_comments:
                        # Concatenate comment texts for summarization (limit to a reasonable character count)
                        all_comments_text = "\n\n".join([c.get("comment_text", "") for c in proj_comments])
                        if len(all_comments_text) > 3000:
                            all_comments_text = all_comments_text[:3000]  # trim to avoid exceeding token limits

                        # Compute a cache key using the project ID and the MD5 hash of the concatenated comments.
                        cache_key = f"project_summary_{project_id}_{compute_hash(all_comments_text)}"
                        
                        # Check for a cached summary
                        cached_summary = get_cached_summary(cache_key)
                        if cached_summary:
                            summary = cached_summary.get("summary")
                            st.markdown("**Cached Comments Summary:**")
                            st.info(summary)
                        else:
                            # Use GPT to summarize the citizen comments since no valid cache exists.
                            try:
                                summarization_prompt = (
                                    "Summarize the following citizen comments for the project in a concise manner. "
                                    "Focus on the key points, concerns, and suggestions provided by the citizens. "
                                    "If there are multiple opinions, mention the predominant sentiment.\n\n"
                                    f"Comments:\n{all_comments_text}\n\nSummary:"
                                )
                                with st.spinner("Summarizing citizen comments..."):
                                    summary_response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are an assistant that summarizes citizen comments."},
                                            {"role": "user", "content": summarization_prompt}
                                        ],
                                        max_tokens=300,
                                        temperature=0.5,
                                    )
                                summary = summary_response["choices"][0]["message"]["content"].strip()
                                # Cache the summary for future use
                                store_cached_summary(cache_key, summary)
                                st.markdown("**Comments Summary:**")
                                st.info(summary)
                            except Exception as e:
                                st.error(f"Error summarizing comments: {e}")
                                summary = "Summary unavailable."
                        
                        # Use a checkbox to toggle raw comments (avoids nested expanders)
                        if st.checkbox("Show Raw Comments", key=f"raw_{project_id}"):
                            for c in proj_comments:
                                c_date = c.get("date_submitted", "Unknown Date")
                                c_author = c.get("citizen_name", "Anonymous")
                                c_text = c.get("comment_text", "")
                                st.markdown(f"**{c_author}** on {c_date}:")
                                st.markdown(f"> {c_text}")
                                st.markdown("---")
                    else:
                        st.info("No citizen comments for this project.")
            st.markdown('</div>', unsafe_allow_html=True)


import json
from datetime import datetime

def update_comment_sentiment_in_mongodb(comment_id: str, sentiment: str, polarity: float):
    """
    Upsert the sentiment analysis result for a given comment_id into MongoDB.
    The document is stored in the 'gpt_sentiment_results' collection.
    """
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        db["gpt_sentiment_results"].update_one(
            {"comment_id": comment_id},
            {"$set": {
                "sentiment": sentiment,
                "polarity": polarity,
                "last_updated": datetime.utcnow()
            }},
            upsert=True
        )
    except Exception as e:
        st.error(f"Error updating sentiment in MongoDB for comment {comment_id}: {e}")
    finally:
        client.close()

def get_cached_sentiment(comment_id: str):
    """
    Retrieve cached sentiment and polarity for a given comment_id from MongoDB.
    Returns a tuple (sentiment, polarity) if found; otherwise, returns None.
    """
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = db["gpt_sentiment_results"].find_one({"comment_id": comment_id})
        if doc:
            sentiment = doc.get("sentiment", "NEU")
            polarity = float(doc.get("polarity", 0.0))
            return sentiment, polarity
        else:
            return None
    except Exception as e:
        st.error(f"Error retrieving cached sentiment for comment {comment_id}: {e}")
        return None
    finally:
        client.close()

def get_sentiment_via_gpt(comment_text: str) -> (str, float):
    """
    Use GPT to analyze the sentiment of a comment.
    Returns a tuple of (sentiment, polarity) where sentiment is one of 'POS', 'NEG', or 'NEU',
    and polarity is a float value.
    """
    prompt = (
        f"Determine the sentiment and polarity of the following comment. "
        f"Return only a JSON object with keys 'sentiment' and 'polarity'. "
        f"Sentiment should be one of 'POS', 'NEG', or 'NEU'.\n\n"
        f"Comment: {comment_text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that analyzes sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.5,
        )
        result_text = response["choices"][0]["message"]["content"].strip()
        result = json.loads(result_text)
        sentiment = result.get("sentiment", "NEU").upper()
        polarity = float(result.get("polarity", 0.0))
        return sentiment, polarity
    except Exception as e:
        st.error(f"Error during GPT sentiment analysis: {e}")
        return "NEU", 0.0

# -------------------------------
# New Tab: Pros & Cons with Caching
# -------------------------------
with tabs[1]:
    st.header("Pros & Cons")
    
    # Checkbox to force re-run of GPT sentiment analysis even if cache exists
    run_gpt_analysis = st.checkbox("Force re-run GPT Sentiment Analysis for all comments", value=False)
    
    # Load citizen comments from Qdrant (ensure vector_dim matches your data)
    comments = load_qdrant_documents("citizen_comments", vector_dim=384)
    
    if comments:
        # Process each comment using caching to minimize GPT calls
        for comment in comments:
            comment_text = comment.get("comment_text", "")
            comment_id = comment.get("comment_id")
            if comment_text and comment_id:
                if run_gpt_analysis:
                    # Force re-run GPT analysis
                    sentiment, polarity = get_sentiment_via_gpt(comment_text)
                    update_comment_sentiment_in_mongodb(comment_id, sentiment, polarity)
                else:
                    # Attempt to retrieve cached result
                    cached = get_cached_sentiment(comment_id)
                    if cached:
                        sentiment, polarity = cached
                    else:
                        sentiment, polarity = get_sentiment_via_gpt(comment_text)
                        update_comment_sentiment_in_mongodb(comment_id, sentiment, polarity)
                comment["sentiment"] = sentiment
                comment["polarity"] = polarity
        
        # Separate comments based on sentiment: "POS" for Pros, "NEG" for Cons.
        pros = [c for c in comments if c.get("sentiment", "").upper() == "POS"]
        cons = [c for c in comments if c.get("sentiment", "").upper() == "NEG"]

        # Display summary metrics
        st.subheader("Summary Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Pros", len(pros))
        with col2:
            st.metric("Total Cons", len(cons))
        avg_pros = sum(c.get("polarity", 0.0) for c in pros) / len(pros) if pros else 0.0
        avg_cons = sum(c.get("polarity", 0.0) for c in cons) / len(cons) if cons else 0.0
        st.write(f"Average Pros Polarity: {avg_pros:.2f}")
        st.write(f"Average Cons Polarity: {avg_cons:.2f}")

        st.markdown("---")

        # Display individual comments in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pros (Positive Comments)")
            for pro in pros:
                st.markdown(f"**{pro.get('citizen_name', 'Anonymous')}** on {pro.get('date_submitted', 'Unknown Date')}")
                st.markdown(f"> {pro.get('comment_text', '')}")
                st.markdown(f"*Project:* {pro.get('project_title', 'N/A')}")
                st.markdown("---")
        with col2:
            st.subheader("Cons (Negative Comments)")
            for con in cons:
                st.markdown(f"**{con.get('citizen_name', 'Anonymous')}** on {con.get('date_submitted', 'Unknown Date')}")
                st.markdown(f"> {con.get('comment_text', '')}")
                st.markdown(f"*Project:* {con.get('project_title', 'N/A')}")
                st.markdown("---")
    else:
        st.info("No citizen comments found.")

# -----------------------------------------------------------------------------
# TAB 2: Submit Idea
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Submit Your Idea")
    st.write("Please provide the required information below. All fields marked as **Mandatory** must be completed. Your idea will be reviewed by moderators.")

    with st.form("submit_idea_form", clear_on_submit=True):
        # Mandatory idea text field
        idea_text = st.text_area(
            "Your Idea/Comment (Mandatory)",
            placeholder="Type your idea here...",
            help="Enter the core of your idea or comment. This field is required."
        )
        # Optional field for linking to an existing project
        project_id = st.text_input(
            "Project ID (Optional)",
            placeholder="Link to a project if applicable (optional)"
        )
        # Additional mandatory fields for structured details
        axis_manual = st.selectbox(
            "Idea Category (Mandatory)",
            options=[
                "Economic Development", "Environmental Improvement", "Public Safety",
                "Education", "Healthcare", "Infrastructure", "Other"
            ]
        )
        challenge_manual = st.text_area(
            "Challenge Description (Mandatory)",
            placeholder="Describe the challenge or problem",
            help="Provide a clear description of the problem your idea addresses."
        )
        solution_manual = st.text_area(
            "Proposed Solution (Mandatory)",
            placeholder="Describe your proposed solution",
            help="Explain how your idea can address the challenge."
        )
        topic_manual = st.text_input(
            "Main Topic (Mandatory)",
            placeholder="Enter the main topic of your idea"
        )
        city_manual = st.text_input(
            "City (Mandatory)",
            placeholder="Enter the city"
        )
        commune_manual = st.text_input(
            "Commune (Mandatory)",
            placeholder="Enter the commune"
        )
        province_manual = st.text_input(
            "Province (Mandatory)",
            placeholder="Enter the province"
        )
        channel_manual = st.selectbox(
            "Submission Channel (Mandatory)",
            options=["Phone", "Website", "Social Media", "In-Person", "Other"]
        )
        agree_terms = st.checkbox("I agree to the terms and conditions (Mandatory)")
        submit_btn = st.form_submit_button("Submit Idea")

    if submit_btn:
        # Validate mandatory fields
        if not idea_text.strip():
            st.error("Idea text cannot be empty.")
        elif not topic_manual.strip():
            st.error("Main Topic is required.")
        elif not challenge_manual.strip():
            st.error("Challenge Description is required.")
        elif not solution_manual.strip():
            st.error("Proposed Solution is required.")
        elif not (city_manual.strip() and commune_manual.strip() and province_manual.strip()):
            st.error("Location details (City, Commune, Province) are required.")
        elif not agree_terms:
            st.error("You must agree to the terms and conditions.")
        elif is_offensive_with_gpt(idea_text):
            st.error("Your idea contains offensive language and cannot be submitted.")
        else:
            # Generate a unique idea ID
            idea_id = str(uuid.uuid4())

            # Attempt to extract keywords from the idea text using GPT
            try:
                extraction_prompt = (
                    f"Extract keywords from the following text as a JSON array of strings:\n\n"
                    f"{idea_text}\n\n"
                    "Only return the JSON array."
                )
                extraction_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that extracts keywords."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.5,
                )
                extraction_content = extraction_response["choices"][0]["message"]["content"].strip()
                extracted_keywords = json.loads(extraction_content)
            except Exception as e:
                st.error(f"Error extracting keywords: {e}")
                extracted_keywords = []

            # Determine overall sentiment using GPT
            try:
                sentiment_prompt = (
                    f"Determine the overall sentiment of the following idea text. "
                    f"Return only one of the following codes: POS, NEG, or NEU.\n\n"
                    f"Text: {idea_text}\n\n"
                    "Answer only with POS, NEG, or NEU."
                )
                sentiment_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that determines sentiment."},
                        {"role": "user", "content": sentiment_prompt}
                    ],
                    max_tokens=10,
                    temperature=0.5,
                )
                sentiment_output = sentiment_response["choices"][0]["message"]["content"].strip().upper()
                if sentiment_output not in ["POS", "NEG", "NEU"]:
                    sentiment_output = "NEU"
            except Exception as e:
                st.error(f"Error determining sentiment: {e}")
                sentiment_output = "NEU"

            # If a project ID is provided, try to fetch additional project details
            if project_id.strip():
                proj_details = get_project_details(project_id.strip())
                if proj_details:
                    project_title = proj_details.get("title", "")
                    project_themes = proj_details.get("themes", "")
                    project_CT = proj_details.get("CT", "")
                    project_province = proj_details.get("province", "")
                    project_commune = proj_details.get("commune", "")
                    project_status = proj_details.get("status", "")
                else:
                    project_title = project_themes = project_CT = project_province = project_commune = project_status = ""
            else:
                project_title = project_themes = project_CT = project_province = project_commune = project_status = ""

            # Merge all provided and extracted information into the idea payload
            idea_data = {
                "idea_id": idea_id,
                "citizen_name": st.session_state["username"],
                "comment_text": idea_text,
                "project_id": project_id.strip() if project_id.strip() else None,
                "date_submitted": datetime.utcnow().strftime("%Y-%m-%d"),
                "status": "pending",  # Initially pending moderation
                "axis": axis_manual,
                "challenge": challenge_manual,
                "solution": solution_manual,
                "city": city_manual,
                "commune": commune_manual,
                "province": province_manual,
                "CT": "",  # Not provided manually
                "channel": channel_manual,
                "sentiment": sentiment_output,
                "polarity": 0.0,  # Default value; can be updated later if needed
                "topic": topic_manual,
                "keywords": extracted_keywords,
                "project_title": project_title,
                "project_themes": project_themes,
                "project_CT": project_CT,
                "project_province": project_province,
                "project_commune": project_commune,
                "project_status": project_status,
                "votes": {"thumb_up": 0, "thumb_down": 0, "vote_score": 0}
            }

            # (Optional) Display the submitted data for review/debugging
            st.write("Submitted Data:", idea_data)

            # Store the idea in the Qdrant "citizen_ideas" collection
            new_idea_id = add_citizen_idea(idea_data)
            st.success(f"Idea submitted successfully with ID: {new_idea_id} :white_check_mark:")
    

# -----------------------------------------------------------------------------
# TAB 3: Idea Feed
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Approved Ideas Feed")
    # Load ideas from citizen_ideas collection
    all_ideas = load_qdrant_documents("citizen_ideas", vector_dim=384)
    # Filter only approved ideas
    approved_ideas = [idea for idea in all_ideas if idea.get("status", "").lower() == "approved"]

    if approved_ideas:
        # Initialize vote counts if missing
        for idea in approved_ideas:
            if "votes" not in idea:
                idea["votes"] = {"thumb_up": 0, "thumb_down": 0, "vote_score": 0}
        df_feed = pd.DataFrame(approved_ideas)
        st.dataframe(df_feed)

        # Visualization: Bar Chart of Vote Scores
        vote_data = [{"Idea ID": idea.get("idea_id"), "Vote Score": idea["votes"]["vote_score"]} for idea in approved_ideas]
        df_votes = pd.DataFrame(vote_data)
        st.plotly_chart(
            px.bar(df_votes, x="Idea ID", y="Vote Score", title="Vote Score for Each Idea"),
            use_container_width=True
        )

        st.markdown("### Vote on an Idea")
        # Create a selectbox for ideas to vote on
        idea_options = {f"{idea.get('comment_text', '')[:50]}... (ID: {idea.get('idea_id')})": idea.get('idea_id') for idea in approved_ideas}
        selected_idea_option = st.selectbox("Select an Idea", options=list(idea_options.keys()))
        selected_idea_id = idea_options[selected_idea_option]
        vote_action = st.radio("Your Vote", ["Upvote", "Downvote"])
        if st.button("Submit Vote"):
            if update_citizen_idea_vote(selected_idea_id, vote_action):
                st.success("Your vote has been recorded!")
                st.rerun()
            else:
                st.error("Failed to record your vote. Please try again.")

        # Additional Visualization: Distribution of Ideas by Topic
        topics = [idea.get("topic", "Unknown") for idea in approved_ideas]
        if topics:
            df_topics = pd.DataFrame(topics, columns=["Topic"])
            topic_counts = df_topics["Topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            st.plotly_chart(
                px.pie(topic_counts, names="Topic", values="Count", title="Distribution of Ideas by Topic"),
                use_container_width=True
            )
    else:
        st.info("No approved ideas available at the moment. :hourglass:")

# -----------------------------------------------------------------------------
# TAB 4: Projects
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Municipal Projects")
    
    # Load projects from Qdrant
    projects = load_qdrant_documents("municipal_projects", vector_dim=384)
    if projects:
        df_projects = pd.DataFrame(projects)
        st.dataframe(df_projects)
        
        # -----------------------------------------------------------------------------
        # Filtering Options
        # -----------------------------------------------------------------------------
        st.markdown("#### Filter Projects")
        # Filter by status
        if "status" in df_projects.columns:
            statuses = sorted(df_projects["status"].dropna().unique().tolist())
            selected_status = st.multiselect("Status", options=statuses, default=statuses, key="status_multiselect")

        else:
            selected_status = None

        # Filter by CT / City
        if "CT" in df_projects.columns:
            CTs = sorted(df_projects["CT"].dropna().unique().tolist())
            selected_CT = st.multiselect("CT / City", options=CTs, default=CTs,key="ct_multiselect")
        else:
            selected_CT = None

        # Filter by province
        if "province" in df_projects.columns:
            provinces = sorted(df_projects["province"].dropna().unique().tolist())
            selected_province = st.multiselect("Province", options=provinces, default=provinces,key="pr_multiselect")
        else:
            selected_province = None

        # Filter by commune
        if "commune" in df_projects.columns:
            communes = sorted(df_projects["commune"].dropna().unique().tolist())
            selected_commune = st.multiselect("Commune", options=communes, default=communes,key="cm_multiselect")
        else:
            selected_commune = None

        # Filter by budget range (if budget column exists)
        if "budget" in df_projects.columns:
            min_budget = int(df_projects["budget"].min())
            max_budget = int(df_projects["budget"].max())
            budget_range = st.slider("Budget Range", min_budget, max_budget, (min_budget, max_budget))
        else:
            budget_range = None

        # Filter by completion percentage (if available)
        if "completion_percentage" in df_projects.columns:
            min_completion = float(df_projects["completion_percentage"].min())
            max_completion = float(df_projects["completion_percentage"].max())
            completion_range = st.slider("Completion % Range", min_completion, max_completion, (min_completion, max_completion))
        else:
            completion_range = None

        # Apply filters
        df_filtered = df_projects.copy()
        if selected_status:
            df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
        if selected_CT:
            df_filtered = df_filtered[df_filtered["CT"].isin(selected_CT)]
        if selected_province:
            df_filtered = df_filtered[df_filtered["province"].isin(selected_province)]
        if selected_commune:
            df_filtered = df_filtered[df_filtered["commune"].isin(selected_commune)]
        if budget_range:
            df_filtered = df_filtered[(df_filtered["budget"] >= budget_range[0]) & (df_filtered["budget"] <= budget_range[1])]
        if completion_range and "completion_percentage" in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered["completion_percentage"] >= completion_range[0]) & 
                                      (df_filtered["completion_percentage"] <= completion_range[1])]
        st.dataframe(df_filtered)
        
        if st.button("Export Filtered Projects as CSV", key="export_projects"):
            csv = df_filtered.to_csv(index=False)
            st.download_button("Download CSV", csv, "filtered_projects.csv", "text/csv")
    else:
        st.info("No projects available. :warning:")

    st.markdown("---")
    st.markdown("#### Add Comment to a Project")
    st.info("Select a project from the list or manually enter a project ID, then add your comment. Your comment will first be checked for offensive content and, if acceptable, automatically enriched with additional details (sentiment, polarity, keywords) using GPT.")

    # Allow the user to select project by list or manually enter a project ID
    project_selection_option = st.radio("Select project by:", ["Select from list", "Enter Project ID"], index=0)
    selected_project_id = ""
    if project_selection_option == "Select from list":
        if not df_projects.empty:
            project_options = df_projects[["project_id", "title"]].drop_duplicates()
            project_options["option"] = project_options["title"] + " (ID: " + project_options["project_id"] + ")"
            selected_option = st.selectbox("Select Project", options=project_options["option"].tolist())
            selected_project_id = selected_option.split("ID:")[-1].replace(")", "").strip()
        else:
            st.info("No projects available in the list. Please enter Project ID manually.")
            selected_project_id = st.text_input("Enter Project ID")
    else:
        selected_project_id = st.text_input("Enter Project ID")
    
    comment_text = st.text_area("Your Comment", placeholder="Type your comment here...")

    if st.button("Add Comment"):
        if not comment_text.strip():
            st.error("Comment cannot be empty.")
        elif not selected_project_id.strip():
            st.error("Please select or enter a valid project ID.")
        else:
            # Offensive Content Detection using GPT
            offensive_prompt = f"Does the following comment contain any offensive or inappropriate language? Answer only 'Yes' or 'No'.\n\nComment: {comment_text}"
            try:
                offensive_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that detects offensive content."},
                        {"role": "user", "content": offensive_prompt}
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                offensive_answer = offensive_response["choices"][0]["message"]["content"].strip().lower()
            except Exception as e:
                st.error(f"Error during offensive content detection: {e}")
                offensive_answer = "error"
            
            if offensive_answer == "yes":
                st.error("Your comment contains offensive language and cannot be submitted.")
            elif offensive_answer == "error":
                st.error("Could not perform offensive content detection. Please try again later.")
            else:
                # Retrieve project details if available
                proj_details = get_project_details(selected_project_id.strip())
                if proj_details:
                    project_title = proj_details.get("title", "")
                    project_themes = proj_details.get("themes", "")
                    project_CT = proj_details.get("CT", "")
                    project_province = proj_details.get("province", "")
                    project_commune = proj_details.get("commune", "")
                    project_status = proj_details.get("status", "")
                else:
                    project_title = project_themes = project_CT = project_province = project_commune = project_status = ""
                
                # GPT Extraction: Automatically extract sentiment, polarity, and keywords from the comment.
                extraction_prompt = (
                    f"Extract the following details from the citizen comment:\n\n"
                    f"Comment Text: {comment_text}\n\n"
                    "Return a JSON object with the following keys:\n"
                    "- sentiment (e.g., POS, NEG, NEU)\n"
                    "- polarity (a float value, e.g., 0.85 or -0.5)\n"
                    "- keywords (an array of key terms)\n"
                    "Only return the JSON object."
                )
                try:
                    extraction_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an NLP extraction assistant."},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.5,
                    )
                    extraction_content = extraction_response["choices"][0]["message"]["content"].strip()
                    extracted_data = json.loads(extraction_content)
                except Exception as e:
                    st.error(f"Error extracting comment details: {e}")
                    extracted_data = {"sentiment": "", "polarity": 0.0, "keywords": []}
                
                # Construct the full comment payload
                comment_payload = {
                    "citizen_name": st.session_state["username"],
                    "comment_text": comment_text,
                    "project_id": selected_project_id.strip(),
                    "date_submitted": datetime.utcnow().strftime("%Y-%m-%d"),
                    "status": "pending",  # Initially pending moderation
                    "sentiment": extracted_data.get("sentiment", ""),
                    "polarity": extracted_data.get("polarity", 0.0),
                    "keywords": extracted_data.get("keywords", []),
                    "project_title": project_title,
                    "project_themes": project_themes,
                    "project_CT": project_CT,
                    "project_province": project_province,
                    "project_commune": project_commune,
                    "project_status": project_status,
                    "votes": {"thumb_up": 0, "thumb_down": 0, "vote_score": 0}
                }
                
                # Add the comment to the citizen_comments collection in Qdrant
                new_idea_id = add_citizen_idea(comment_payload)
                st.success(f"Comment added successfully with ID: {new_idea_id}")



# -----------------------------------------------------------------------------
# TAB 5: News
# -----------------------------------------------------------------------------

with tabs[5]:
    st.header("News")
    st.write(
        "This section finds news comments (hespress_politics_comments) that are similar to citizen comments "
        "by using Qdrant cosine similarity. For each news comment, the most similar citizen comment is retrieved, "
        "and its project_id is displayed alongside the news comment. Additionally, GPT (with caching) is used "
        "to perform topics modeling and sentiment analysis on the news comment."
    )
    
    # Load news comments from Qdrant using vector dimension 384
    news_points = load_qdrant_points("hespress_politics_comments", vector_dim=384)
    if not news_points:
        st.info("No news comments found.")
    else:
        # Build DataFrame from news points (include payload fields and vector)
        df_news = pd.DataFrame([dict(**pt["payload"], vector=pt["vector"]) for pt in news_points])
        
        # ---------------------------
        # Filtering Options for News Comments
        # ---------------------------
        st.markdown("#### Filter News Comments")
        
        # Filter by Date Range (assumes 'date_added' in YYYY-MM-DD format)
        if "date_added" in df_news.columns:
            date_range = st.date_input("Date Range", [], key="news_date_range")
        else:
            date_range = []
        
        # Filter by Keyword in Comment Text
        keyword_filter = st.text_input("Keyword in Comment Text", "", key="news_keyword")
        
        # Apply filters to the news DataFrame
        df_filtered_news = df_news.copy()
        if date_range and isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered_news = df_filtered_news[df_filtered_news["date_added"].between(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))]
        if keyword_filter:
            df_filtered_news = df_filtered_news[df_filtered_news["comment"].str.contains(keyword_filter, case=False, na=False)]
        
        st.markdown("#### Filtered News Comments")
        st.dataframe(df_filtered_news)
        
        # ---------------------------
        # Similarity & GPT Analysis
        # ---------------------------
        if st.button("Analyze Similarity with Citizen Comments"):
            if df_filtered_news.empty:
                st.warning("No news comments to analyze. Adjust the filters.")
            else:
                # Load citizen comments points (including vectors)
                citizen_points = load_qdrant_points("citizen_comments", vector_dim=384)
                if not citizen_points:
                    st.warning("No citizen comments found.")
                else:
                    analysis_results = []
                    # Process each filtered news comment
                    for idx, row in df_filtered_news.iterrows():
                        news_comment = row.get("comment", "")
                        news_vector = row.get("vector")
                        
                        # Use Qdrant cosine similarity to find the most similar citizen comment
                        if news_vector:
                            results = search_qdrant("citizen_comments", news_vector, top=1)
                            if results:
                                best_match = results[0].payload
                                citizen_project_id = best_match.get("project_id", "N/A")
                            else:
                                citizen_project_id = "No match"
                        else:
                            citizen_project_id = "No vector"
                        
                        # Use GPT to analyze the news comment (topics and sentiment)
                        cache_key = f"news_analysis_{hashlib.md5(news_comment.encode('utf-8')).hexdigest()}"
                        cached_analysis = get_cached_summary(cache_key)
                        if cached_analysis:
                            try:
                                analysis = json.loads(cached_analysis.get("summary"))
                            except Exception as e:
                                st.error(f"Error parsing cached summary for news comment {idx+1}: {e}")
                                analysis = {"topics": [], "sentiment": "NEU"}
                        else:
                            try:
                                analysis_prompt = (
                                    "Analyze the following news comment and provide a JSON object with exactly two keys:\n"
                                    " - topics: an array of main topics mentioned in the comment (array of strings)\n"
                                    " - sentiment: overall sentiment (one of 'POS', 'NEG', or 'NEU')\n\n"
                                    f"News Comment: {news_comment}\n\n"
                                    "Return only a valid JSON object with no additional text."
                                )
                                with st.spinner(f"Analyzing news comment {idx+1}/{len(df_filtered_news)}..."):
                                    analysis_response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are an assistant that performs topics modeling and sentiment analysis on news comments."},
                                            {"role": "user", "content": analysis_prompt}
                                        ],
                                        max_tokens=300,
                                        temperature=0.5,
                                    )
                                analysis_output = analysis_response["choices"][0]["message"]["content"].strip()
                                st.write(f"Raw GPT output for news comment {idx+1}: {analysis_output}")  # Debug output
                                try:
                                    analysis = json.loads(analysis_output)
                                except Exception as json_error:
                                    st.error(f"JSON parsing error for news comment {idx+1}: {json_error}\nRaw output: {analysis_output}")
                                    analysis = {"topics": [], "sentiment": "NEU"}
                                store_cached_summary(cache_key, json.dumps(analysis))
                            except Exception as e:
                                st.error(f"Error analyzing news comment {idx+1}: {e}")
                                analysis = {"topics": [], "sentiment": "NEU"}
                        
                        analysis_results.append({
                            "hespress_comment": news_comment,
                            "citizen_project_id": citizen_project_id,
                            "topics": analysis.get("topics", []),
                            "sentiment": analysis.get("sentiment", "NEU")
                        })
                    
                    df_results = pd.DataFrame(analysis_results)
                    st.markdown("#### Similarity Analysis Results")
                    st.dataframe(df_results)
                    
                    # ------------- Additional Features -------------
                    # Sentiment Distribution Chart
                    if not df_results.empty:
                        sentiment_counts = df_results["sentiment"].value_counts().reset_index()
                        sentiment_counts.columns = ["Sentiment", "Count"]
                        st.markdown("#### Sentiment Distribution")
                        st.plotly_chart(
                            px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution"),
                            use_container_width=True
                        )
                        
                        # Topics Frequency Chart: Flatten all topics and count frequency
                        all_topics = []
                        for topics in df_results["topics"]:
                            all_topics.extend(topics)
                        if all_topics:
                            topics_series = pd.Series(all_topics)
                            topics_counts = topics_series.value_counts().reset_index()
                            topics_counts.columns = ["Topic", "Count"]
                            st.markdown("#### Topics Frequency")
                            st.plotly_chart(
                                px.bar(topics_counts, x="Topic", y="Count", title="Topics Frequency"),
                                use_container_width=True
                            )
                        
                        # Sorting Options for Analysis Results
                        sort_option = st.selectbox("Sort Analysis Results by:", options=["Default", "Sentiment", "Number of Topics"], key="sort_option")
                        if sort_option == "Sentiment":
                            order = {"POS": 0, "NEU": 1, "NEG": 2}
                            df_results = df_results.sort_values(by="sentiment", key=lambda x: x.map(order))
                        elif sort_option == "Number of Topics":
                            df_results = df_results.sort_values(by="topics", key=lambda x: x.apply(len), ascending=False)
                        st.markdown("#### Sorted Analysis Results")
                        st.dataframe(df_results)
                        
                        # Download Button for Analysis Results
                        csv_data = df_results.to_csv(index=False)
                        st.download_button("Download Analysis Results as CSV", data=csv_data, file_name="news_analysis_results.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# TAB 6: Profile
# -----------------------------------------------------------------------------
with tabs[6]:
    st.header("Your Profile")
    st.write(f"**Username:** {st.session_state['username']} :bust_in_silhouette:")
    st.write(f"**Role:** {st.session_state['role']} :lock:")
    st.write("Manage your personal settings below.")

    # Only citizens are allowed to change their own password.
    if st.session_state['role'] == "citizen":
        st.markdown("### Change Your Password :key:")
        with st.form("change_password_form", clear_on_submit=True):
            old_password = st.text_input("Old Password", type="password", help="Enter your current password")
            new_password = st.text_input("New Password", type="password", help="Enter your new password")
            confirm_password = st.text_input("Confirm New Password", type="password", help="Re-enter your new password")
            change_btn = st.form_submit_button("Change Password")
        if change_btn:
            if new_password != confirm_password:
                st.error("New password and confirmation do not match.")
            else:
                valid, _ = verify_user(st.session_state["username"], old_password)
                if not valid:
                    st.error("Old password is incorrect.")
                else:
                    if update_user_password(st.session_state["username"], new_password):
                        st.success("Password updated successfully!")
                    else:
                        st.error("Failed to update password. Please try again.")

    # (Optionally, for admins additional profile features may be added in another section.)
    st.markdown("### Personal Statistics :bar_chart:")
    # Here you could add additional statistics related to the citizen's ideas, etc.
    # For example, display a small table of recent approved ideas or similar metrics.
    # (This part can be extended as needed.)

    st.markdown("---")
    st.info("Only password changes are permitted for citizens. Other administrative rights are managed separately.")

# -----------------------------------------------------------------------------
# TAB 7: Chat with Assistant
# -----------------------------------------------------------------------------
with tabs[7]:
    st.header("Chat with Assistant")
    st.write("Ask your questions and get personalized answers from our assistant. :speech_balloon:")

    # Display chat history for the logged-in citizen
    st.markdown("### Your Chat History")
    history = get_chat_history(st.session_state["username"], limit=10)
    if history:
        for record in history:
            ts = record.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**{ts}**")
            st.markdown(f"> **Q:** {record.get('prompt')}")
            st.markdown(f"> **A:** {record.get('answer')}")
            st.markdown("---")
    else:
        st.info("No chat history found. :open_file_folder:")

    # Input for new query
    with st.container():
        citizen_query = st.text_input("Your Query:", key="citizen_query", help="Type your question here...")

    # Process the new query when the user clicks "Get Answer"
    if st.button("Get Answer"):
        if not citizen_query.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                # Build context from all Qdrant collections
                context_docs = []
                for coll in qdrant_collections:
                    # Set query dimension based on the collection (1536 for hespress_politics_details, 384 otherwise)
                    q_dim = 1536 if coll == "hespress_politics_details" else 384
                    # Generate a simulated query embedding (replace with actual embedding model if available)
                    query_vector = np.random.rand(q_dim).tolist()
                    results = search_qdrant(coll, query_vector, top=2)
                    for point in results:
                        payload = point.payload
                        text = payload.get("comment_text") or payload.get("description") or ""
                        if text:
                            context_docs.append(f"[{coll}] {text}")
                
                # Join all context pieces into a single context block
                context_block = "\n\n".join(context_docs)
                st.markdown(
                    f"<div class='context-bubble'><strong>Retrieved Context:</strong><br>{context_block}</div>",
                    unsafe_allow_html=True
                )
                
                # Build the final prompt for GPT
                prompt = (
                    f"You are a helpful assistant. Use the following context to answer the user query:\n\n"
                    f"Context:\n{context_block}\n\nUser Query: {citizen_query}\n\nAnswer:"
                )
                
                # Check if this prompt has already been answered and cached
                cached = get_cached_answer(prompt)
                if cached:
                    answer = cached.get("answer")
                    st.markdown(
                        f"<div class='answer-bubble'><strong>Cached Chatbot Answer:</strong><br>{answer}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # Use selected LLM with cost optimization settings
                    model_name = llm_options[selected_llm]
                    temperature = 0.3 if cost_opt else 0.7
                    try:
                        response = openai.ChatCompletion.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a knowledgeable assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        answer = response["choices"][0]["message"]["content"].strip()
                        st.markdown(
                            f"<div class='answer-bubble'><strong>Chatbot Answer:</strong><br>{answer}</div>",
                            unsafe_allow_html=True
                        )
                        # Cache the new answer
                        store_cached_answer(prompt, answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        answer = "Sorry, an error occurred while generating the answer."
                
                # Save the new chat history for the citizen
                store_chat_history(st.session_state["username"], prompt, answer)


import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from pymongo import MongoClient
import hashlib

# ----------------------------------------------------------------
# Sidebar Settings: LLM Selection, Token Limit, Cost Optimization, and Chat History Toggle
# ----------------------------------------------------------------
if not st.session_state.get("username") or not st.session_state.get("role"):
    st.error("Access Denied. Please log in to use the chatbot.")
    st.stop()

st.sidebar.header("LLM & Cost Settings")
llm_options = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "O1 (Custom Model)": "o1"
}

selected_llm = st.sidebar.selectbox("Select LLM", list(llm_options.keys()))
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=150, step=50)
cost_opt = st.sidebar.checkbox("Enable Cost Optimization", value=True)
show_history = st.sidebar.checkbox("Show My Chat History", value=True)

# Extra sidebar features for the chatbot
st.sidebar.markdown("---")
st.sidebar.subheader("Chatbot Extras")
if st.sidebar.button("Reset Conversation"):
    # Clear the current query (could also clear session conversation state if you maintain one)
    if "user_query" in st.session_state:
        del st.session_state["user_query"]
    st.sidebar.success("Conversation reset!")
if st.sidebar.button("Export Chat History"):
    # Retrieve chat history and allow download as CSV
    history = []
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        history = list(db["chat_history"].find({"username": st.session_state.get("username")}).sort("timestamp", -1))
    except Exception as e:
        st.sidebar.error(f"Error exporting chat history: {e}")
    finally:
        client.close()
    if history:
        df_history = pd.DataFrame(history)
        st.sidebar.download_button("Download History CSV", df_history.to_csv(index=False), "chat_history.csv", "text/csv")
if st.sidebar.button("Rate Last Answer"):
    st.sidebar.info("Feature coming soon!")  # Placeholder for rating functionality
if st.sidebar.button("Provide Feedback"):
    st.sidebar.info("Feedback form coming soon!")  # Placeholder for feedback functionality
if st.sidebar.button("View Conversation Summary"):
    st.sidebar.info("Conversation summary feature coming soon!")  # Placeholder

st.sidebar.markdown("---")
st.sidebar.write("Adjust parameters to balance cost and performance.")

# Set OpenAI API key
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
if st.session_state["openai_api_key"]:
    openai.api_key = st.session_state["openai_api_key"]

# ----------------------------------------------------------------
# Custom CSS for a Modern Dark Chatbot Style
# ----------------------------------------------------------------
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
# Qdrant Helper Functions
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Qdrant Collections to Use
# ----------------------------------------------------------------
qdrant_collections = [
    "citizen_comments",
    "citizen_ideas",
    "hespress_politics_comments",
    "hespress_politics_details",  # expects 1536-dim vectors
    "municipal_projects",
    "remacto_comments",
    "remacto_projects"
]

# ----------------------------------------------------------------
# MongoDB Caching Functions (for chatbot cache)
# ----------------------------------------------------------------
def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_answer(prompt: str) -> dict:
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        cache_doc = db["chatbot_cache"].find_one({"prompt_hash": compute_hash(prompt)})
        return cache_doc
    except Exception as e:
        st.error(f"Error retrieving cached answer: {e}")
        return None
    finally:
        client.close()

def store_cached_answer(prompt: str, answer: str):
    try:
        client = MongoClient("mongodb://localhost:27017")
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

# ----------------------------------------------------------------
# MongoDB Chat History Functions
# ----------------------------------------------------------------
def store_chat_history(username: str, prompt: str, answer: str):
    try:
        client = MongoClient("mongodb://localhost:27017")
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
        client = MongoClient("mongodb://localhost:27017")
        db = client["CivicCatalyst"]
        docs = list(db["chat_history"].find({"username": username}).sort("timestamp", -1).limit(limit))
        return docs
    except Exception as e:
        st.error(f"Error retrieving chat history: {e}")
        return []
    finally:
        client.close()

# ----------------------------------------------------------------
# Login Mechanism (Same as in Login.py)
# ----------------------------------------------------------------
if not st.session_state.get("username") or not st.session_state.get("role"):
    st.error("Access Denied. Please log in to use the chatbot.")
    st.markdown("### Login")
    with st.form("login_form", clear_on_submit=True):
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_submitted = st.form_submit_button("Login")
    if login_submitted:
        try:
            client = MongoClient("mongodb://localhost:27017")
            db = client["CivicCatalyst"]
            users_collection = db["users"]
            user = users_collection.find_one({"username": login_username})
            if user and user["password_hash"] == hashlib.sha256(login_password.encode()).hexdigest():
                st.session_state["username"] = login_username
                st.session_state["role"] = user.get("role", "user")
                st.success(f"Login successful! Welcome {login_username}.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        except Exception as e:
            st.error(f"Error during login: {e}")
        finally:
            client.close()
    st.stop()

# ----------------------------------------------------------------
# RAG Chatbot Functionality
# ----------------------------------------------------------------
st.markdown("<div class='chat-title'>CivicCatalyst Chatbot</div>", unsafe_allow_html=True)

# Sidebar: Show chat history for logged-in user
if show_history:
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
        st.info("No chat history found.")

st.write("Ask your question and get answers augmented with context from all Qdrant collections.")

with st.container():
    user_query = st.text_input("Your Query:", key="user_query", help="Type your question here...")

if st.button("Get Answer"):
    if not user_query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            context_docs = []
            for coll in qdrant_collections:
                q_dim = 1536 if coll == "hespress_politics_details" else 384
                query_vector = np.random.rand(q_dim).tolist()  # Simulated embedding; replace with your actual model's embedding
                results = search_qdrant(coll, query_vector, top=2)
                for point in results:
                    payload = point.payload
                    text = payload.get("comment_text") or payload.get("description") or ""
                    if text:
                        context_docs.append(f"[{coll}] {text}")
            
            context_block = "\n\n".join(context_docs)
            st.markdown("<div class='context-bubble'><strong>Retrieved Context:</strong><br>" + context_block + "</div>", unsafe_allow_html=True)
            
            prompt = f"You are a helpful assistant. Use the following context to answer the user query:\n\nContext:\n{context_block}\n\nUser Query: {user_query}\n\nAnswer:"
            
            # Check cache
            cached = get_cached_answer(prompt)
            if cached:
                answer = cached.get("answer")
                st.markdown("<div class='answer-bubble'><strong>Cached Chatbot Answer:</strong><br>" + answer + "</div>", unsafe_allow_html=True)
            else:
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
                    st.markdown("<div class='answer-bubble'><strong>Chatbot Answer:</strong><br>" + answer + "</div>", unsafe_allow_html=True)
                    store_cached_answer(prompt, answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
            
            # Store chat history for the current user
            store_chat_history(st.session_state["username"], prompt, answer)

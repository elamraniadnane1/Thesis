import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SearchRequest
from pymongo import MongoClient
import hashlib
import random
import logging
import tiktoken
import plotly.express as px
import re
from typing import List, Dict, Any, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CivicCatalyst-Chatbot")

# ----------------------------------------------------------------
# Constants and Configuration
# ----------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODELS = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "GPT-4 Turbo": "gpt-4-turbo",
    "Claude 3 Opus": "claude-3-opus-20240229",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Llama 3 70B": "meta/llama-3-70b",
    "O1 (Custom Model)": "o1"
}

DEFAULT_EMBEDDING_DIM = 384
SPECIAL_DIMS = {"hespress_politics_details": 1536}
MAX_CONTEXT_TOKENS = 4000
TYPICAL_TOKEN_TO_CHARS = 4  # Average ratio of tokens to characters

# ----------------------------------------------------------------
# Initialization and Session State Management
# ----------------------------------------------------------------
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = generate_conversation_id()
    
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = {}
    
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    
    if "token_counts" not in st.session_state:
        st.session_state.token_counts = {"prompt": 0, "completion": 0, "total": 0}
    
    if "cost_estimate" not in st.session_state:
        st.session_state.cost_estimate = 0.0
    
    # Default settings
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 500
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"

def generate_conversation_id():
    """Generate a unique conversation ID"""
    return f"conv-{int(time.time())}-{random.randint(1000, 9999)}"

def reset_conversation():
    """Reset the current conversation while preserving settings"""
    st.session_state.current_conversation = []
    st.session_state.conversation_id = generate_conversation_id()
    st.session_state.last_contexts = {}
    st.session_state.suggested_questions = []
    st.session_state.token_counts = {"prompt": 0, "completion": 0, "total": 0}
    st.session_state.cost_estimate = 0.0

# ----------------------------------------------------------------
# Database Connections and Helper Functions
# ----------------------------------------------------------------
def get_mongodb_client():
    """Get a MongoDB client connection"""
    try:
        client = MongoClient("mongodb://ac-aurbbb0-shard-00-01.mvvbpez.mongodb.net:27017")
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None

def get_qdrant_client():
    """Get a Qdrant client connection"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        # Test connection by getting collections list
        client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        return None

def fetch_qdrant_documents(collection_name: str, vector_dim: int, limit: int = 100):
    """Fetch documents from Qdrant with error handling and pagination support"""
    try:
        client = get_qdrant_client()
        if not client:
            return []
            
        all_docs = []
        
        try:
            # Basic scroll parameters without filter
            scroll_params = {
                "collection_name": collection_name,
                "limit": min(limit, 100)  # Fetch in batches of 100 max
            }
            
            # First scroll request
            scroll_result = client.scroll(**scroll_params)
            
            # Extract results based on the return format
            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                # Older API might return just points
                points = scroll_result
                next_offset = len(points) if len(points) > 0 else None
            
            # Extract payloads
            all_docs.extend([pt.payload for pt in points])
            
            # Continue scrolling until no more points or reached limit
            while next_offset is not None and len(all_docs) < limit:
                scroll_params["offset"] = next_offset
                
                if isinstance(scroll_result, tuple):
                    points, next_offset = client.scroll(**scroll_params)
                else:
                    points = client.scroll(**scroll_params)
                    next_offset = next_offset + len(points) if len(points) > 0 else None
                
                all_docs.extend([pt.payload for pt in points])
                
            return all_docs[:limit]
                
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            
            # Fallback to search without filter if scroll fails
            try:
                # Try basic search without filter
                search_results = client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * vector_dim,
                    limit=limit,
                    with_payload=True
                )
                
                all_docs = [result.payload for result in search_results]
                return all_docs
            except Exception as e2:
                logger.error(f"Error in search operation: {e2}")
                return []
                
    except Exception as e:
        logger.error(f"Error loading documents from Qdrant ({collection_name}): {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

def semantic_search(collection_name: str, query_vector: list, top: int = 5, filter_conditions=None):
    """Perform semantic search in Qdrant with error handling"""
    try:
        client = get_qdrant_client()
        if not client:
            return []
        
        try:
            # First try without filter
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top,
                with_payload=True
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            
            # Try simplified search as fallback
            try:
                search_results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top
                )
                return search_results
            except:
                return []
                
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

# ----------------------------------------------------------------
# Embedding and Text Processing
# ----------------------------------------------------------------
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string"""
    try:
        # Get the correct encoding for the model
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
        
        # Count tokens
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to approximate counting if tiktoken fails
        logger.error(f"Error counting tokens: {e}")
        return len(text) // TYPICAL_TOKEN_TO_CHARS

def truncate_text_to_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit"""
    tokens = count_tokens(text, model)
    if tokens <= max_tokens:
        return text
    
    # If over limit, estimate a character count and truncate
    estimated_chars = int(max_tokens * TYPICAL_TOKEN_TO_CHARS * 0.9)  # 90% to be safe
    truncated = text[:estimated_chars] + "... [truncated]"
    
    # Check if still too long and adjust further if needed
    while count_tokens(truncated, model) > max_tokens:
        estimated_chars = int(estimated_chars * 0.9)
        truncated = text[:estimated_chars] + "... [truncated]"
    
    return truncated

def create_embeddings(text: str) -> list:
    """Create embeddings for a text using OpenAI's API"""
    if not openai.api_key:
        logger.error("OpenAI API key not set")
        # Return random vector of correct dimension as fallback
        return np.random.rand(1536).tolist()
    
    try:
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        # Return random vector of correct dimension as fallback
        return np.random.rand(1536).tolist()

def extract_key_text_from_payload(payload: dict, collection_name: str) -> str:
    """Extract the most relevant text content from a payload based on collection type"""
    if not payload:
        return ""
    
    # Different collections store text in different fields
    if collection_name == "citizen_comments":
        return payload.get("content", "")
    elif collection_name == "citizen_ideas":
        challenge = payload.get("challenge", "")
        solution = payload.get("solution", "")
        return f"Challenge: {challenge}\nSolution: {solution}"
    elif collection_name == "hespress_politics_comments":
        return payload.get("comment_text", "")
    elif collection_name == "hespress_politics_details":
        title = payload.get("title", "")
        content = payload.get("content", "")
        return f"Title: {title}\n\n{content}"
    elif collection_name == "municipal_projects":
        title = payload.get("title", "")
        description = payload.get("description", "")
        status = payload.get("status", "")
        budget = payload.get("budget", "")
        return f"Project: {title}\nStatus: {status}\nBudget: {budget}\nDescription: {description}"
    elif collection_name == "remacto_comments":
        return payload.get("comment_text", "")
    elif collection_name == "remacto_projects":
        title = payload.get("title", "")
        description = payload.get("description", "")
        return f"Project: {title}\nDescription: {description}"
    else:
        # Try to extract any text fields
        text_fields = ["content", "text", "description", "comment_text", "title", "body", "challenge", "solution"]
        for field in text_fields:
            if field in payload and isinstance(payload[field], str):
                return payload[field]
    
    # If no text found, convert entire payload to string
    return str(payload)

# ----------------------------------------------------------------
# Chat History and Cache Functions
# ----------------------------------------------------------------
def compute_hash(text: str) -> str:
    """Create a hash of a text string"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_answer(prompt: str) -> dict:
    """Get a cached answer from the database"""
    try:
        client = get_mongodb_client()
        if not client:
            return None
        
        db = client["CivicCatalyst"]
        cache_doc = db["chatbot_cache"].find_one({"prompt_hash": compute_hash(prompt)})
        return cache_doc
    except Exception as e:
        logger.error(f"Error retrieving cached answer: {e}")
        return None
    finally:
        if 'client' in locals() and client:
            client.close()

def store_cached_answer(prompt: str, answer: str, model: str):
    """Store a cached answer in the database"""
    try:
        client = get_mongodb_client()
        if not client:
            return
        
        db = client["CivicCatalyst"]
        doc = {
            "prompt_hash": compute_hash(prompt),
            "prompt": prompt,
            "answer": answer,
            "model": model,
            "timestamp": datetime.utcnow()
        }
        db["chatbot_cache"].update_one(
            {"prompt_hash": doc["prompt_hash"]}, 
            {"$set": doc}, 
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error storing cached answer: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()

def store_chat_history(username: str, question: str, answer: str, model: str, conversation_id: str, contexts: List[str] = None):
    """Store a chat interaction in the history database"""
    try:
        client = get_mongodb_client()
        if not client:
            return
        
        db = client["CivicCatalyst"]
        doc = {
            "username": username,
            "question": question,
            "answer": answer,
            "model": model,
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow(),
            "contexts": contexts or []
        }
        db["chat_history"].insert_one(doc)
    except Exception as e:
        logger.error(f"Error storing chat history: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()

def get_chat_history(username: str, limit: int = 10):
    """Get chat history for a user"""
    try:
        client = get_mongodb_client()
        if not client:
            return []
        
        db = client["CivicCatalyst"]
        docs = list(db["chat_history"].find(
            {"username": username}
        ).sort("timestamp", -1).limit(limit))
        
        return docs
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

def get_user_conversations(username: str):
    """Get all conversation IDs for a user"""
    try:
        client = get_mongodb_client()
        if not client:
            return []
        
        db = client["CivicCatalyst"]
        pipeline = [
            {"$match": {"username": username}},
            {"$group": {"_id": "$conversation_id"}},
            {"$sort": {"_id": -1}}
        ]
        
        result = db["chat_history"].aggregate(pipeline)
        conversation_ids = [item["_id"] for item in result]
        
        return conversation_ids
    except Exception as e:
        logger.error(f"Error retrieving user conversations: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

def get_conversation(conversation_id: str):
    """Get all messages in a conversation"""
    try:
        client = get_mongodb_client()
        if not client:
            return []
        
        db = client["CivicCatalyst"]
        docs = list(db["chat_history"].find(
            {"conversation_id": conversation_id}
        ).sort("timestamp", 1))
        
        return docs
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

def rate_answer(username: str, conversation_id: str, timestamp: datetime, rating: int, feedback: str = None):
    """Store a rating for a chat answer"""
    try:
        client = get_mongodb_client()
        if not client:
            return False
        
        db = client["CivicCatalyst"]
        result = db["chat_history"].update_one(
            {
                "username": username,
                "conversation_id": conversation_id,
                "timestamp": timestamp
            },
            {
                "$set": {
                    "rating": rating,
                    "feedback": feedback,
                    "rated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error storing rating: {e}")
        return False
    finally:
        if 'client' in locals() and client:
            client.close()

# ----------------------------------------------------------------
# Prompt Construction and LLM Interaction
# ----------------------------------------------------------------
def build_system_prompt():
    """Build a system prompt for the chatbot"""
    return """You are CivicAssist, an intelligent assistant for the CivicCatalyst platform. 
Your role is to help citizens, government officials, and analysts understand public opinions, 
municipal projects, and civic engagement data.

Guidelines:
- Answer questions accurately based on the provided context
- When context is insufficient, clearly state what you don't know
- Avoid making up information not present in the context
- Stay politically neutral and objective in your responses
- Format your responses in clear, concise language
- When technical terms are used, explain them in simple terms
- Provide specific citations to the context when possible
- For data-related questions, summarize key findings clearly
- If asked about trends, note if the provided data is sufficient to establish them
- Maintain a helpful, conversational tone

Ensure your responses are accurate, factual, and based only on the context provided."""

def create_prompt_with_context(query: str, context: List[str]) -> Tuple[str, str]:
    """Create a prompt with embedded context"""
    # Format the context nicely
    formatted_context = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
    
    # Create the combined prompt
    system_prompt = build_system_prompt()
    user_prompt = f"Here is the context to help answer the question:\n\n{formatted_context}\n\nBased on the above context, please answer this question: {query}"
    
    return system_prompt, user_prompt

def get_answer_from_llm(system_prompt: str, user_prompt: str, model: str, max_tokens: int, temperature: float) -> Tuple[str, Dict[str, int]]:
    """Get an answer from the LLM with token tracking"""
    if not openai.api_key:
        return "Error: OpenAI API key not set. Please provide your API key in the sidebar.", {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        # Count tokens in the prompt
        prompt_tokens = count_tokens(system_prompt + user_prompt, model)
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Extract the answer and token counts
        answer = response["choices"][0]["message"]["content"].strip()
        completion_tokens = response["usage"]["completion_tokens"]
        total_tokens = response["usage"]["total_tokens"]
        
        return answer, {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}", {"prompt": 0, "completion": 0, "total": 0}

def generate_follow_up_questions(question: str, answer: str, model: str):
    """Generate follow-up questions based on the conversation so far"""
    if not openai.api_key:
        return []
    
    try:
        system_prompt = "You are a helpful assistant that suggests relevant follow-up questions based on a conversation."
        user_prompt = f"""Based on the following question and answer exchange, suggest 3 concise, specific follow-up questions the user might want to ask next.
        
Question: {question}

Answer: {answer}

Provide exactly 3 natural follow-up questions that explore different aspects of the topic, each on a new line. They should be directly related to the conversation above and be specific, not generic. Make each question 15 words or less."""
        
        # Call OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        
        # Extract and format suggestions
        suggestions_text = response["choices"][0]["message"]["content"].strip()
        suggested_questions = [q.strip() for q in suggestions_text.split('\n') if q.strip()]
        
        # Limit to the first 3
        return suggested_questions[:3]
    
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return []

def estimate_cost(token_counts: Dict[str, int], model: str) -> float:
    """Estimate cost based on token usage and model"""
    # Default rates per 1000 tokens (approximated)
    rates = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "o1": {"input": 0.015, "output": 0.075},  # Approximate based on comparable models
    }
    
    # Get rates for the selected model or use gpt-3.5-turbo as fallback
    model_rates = rates.get(model, rates["gpt-3.5-turbo"])
    
    # Calculate costs
    input_cost = (token_counts["prompt"] / 1000) * model_rates["input"]
    output_cost = (token_counts["completion"] / 1000) * model_rates["output"]
    
    return input_cost + output_cost

# ----------------------------------------------------------------
# Main Chatbot UI and Logic
# ----------------------------------------------------------------
def main():
    """Main function for the chatbot UI"""
   
    
    # Initialize session state
    initialize_session_state()
    
    # Check for login
    if not st.session_state.get("username") or not st.session_state.get("role"):
        st.error("Access Denied. Please log in to use the chatbot.")
        st.stop()
    
    # Apply custom CSS
    st.markdown(
        """
        <style>
        /* Main chat container */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
            padding: 10px;
        }
        
        /* User message styling */
        .user-message {
            background-color: #2b5ff6;
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 12px 18px;
            margin-left: 20%;
            margin-right: 10px;
            align-self: flex-end;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Bot message styling */
        .bot-message {
            background-color: #f1f1f1;
            color: #333;
            border-radius: 18px 18px 18px 0;
            padding: 12px 18px;
            margin-right: 20%;
            margin-left: 10px;
            align-self: flex-start;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Context styling */
        .context-box {
            background-color: #f8f9fa;
            border-left: 3px solid #4CAF50;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 5px;
            font-size: 0.9em;
            color: #555;
        }
        
        /* Chat header */
        .chat-header {
            background-color: #4285f4;
            color: white;
            padding: 15px;
            border-radius: 10px 10px 0 0;
            margin-bottom: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        /* Input area styling */
        .input-area {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            padding: 10px;
            background-color: #f1f1f1;
            border-radius: 10px;
        }
        
        /* Suggested questions */
        .suggested-question {
            background-color: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 18px;
            padding: 8px 15px;
            margin: 5px;
            display: inline-block;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .suggested-question:hover {
            background-color: #bbdefb;
        }
        
        /* Metrics box */
        .metrics-box {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            font-size: 0.9em;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(66, 133, 244, 0.1);
            border-bottom: 2px solid #4285f4;
        }
        
        /* Improved button styling */
        .stButton>button {
            border-radius: 20px;
            padding: 5px 15px;
            font-weight: 500;
            background-color: #4285f4;
            color: white;
            border: none;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #3367d6;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* ChatMessage timestamps and metadata */
        .message-meta {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }
        
        /* Rating stars */
        .rating-stars {
            display: flex;
            gap: 5px;
        }
        
        .star {
            cursor: pointer;
            font-size: 20px;
            color: #ddd;
        }
        
        .star.filled {
            color: #FFD700;
        }
        
        /* Syntax highlighting for code blocks */
        code {
            background-color: #f7f7f7;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        pre {
            background-color: #f7f7f7;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # --------- Sidebar UI ---------
    with st.sidebar:
        st.header("⚙️ Chatbot Settings")
        
        # Model selection
        selected_model_name = st.selectbox("Select Model", list(CHAT_MODELS.keys()))
        st.session_state.model = CHAT_MODELS[selected_model_name]
        
        # Parameters
        st.session_state.max_tokens = st.slider("Max Response Length", min_value=50, max_value=2000, value=st.session_state.max_tokens, step=50)
        st.session_state.temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1)
        
        # Cost optimization
        cost_opt = st.checkbox("Enable Cost Optimization", value=True)
        
        # API Key input
        with st.expander("API Configuration"):
            if "openai_api_key" not in st.session_state:
                st.session_state["openai_api_key"] = ""
            
            api_key = st.text_input("OpenAI API Key", value=st.session_state["openai_api_key"], type="password", placeholder="sk-...")
            
            if api_key:
                st.session_state["openai_api_key"] = api_key
                openai.api_key = api_key
            
            st.caption("Your API key is stored temporarily in the session and not saved permanently.")
        
        # Statistics
        # Statistics continued from sidebar expander
            st.markdown(
                f"""
                <div class="metrics-box">
                    <h4>Current Session</h4>
                    <ul>
                        <li>Tokens Used: {st.session_state.token_counts['total']}</li>
                        <li>Estimated Cost: ${st.session_state.cost_estimate:.4f}</li>
                        <li>Current Model: {st.session_state.model}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Conversation management
        st.markdown("---")
        st.markdown("### 💬 Conversation")
        
        if st.button("🔄 New Conversation", key="reset_conversation"):
            reset_conversation()
            st.success("Started a new conversation!")
            st.rerun()
        
        # Chat history control
        show_history = st.checkbox("Show Chat History", value=True)
        
        # Collection selection
        st.markdown("### 🗃️ Data Sources")
        
        # Available Qdrant collections
        qdrant_collections = [
            "citizen_comments",
            "citizen_ideas",
            "hespress_politics_comments",
            "hespress_politics_details",
            "municipal_projects",
            "remacto_comments",
            "remacto_projects"
        ]
        
        # Allow selecting which collections to search
        selected_collections = st.multiselect(
            "Select Data Sources",
            qdrant_collections,
            default=["citizen_comments", "citizen_ideas", "municipal_projects"]
        )
        
        if not selected_collections:
            st.warning("Please select at least one data source.")
        
        # Export options
        st.markdown("### 📊 Export Options")
        
        if st.button("📥 Export Chat History"):
            history = get_chat_history(st.session_state.get("username"), limit=100)
            if history:
                # Convert to pandas DataFrame
                history_data = []
                for item in history:
                    history_data.append({
                        "timestamp": item.get("timestamp"),
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "model": item.get("model", "unknown"),
                        "conversation_id": item.get("conversation_id", "unknown")
                    })
                
                df_history = pd.DataFrame(history_data)
                
                # Convert to CSV for download
                csv = df_history.to_csv(index=False)
                
                st.download_button(
                    "Download History CSV",
                    csv,
                    "chat_history.csv",
                    "text/csv"
                )
            else:
                st.info("No chat history to export.")
    
    # --------- Main Chat UI ---------
    st.markdown('<div class="chat-header">CivicCatalyst AI Assistant</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Chat", "History", "Collections"])
    
    # Chat Tab
    with tabs[0]:
        # Display current conversation
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.current_conversation):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:  # assistant message
                st.markdown(
                    f'<div class="bot-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Show metadata after bot messages
                if "timestamp" in message:
                    formatted_time = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(message["timestamp"], datetime) else str(message["timestamp"])
                    st.markdown(
                        f'<div class="message-meta">Generated at {formatted_time} using {message.get("model", "unknown")}</div>',
                        unsafe_allow_html=True
                    )
                
                # Show rating option for bot messages
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        if "timestamp" in message and "conversation_id" in message:
                            rate_answer(
                                st.session_state.username,
                                message["conversation_id"],
                                message["timestamp"],
                                5,
                                "Liked by user"
                            )
                            st.success("Thanks for the feedback!")
                
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        if "timestamp" in message and "conversation_id" in message:
                            rate_answer(
                                st.session_state.username,
                                message["conversation_id"],
                                message["timestamp"],
                                1,
                                "Disliked by user"
                            )
                            feedback = st.text_input("What could be improved?", key=f"feedback_{i}")
                            if feedback:
                                rate_answer(
                                    st.session_state.username,
                                    message["conversation_id"],
                                    message["timestamp"],
                                    1,
                                    feedback
                                )
                                st.success("Thanks for your feedback!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show suggested follow-up questions if available
        if st.session_state.suggested_questions:
            st.markdown("#### Suggested Follow-up Questions")
            cols = st.columns(len(st.session_state.suggested_questions))
            
            for i, question in enumerate(st.session_state.suggested_questions):
                with cols[i]:
                    if st.button(question, key=f"suggested_{i}"):
                        st.session_state.user_query = question
                        st.rerun()
        # --------- Input area ---------
        # Ensure the session state has a default for user_query
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        # Input area
        user_query = st.text_input("Your question:", value=st.session_state.get("query_value", ""), key="user_query_updated")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_btn = st.button("🚀 Send", key="send_btn")
        with col2:
            clear_btn = st.button("🧹 Clear Input", key="clear_btn")
            
        if clear_btn:
            st.session_state.user_query = ""
            st.rerun()
        
        # Process user query
        if (send_btn or (user_query and user_query != st.session_state.get("last_query", ""))) and user_query.strip():
            st.session_state.last_query = user_query
            if user_query.strip():
                st.session_state.last_query = user_query
                
                # Add user message to conversation
                st.session_state.current_conversation.append({
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.now()
                })
                
                with st.spinner("Thinking..."):
                    # Generate vector embedding for the query
                    query_embedding = create_embeddings(user_query)
                    
                    # Retrieve relevant context from Qdrant collections
                    context_docs = []
                    
                    if not selected_collections:
                        selected_collections = ["citizen_comments", "municipal_projects"]
                    
                    # Search across selected collections
                    for collection_name in selected_collections:
                        # Get the right vector dimension for this collection
                        vector_dim = SPECIAL_DIMS.get(collection_name, DEFAULT_EMBEDDING_DIM)
                        
                        try:
                            # Search in this collection
                            results = semantic_search(collection_name, query_embedding, top=3)
                            
                            # Extract relevant text from each result
                            for result in results:
                                payload = result.payload
                                text = extract_key_text_from_payload(payload, collection_name)
                                if text:
                                    source_prefix = f"[{collection_name}]"
                                    context_docs.append(f"{source_prefix} {text}")
                        except Exception as e:
                            logger.error(f"Error searching {collection_name}: {e}")
                    
                    # If no context found, provide a message
                    if not context_docs:
                        context_docs = ["No relevant information found in the database."]
                    
                    # Save contexts for later reference
                    st.session_state.last_contexts[user_query] = context_docs
                    
                    # Create the prompt with context
                    system_prompt, user_prompt = create_prompt_with_context(user_query, context_docs)
                    
                    # Check cache
                    cached = get_cached_answer(user_prompt)
                    
                    if cached and cached.get("answer") and cost_opt:
                        # Use cached answer
                        answer = cached.get("answer")
                        token_counts = {"prompt": 0, "completion": 0, "total": 0}  # Unknown for cached
                        
                        # Show that we used a cached response
                        answer = f"{answer}\n\n[Cached response]"
                    else:
                        # Get answer from LLM
                        answer, token_counts = get_answer_from_llm(
                            system_prompt,
                            user_prompt,
                            st.session_state.model,
                            st.session_state.max_tokens,
                            st.session_state.temperature
                        )
                        
                        # Cache the answer for future use
                        if answer and not answer.startswith("Error:"):
                            store_cached_answer(user_prompt, answer, st.session_state.model)
                    
                    # Update token counts and cost estimate
                    st.session_state.token_counts["prompt"] += token_counts["prompt"]
                    st.session_state.token_counts["completion"] += token_counts["completion"]
                    st.session_state.token_counts["total"] += token_counts["total"]
                    
                    # Calculate cost
                    this_cost = estimate_cost(token_counts, st.session_state.model)
                    st.session_state.cost_estimate += this_cost
                    
                    # Save answer to conversation and history
                    timestamp = datetime.now()
                    
                    # Add to the current conversation
                    st.session_state.current_conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": timestamp,
                        "model": st.session_state.model,
                        "conversation_id": st.session_state.conversation_id,
                        "token_counts": token_counts,
                        "cost": this_cost
                    })
                    
                    # Store in persistent chat history
                    store_chat_history(
                        st.session_state.username,
                        user_query,
                        answer,
                        st.session_state.model,
                        st.session_state.conversation_id,
                        context_docs
                    )
                    
                    # Generate follow-up questions
                    suggested_questions = generate_follow_up_questions(
                        user_query,
                        answer,
                        st.session_state.model
                    )
                    
                    st.session_state.suggested_questions = suggested_questions
                
                # Before creating your text input widget
                if "clear_input" not in st.session_state:
                    st.session_state.clear_input = False

                if st.session_state.clear_input:
                    # Reset the flag
                    st.session_state.clear_input = False
                    # Set an empty default value for the widget
                    st.session_state.user_query = ""

                user_query = st.text_input("Your question:", key="user_query")

                # When you want to clear the input
                st.session_state.clear_input = True
                st.rerun()
                    
    # History Tab
    with tabs[1]:
        if show_history:
            st.subheader("Your Chat History")
            
            # Get list of conversations
            conversations = get_user_conversations(st.session_state.username)
            
            if conversations:
                # Let user select a conversation
                selected_conversation = st.selectbox(
                    "Select Conversation",
                    conversations,
                    format_func=lambda x: f"Conversation {x.split('-')[1]}" if x is not None else "Unknown Conversation"
                )
                  
                
                if selected_conversation:
                    # Load and display this conversation
                    messages = get_conversation(selected_conversation)
                    
                    for message in messages:
                        st.markdown(f"**Time:** {message.get('timestamp')}")
                        st.markdown(f"**Question:** {message.get('question')}")
                        st.markdown(f"**Answer:** {message.get('answer')}")
                        
                        # Show contexts if available
                        contexts = message.get("contexts", [])
                        if contexts:
                            with st.expander("View reference sources", expanded=False):
                                for i, ctx in enumerate(contexts):
                                    st.markdown(f"**Source {i+1}:** {ctx}")
                        
                        st.markdown("---")
                    
                    # Option to continue this conversation
                    if st.button("Continue this conversation"):
                        # Load the conversation into the current session
                        st.session_state.conversation_id = selected_conversation
                        
                        # Convert format from history to current_conversation format
                        st.session_state.current_conversation = []
                        for message in messages:
                            st.session_state.current_conversation.append({
                                "role": "user",
                                "content": message.get("question", ""),
                                "timestamp": message.get("timestamp")
                            })
                            st.session_state.current_conversation.append({
                                "role": "assistant",
                                "content": message.get("answer", ""),
                                "timestamp": message.get("timestamp"),
                                "model": message.get("model", "unknown"),
                                "conversation_id": message.get("conversation_id")
                            })
                        
                        # Switch to chat tab and rerun
                        st.rerun()
            else:
                st.info("No chat history found.")
    
    # Collections Tab
    with tabs[2]:
        st.subheader("Available Data Collections")
        
        # Show information about each collection
        for collection_name in qdrant_collections:
            with st.expander(f"{collection_name}", expanded=False):
                # Load a sample of documents from this collection
                vector_dim = SPECIAL_DIMS.get(collection_name, DEFAULT_EMBEDDING_DIM)
                
                try:
                    docs = fetch_qdrant_documents(collection_name, vector_dim, limit=5)
                    
                    if docs:
                        st.markdown(f"**Sample documents from {collection_name}:**")
                        
                        for i, doc in enumerate(docs):
                            with st.container():
                                st.markdown(f"**Document {i+1}**")
                                
                                # Extract and display key text
                                text = extract_key_text_from_payload(doc, collection_name)
                                st.markdown(f"```\n{text[:300]}...\n```")
                    else:
                        st.info(f"No documents found in {collection_name}.")
                
                except Exception as e:
                    st.error(f"Error loading documents from {collection_name}: {str(e)}")
        
        # Option to manually explore collections
        st.subheader("Explore Collections")
        
        explore_collection = st.selectbox(
            "Select collection to explore:",
            qdrant_collections
        )
        
        sample_size = st.slider("Sample size", min_value=5, max_value=50, value=10)
        
        if st.button("Load Samples"):
            with st.spinner(f"Loading samples from {explore_collection}..."):
                vector_dim = SPECIAL_DIMS.get(explore_collection, DEFAULT_EMBEDDING_DIM)
                samples = fetch_qdrant_documents(explore_collection, vector_dim, limit=sample_size)
                
                if samples:
                    st.success(f"Loaded {len(samples)} samples from {explore_collection}")
                    
                    # Convert to DataFrame for better display
                    samples_data = []
                    
                    for sample in samples:
                        # Extract key fields - this will depend on collection structure
                        sample_item = {}
                        
                        # Try to extract common fields
                        text_fields = ["content", "text", "description", "comment_text", "title", "body", "challenge", "solution"]
                        meta_fields = ["timestamp", "date", "user_id", "author", "province", "region", "status"]
                        
                        # Get text content
                        for field in text_fields:
                            if field in sample:
                                text = sample[field]
                                # Truncate long text
                                if isinstance(text, str) and len(text) > 100:
                                    sample_item[field] = text[:100] + "..."
                                else:
                                    sample_item[field] = text
                        
                        # Get metadata
                        for field in meta_fields:
                            if field in sample:
                                sample_item[field] = sample[field]
                        
                        samples_data.append(sample_item)
                    
                    # Convert to DataFrame and display
                    df = pd.DataFrame(samples_data)
                    st.dataframe(df)
                    
                    # Option to download the samples
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Samples CSV",
                        csv,
                        f"{explore_collection}_samples.csv",
                        "text/csv"
                    )
                else:
                    st.warning(f"No samples found in {explore_collection}.")

# ----------------------------------------------------------------
# App Entry Point
# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
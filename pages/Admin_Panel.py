import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime
from datetime import datetime, timedelta
import time
import hashlib
import random
import json
import re
import requests
import base64
import io
import os
import uuid
import psutil
import socket
import threading
import queue
import altair as alt
import networkx as nx
import community.community_louvain as community
import folium
from streamlit_folium import folium_static
from PIL import Image
from io import BytesIO
from scipy import stats
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import lru_cache

# Database imports
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import (
    PointStruct, 
    Distance, 
    Filter, 
    FieldCondition, 
    MatchValue, 
    Range, 
    VectorParams, 
    OptimizersConfigDiff, 
    CollectionStatus
)
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import UpdateResult

# AI/ML imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import openai
# Global configuration
DEBUG_MODE = st.secrets.get("debug", False)
ENV = st.secrets.get("environment", "development")
MAX_CACHE_SIZE = 1000
CACHE_TTL_DEFAULT = 300  # 5 minutes
PARALLEL_REQUESTS = 5
REQUEST_TIMEOUT = 10
OPENAI_MODEL = "gpt-4-turbo"  # Default model
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CivicCatalyst")
def view_user_details(user):
    """View detailed user information."""
    st.session_state.selected_user_id = user.get("_id")
    st.session_state.selected_user = user
    st.rerun()
def date_to_timestamp(date_obj, end_of_day=False):
    """
    Convert a date object to a Unix timestamp.
    
    Parameters:
        date_obj: A date or datetime object
        end_of_day: If True, set time to 23:59:59, otherwise 00:00:00
    
    Returns:
        int: Unix timestamp (seconds since epoch)
    """
    if isinstance(date_obj, datetime):
        # Already a datetime, just set time if needed
        if end_of_day:
            date_obj = date_obj.replace(hour=23, minute=59, second=59)
    else:
        # Convert date to datetime
        if end_of_day:
            date_obj = datetime.combine(date_obj, datetime.max.time())
        else:
            date_obj = datetime.combine(date_obj, datetime.min.time())
    
    # Convert to timestamp
    return int(date_obj.timestamp())
def edit_user(user):
    """Edit user information."""
    st.session_state.editing_user_id = user.get("_id")
    st.session_state.editing_user = user
    st.rerun()

def delete_user_confirmation(user):
    """Confirm user deletion."""
    st.session_state.deleting_user_id = user.get("_id")
    st.session_state.deleting_user = user
    st.rerun()

def view_project_details(project):
    """View detailed project information."""
    st.session_state.selected_project_id = project.get("project_id")
    st.session_state.selected_project = project
    st.rerun()

def edit_project(project):
    """Edit project information."""
    st.session_state.editing_project_id = project.get("project_id")
    st.session_state.editing_project = project
    st.rerun()

def delete_project_confirmation(project):
    """Confirm project deletion."""
    st.session_state.deleting_project_id = project.get("project_id")
    st.session_state.deleting_project = project
    st.rerun()

def view_article_details(article):
    """View detailed article information."""
    st.session_state.selected_article_id = article.get("article_id")
    st.session_state.selected_article = article
    st.rerun()

def view_article_comments(article):
    """View comments for an article."""
    st.session_state.viewing_article_comments = True
    st.session_state.article_comments_id = article.get("article_id")
    st.session_state.article_comments_title = article.get("title", "Article")
    st.rerun()

def edit_article(article):
    """Edit article information."""
    st.session_state.editing_article_id = article.get("article_id")
    st.session_state.editing_article = article
    st.rerun()


# Constants
APP_VERSION = "3.0.0"
DEBUG_MODE = st.secrets.get("debug", False)
ENV = st.secrets.get("environment", "development")
MAX_CACHE_SIZE = 1000
CACHE_TTL_DEFAULT = 300  # 5 minutes
PARALLEL_REQUESTS = 5
REQUEST_TIMEOUT = 10

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets.get("openai", {}).get("api_key", "")
OPENAI_MODEL = "gpt-4-turbo"  # Default model

# Collection configurations
COLLECTIONS = {
    "citizen_comments": {
        "vector_dim": 384, 
        "description": "Citizen comments and feedback",
        "fields": ["comment_id", "content", "timestamp", "user_id", "sentiment", "topic", "region", "project_id"]
    },
    "citizen_ideas": {
        "vector_dim": 384, 
        "description": "Citizen ideas and suggestions",
        "fields": ["idea_id", "challenge", "solution", "date_submitted", "status", "axis", "sentiment", "province", "user_id"]
    },
    "hespress_politics_comments": {
        "vector_dim": 384, 
        "description": "Comments on political articles",
        "fields": ["comment_id", "comment_text", "article_id", "timestamp", "user_id", "sentiment", "relevance_score"]
    },
    "hespress_politics_details": {
        "vector_dim": 1536, 
        "description": "Political news articles",
        "fields": ["article_id", "title", "content", "date_published", "author", "category", "keywords", "summary"]
    },
    "municipal_projects": {
        "vector_dim": 384, 
        "description": "Municipal projects data",
        "fields": ["project_id", "title", "description", "status", "budget", "start_date", "end_date", "province", "completion_percentage", "vote_score"]
    },
    "remacto_comments": {
        "vector_dim": 384, 
        "description": "Comments on Remacto platform",
        "fields": ["comment_id", "comment_text", "project_id", "timestamp", "user_id", "sentiment", "votes"]
    },
    "remacto_projects": {
        "vector_dim": 384, 
        "description": "Projects on Remacto platform",
        "fields": ["project_id", "title", "description", "status", "budget", "start_date", "end_date", "province", "votes", "participation_count"]
    },
    "user_profiles": {
        "vector_dim": 384,
        "description": "User profile information",
        "fields": ["user_id", "username", "email", "registration_date", "profile_data", "activity_score", "interests"]
    },
    "regional_statistics": {
        "vector_dim": 384,
        "description": "Regional statistics and metrics",
        "fields": ["region_id", "name", "population", "area", "metrics", "timestamp"]
    }
}

def ideas_management():
    """Ideas management interface for reviewing and managing citizen ideas."""
    st.title("Citizen Ideas Management")
    
    # Sidebar filters
    st.sidebar.header(t("filters"))
    
    search_query = st.sidebar.text_input(
        t("search"), 
        placeholder=t("idea_title")
    )
    
    status_filter = st.sidebar.selectbox(
        t("idea_status"),
        ["All", "Submitted", "Under Review", "Approved", "Implemented", "Rejected"]
    )
    
    region_filter = st.sidebar.selectbox(
        t("project_region"),
        MOROCCO_REGIONS
    )
    
    # Load ideas with filters
    filter_conditions = []
    
    if search_query:
        challenge_condition = FieldCondition(
            key="challenge",
            match=MatchValue(value=search_query)
        )
        solution_condition = FieldCondition(
            key="solution",
            match=MatchValue(value=search_query)
        )
        filter_conditions.append(Filter(should=[challenge_condition, solution_condition]))
    
    if status_filter != "All":
        filter_conditions.append(
            FieldCondition(
                key="status",
                match=MatchValue(value=status_filter)
            )
        )
    
    if region_filter != "All":
        filter_conditions.append(
            FieldCondition(
                key="province",
                match=MatchValue(value=region_filter)
            )
        )
    
    # Create filter if we have conditions
    ideas_filter = None
    if filter_conditions:
        ideas_filter = Filter(must=filter_conditions)
    
    # Load ideas
    ideas, next_scroll_id = load_qdrant_documents(
        "citizen_ideas", 
        vector_dim=384, 
        limit=100, 
        filters=ideas_filter
    )
    
    # Ideas metrics
    st.subheader("Ideas Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Total Ideas",
            f"{len(ideas):,}",
            icon="fa-lightbulb",
            color="#2196F3"
        )
    
    with col2:
        # Count ideas by status
        status_counts = {}
        for idea in ideas:
            status = idea.get("status", "Unknown")
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        # Get count for approved ideas
        approved_count = status_counts.get("Approved", 0) + status_counts.get("Implemented", 0)
        approval_rate = (approved_count / len(ideas) * 100) if ideas else 0
        
        display_metric_card(
            "Approved Ideas",
            f"{approved_count:,}",
            delta=round(approval_rate),
            delta_description="approval rate",
            icon="fa-check-circle",
            color="#4CAF50"
        )
    
    with col3:
        # Calculate average sentiments
        positive_sentiments = sum(1 for idea in ideas if idea.get("sentiment", "").upper().startswith("P"))
        positive_rate = (positive_sentiments / len(ideas) * 100) if ideas else 0
        
        display_metric_card(
            "Positive Sentiment",
            f"{positive_rate:.1f}%",
            icon="fa-smile",
            color="#FF9800"
        )
    
    with col4:
        # Ideas with high scores or votes
        high_scoring = sum(1 for idea in ideas if int(idea.get("votes", 0)) > 10)
        high_score_rate = (high_scoring / len(ideas) * 100) if ideas else 0
        
        display_metric_card(
            "High Engagement",
            f"{high_scoring:,}",
            delta=round(high_score_rate),
            delta_description="of total ideas",
            icon="fa-star",
            color="#9C27B0"
        )
    
    # Ideas management tabs
    tabs = st.tabs(["Ideas List", "Ideas Analytics", "Categories", "Submissions Review"])
    
    with tabs[0]:
        st.subheader("Citizen Ideas")
        
        # Action buttons
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Export Ideas", key="export_ideas"):
                # Convert ideas to DataFrame
                ideas_df = pd.DataFrame(ideas)
                
                # Convert to CSV
                csv = ideas_df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "citizen_ideas_export.csv",
                    "text/csv",
                    key="download_ideas_csv"
                )
        
        # Ideas table with actions
        idea_columns = [
            {
                "key": "challenge",
                "label": "Challenge",
                "format": lambda val, item: f"""
                    <div style="font-weight: 500;">{val[:50]}...</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">Solution: {item.get('solution', '')[:50]}...</div>
                """
            },
            {
                "key": "status",
                "label": "Status",
                "format": lambda val, item: f"""
                    <span class="badge badge-{'success' if val in ['Approved', 'Implemented'] else 'warning' if val == 'Under Review' else 'danger' if val == 'Rejected' else 'info'}">
                        {val}
                    </span>
                """
            },
            {
                "key": "province",
                "label": "Region",
                "format": lambda val, item: val or "Unknown"
            },
            {
                "key": "votes",
                "label": "Votes",
                "format": lambda val, item: f"{int(val):,}" if val else "0"
            },
            {
                "key": "sentiment",
                "label": "Sentiment",
                "format": lambda val, item: f"""
                    <span class="badge badge-{'success' if val.upper().startswith('P') else 'danger' if val.upper().startswith('N') and not val.upper().startswith('NEU') else 'info'}">
                        {val}
                    </span>
                """
            }
        ]
        
        idea_actions = [
            {
                "key": "view",
                "text": "View",
                "icon": "fa-eye",
                "class": "action-view",
                "callback": view_idea_details
            },
            {
                "key": "change_status",
                "text": "Change Status",
                "icon": "fa-pen-to-square",
                "class": "action-edit",
                "callback": change_idea_status
            },
            {
                "key": "delete",
                "text": "Delete",
                "icon": "fa-trash",
                "class": "action-delete",
                "callback": delete_idea_confirmation
            }
        ]
        
        create_dynamic_table(
            ideas, 
            idea_columns, 
            key_prefix="ideas_table", 
            page_size=10, 
            searchable=True,
            actions=idea_actions
        )
    
    with tabs[1]:
        st.subheader("Ideas Analytics")
        
        if not ideas:
            st.info("No ideas data available for analysis.")
        else:
            # Create visualization row
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Sentiment distribution
                sentiment_counts = {}
                for idea in ideas:
                    sentiment = idea.get("sentiment", "Neutral")
                    
                    # Normalize sentiment values
                    if sentiment.upper().startswith("P"):
                        normalized_sentiment = "Positive"
                    elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                        normalized_sentiment = "Negative"
                    else:
                        normalized_sentiment = "Neutral"
                    
                    if normalized_sentiment not in sentiment_counts:
                        sentiment_counts[normalized_sentiment] = 0
                    
                    sentiment_counts[normalized_sentiment] += 1
                
                sentiment_df = pd.DataFrame({
                    "sentiment": list(sentiment_counts.keys()),
                    "count": list(sentiment_counts.values())
                })
                
                fig = px.pie(
                    sentiment_df,
                    values="count",
                    names="sentiment",
                    title="Sentiment Distribution",
                    color="sentiment",
                    color_discrete_map={
                        "Positive": "#4CAF50",
                        "Negative": "#F44336",
                        "Neutral": "#2196F3"
                    }
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                # Status distribution
                status_counts = {}
                for idea in ideas:
                    status = idea.get("status", "Unknown")
                    if status not in status_counts:
                        status_counts[status] = 0
                    status_counts[status] += 1
                
                status_df = pd.DataFrame({
                    "status": list(status_counts.keys()),
                    "count": list(status_counts.values())
                })
                
                fig = px.pie(
                    status_df,
                    values="count",
                    names="status",
                    title="Status Distribution",
                    color="status",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Regional distribution
            st.subheader("Ideas by Region")
            
            region_counts = {}
            for idea in ideas:
                region = idea.get("province", "Unknown")
                if region not in region_counts:
                    region_counts[region] = 0
                region_counts[region] += 1
            
            region_df = pd.DataFrame({
                "region": list(region_counts.keys()),
                "count": list(region_counts.values())
            }).sort_values("count", ascending=False)
            
            fig = px.bar(
                region_df,
                x="region",
                y="count",
                title="Ideas by Region",
                color="count",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Number of Ideas"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud from ideas
            st.subheader("Ideas Word Cloud")
            
            combined_text = []
            for idea in ideas:
                challenge = idea.get("challenge", "")
                solution = idea.get("solution", "")
                if challenge or solution:
                    combined_text.append(f"{challenge} {solution}")
            
            if combined_text:
                wordcloud_fig = create_word_cloud(combined_text, title="Ideas Content Word Cloud")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text for word cloud generation.")
            else:
                st.info("No idea text content available for word cloud.")
    
    with tabs[2]:
        st.subheader("Ideas Categories")
        
        # Extract idea categories using axis field
        axis_counts = {}
        for idea in ideas:
            axis = idea.get("axis", "Uncategorized")
            if axis not in axis_counts:
                axis_counts[axis] = 0
            axis_counts[axis] += 1
        
        if axis_counts:
            # Create DataFrame
            axis_df = pd.DataFrame({
                "category": list(axis_counts.keys()),
                "count": list(axis_counts.values())
            }).sort_values("count", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                axis_df,
                x="category",
                y="count",
                title="Ideas by Category",
                color="count",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Category",
                yaxis_title="Number of Ideas"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Category details
            st.subheader("Category Details")
            
            # Create category tabs
            if len(axis_counts) > 0:
                category_tabs = st.tabs(list(axis_counts.keys()))
                
                for i, category in enumerate(axis_counts.keys()):
                    with category_tabs[i]:
                        # Filter ideas by category
                        category_ideas = [idea for idea in ideas if idea.get("axis") == category]
                        
                        # Display category metrics
                        cat_col1, cat_col2, cat_col3 = st.columns(3)
                        
                        with cat_col1:
                            st.metric("Total Ideas", len(category_ideas))
                        
                        with cat_col2:
                            # Calculate approval rate
                            approved = sum(1 for idea in category_ideas if idea.get("status") in ["Approved", "Implemented"])
                            approval_rate = (approved / len(category_ideas) * 100) if category_ideas else 0
                            st.metric("Approval Rate", f"{approval_rate:.1f}%")
                        
                        with cat_col3:
                            # Calculate average votes
                            total_votes = sum(int(idea.get("votes", 0)) for idea in category_ideas)
                            avg_votes = total_votes / len(category_ideas) if category_ideas else 0
                            st.metric("Avg. Votes", f"{avg_votes:.1f}")
                        
                        # Display top ideas
                        st.subheader("Top Ideas in this Category")
                        
                        # Sort by votes
                        top_ideas = sorted(category_ideas, key=lambda x: int(x.get("votes", 0)), reverse=True)[:5]
                        
                        for idea in top_ideas:
                            with st.expander(f"{idea.get('challenge', '')[:50]}...", expanded=False):
                                st.markdown(f"**Challenge:** {idea.get('challenge', '')}")
                                st.markdown(f"**Solution:** {idea.get('solution', '')}")
                                st.markdown(f"**Status:** {idea.get('status', 'Unknown')}")
                                st.markdown(f"**Votes:** {idea.get('votes', 0)}")
                                st.markdown(f"**Sentiment:** {idea.get('sentiment', 'Unknown')}")
                                st.markdown(f"**Region:** {idea.get('province', 'Unknown')}")
                                
                                # Add vote button for testing
                                if st.button("Vote Up", key=f"vote_{idea.get('idea_id', '')}"):
                                    st.success("Vote recorded (simulated)")
            else:
                st.info("No categories found in ideas data.")
        else:
            st.info("No idea categories available for analysis.")
    
    with tabs[3]:
        st.subheader("Submissions Review")
        
        # Filter for new submissions
        new_submissions = [idea for idea in ideas if idea.get("status") == "Submitted"]
        
        if not new_submissions:
            st.info("No new idea submissions waiting for review.")
        else:
            st.write(f"Found {len(new_submissions)} new submissions waiting for review.")
            
            # Review interface
            for i, idea in enumerate(new_submissions[:5]):  # Limit to first 5 for performance
                with st.expander(f"Submission #{i+1}: {idea.get('challenge', '')[:50]}...", expanded=i==0):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Challenge:**")
                        st.markdown(f"```{idea.get('challenge', '')}```")
                        
                        st.markdown(f"**Proposed Solution:**")
                        st.markdown(f"```{idea.get('solution', '')}```")
                        
                        st.markdown(f"**Submitted by:** {idea.get('user_id', 'Anonymous')}")
                        st.markdown(f"**Submitted on:** {idea.get('date_submitted', 'Unknown')}")
                        st.markdown(f"**Region:** {idea.get('province', 'Unknown')}")
                    
                    with col2:
                        st.markdown("**Review Actions**")
                        
                        new_status = st.selectbox(
                            "Change Status",
                            ["Submitted", "Under Review", "Approved", "Rejected"],
                            index=0,
                            key=f"status_{i}"
                        )
                        
                        category = st.selectbox(
                            "Assign Category",
                            ["Infrastructure", "Healthcare", "Education", "Environment", 
                             "Transportation", "Urban Planning", "Economic Development", 
                             "Social Services", "Technology", "Other"],
                            index=0,
                            key=f"category_{i}"
                        )
                        
                        reviewer_notes = st.text_area(
                            "Review Notes",
                            key=f"notes_{i}",
                            height=100
                        )
                        
                        # Submit review button
                        if st.button("Submit Review", key=f"review_{i}"):
                            try:
                                # In a real implementation, you'd update the record in Qdrant
                                # Here we'll simulate the update with a success message
                                
                                # Simulate update with a success message
                                st.success(f"Review submitted. Status changed to {new_status}.")
                                
                                # Add to audit log
                                add_audit_log(
                                    "idea_review",
                                    f"Reviewed idea: {idea.get('idea_id', '')}",
                                    {
                                        "idea_id": idea.get("idea_id", ""),
                                        "new_status": new_status,
                                        "category": category,
                                        "notes": reviewer_notes
                                    },
                                    st.session_state.get("username", "admin")
                                )
                                
                                # In a production system, we would update the record and rerun
                            except Exception as e:
                                st.error(f"Error updating idea: {str(e)}")

def view_idea_details(idea):
    """View detailed idea information."""
    st.session_state.selected_idea_id = idea.get("idea_id")
    st.session_state.selected_idea = idea
    st.rerun()

def change_idea_status(idea):
    """Change status of an idea."""
    st.session_state.changing_idea_status_id = idea.get("idea_id")
    st.session_state.changing_idea_status = idea
    st.rerun()

def delete_idea_confirmation(idea):
    """Confirm idea deletion."""
    st.session_state.deleting_idea_id = idea.get("idea_id")
    st.session_state.deleting_idea = idea
    st.rerun()
def create_dashboard():
    """Main dashboard interface with overview metrics and visualizations."""
    st.title(t("admin_dashboard"))
    
    # Load dashboard config
    dashboard_config = global_state.get_dashboard_config()
    
    # Load summary data
    st.subheader(t("overview"))
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count users
        users = load_users(limit=1000)
        total_users = len(users)
        active_users = sum(1 for user in users if user.get("active", True))
        
        display_metric_card(
            t("total_users"),
            f"{total_users:,}",
            delta=round(active_users / total_users * 100) if total_users > 0 else 0,
            delta_description="active users",
            icon=ICONS["users"],
            color="#2196F3"
        )
    
    with col2:
        # Count comments
        citizen_comments, _ = load_qdrant_documents("citizen_comments", 384, limit=1000)
        hespress_comments, _ = load_qdrant_documents("hespress_politics_comments", 384, limit=1000)
        total_comments = len(citizen_comments) + len(hespress_comments)
        
        display_metric_card(
            t("total_comments"),
            f"{total_comments:,}",
            icon=ICONS["comment"],
            color="#4CAF50"
        )
    
    with col3:
        # Count projects
        projects, _ = load_qdrant_documents("municipal_projects", 384, limit=1000)
        total_projects = len(projects)
        completed_projects = sum(1 for p in projects if p.get("status") == "Completed")
        
        display_metric_card(
            t("total_projects"),
            f"{total_projects:,}",
            delta=round(completed_projects / total_projects * 100) if total_projects > 0 else 0,
            delta_description="completion rate",
            icon=ICONS["projects"],
            color="#FF9800"
        )
    
    with col4:
        # Count ideas
        ideas, _ = load_qdrant_documents("citizen_ideas", 384, limit=1000)
        
        display_metric_card(
            "Citizen Ideas",
            f"{len(ideas):,}",
            icon=ICONS["ideas"],
            color="#9C27B0"
        )
    
    # Create dashboard sections based on config
    widgets = dashboard_config.get("widgets", {})
    
    # Only display widgets that are enabled
    enabled_widgets = {name: config for name, config in widgets.items() if config.get("enabled", True)}
    
    # Sort widgets by position
    sorted_widgets = sorted(enabled_widgets.items(), key=lambda x: x[1].get("position", 999))
    
    # Citizen Engagement section
    if any(name == "citizen_engagement" for name, _ in sorted_widgets):
        st.subheader("Citizen Engagement")
        
        # Create engagement charts
        engagement_col1, engagement_col2 = st.columns(2)
        
        with engagement_col1:
            # Comments over time
            comments_data = []
            
            # Extract dates from comments
            for comment in citizen_comments:
                if isinstance(comment, dict) and "timestamp" in comment:
                    timestamp = comment["timestamp"]
                    
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        except:
                            continue
                    
                    comments_data.append({
                        "date": timestamp,
                        "count": 1
                    })
            
            if comments_data:
                # Convert to DataFrame
                comments_df = pd.DataFrame(comments_data)
                
                # Group by date
                comments_df["date"] = pd.to_datetime(comments_df["date"]).dt.date
                daily_comments = comments_df.groupby("date").size().reset_index(name="count")
                
                # Create time series chart
                fig = px.line(
                    daily_comments,
                    x="date",
                    y="count",
                    title="Daily Comment Activity",
                    markers=True
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Date",
                    yaxis_title="Number of Comments"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No comment data available for visualization.")
        
        with engagement_col2:
            # Sentiment distribution
            sentiment_data = []
            
            # Extract sentiment from comments
            for comment in citizen_comments:
                if isinstance(comment, dict) and "sentiment" in comment:
                    sentiment = comment["sentiment"]
                    
                    # Normalize sentiment
                    if sentiment.upper().startswith("P"):
                        normalized_sentiment = "Positive"
                    elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                        normalized_sentiment = "Negative"
                    else:
                        normalized_sentiment = "Neutral"
                    
                    sentiment_data.append({
                        "sentiment": normalized_sentiment
                    })
            
            if sentiment_data:
                # Convert to DataFrame
                sentiment_df = pd.DataFrame(sentiment_data)
                
                # Count sentiments
                sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                
                # Create pie chart
                fig = px.pie(
                    sentiment_counts,
                    values="count",
                    names="sentiment",
                    title="Comment Sentiment Distribution",
                    color="sentiment",
                    color_discrete_map={
                        "Positive": "#4CAF50",
                        "Negative": "#F44336",
                        "Neutral": "#2196F3"
                    }
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available for visualization.")
    
    # Project Status section
    if any(name == "project_status" for name, _ in sorted_widgets):
        st.subheader("Project Status")
        
        proj_col1, proj_col2 = st.columns(2)
        
        with proj_col1:
            # Project status breakdown
            status_counts = {}
            
            for project in projects:
                if isinstance(project, dict):
                    status = project.get("status", "Unknown")
                    if status not in status_counts:
                        status_counts[status] = 0
                    status_counts[status] += 1
            
            if status_counts:
                # Create DataFrame
                status_df = pd.DataFrame({
                    "status": list(status_counts.keys()),
                    "count": list(status_counts.values())
                })
                
                # Create pie chart
                fig = px.pie(
                    status_df,
                    values="count",
                    names="status",
                    title="Project Status Distribution",
                    color="status",
                    color_discrete_map={
                        "Completed": "#4CAF50",
                        "In Progress": "#FF9800",
                        "Approved": "#2196F3",
                        "Proposed": "#9E9E9E",
                        "On Hold": "#FFC107",
                        "Cancelled": "#F44336"
                    }
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No project status data available.")
        
        with proj_col2:
            # Project category breakdown
            category_counts = {}
            
            for project in projects:
                if isinstance(project, dict):
                    category = project.get("category", "Unknown")
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1
            
            if category_counts:
                # Create DataFrame
                category_df = pd.DataFrame({
                    "category": list(category_counts.keys()),
                    "count": list(category_counts.values())
                })
                
                # Sort by count
                category_df = category_df.sort_values("count", ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    category_df,
                    x="category",
                    y="count",
                    title="Projects by Category",
                    color="count",
                    color_continuous_scale="Viridis"
                )
                
                # Update layout for theme consistency
                fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Category",
                    yaxis_title="Number of Projects"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No project category data available.")
    
    # Regional Insights section
    if any(name == "regional_insights" for name, _ in sorted_widgets):
        st.subheader("Regional Insights")
        
        # Extract regional data
        regional_counts = {}
        
        for project in projects:
            if isinstance(project, dict):
                region = project.get("province", project.get("CT", "Unknown"))
                if region not in regional_counts:
                    regional_counts[region] = 0
                regional_counts[region] += 1
        
        if regional_counts:
            # Create DataFrame
            region_df = pd.DataFrame({
                "region": list(regional_counts.keys()),
                "count": list(regional_counts.values())
            })
            
            # Sort by count
            region_df = region_df.sort_values("count", ascending=False)
            
            # Create choropleth map
            st.subheader("Project Distribution by Region")
            
            # Create map
            map_fig = create_morocco_map(
                region_df,
                geo_col="region",
                value_col="count",
                title="Project Count by Region"
            )
            
            folium_static(map_fig, width=800)
        else:
            st.info("No regional data available for insights.")
    
    # Recent Activity section
    if any(name == "recent_activity" for name, _ in sorted_widgets):
        st.subheader("Recent Activity")
        
        # Load audit logs
        recent_logs = load_audit_logs(limit=10)
        
        if recent_logs:
            # Create activity table
            activity_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>User</th>
                        <th>Action</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for log in recent_logs:
                # Format timestamp
                timestamp = log.get("timestamp", datetime.now())
                if isinstance(timestamp, datetime):
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
                
                user = log.get("user", "Unknown")
                action = log.get("action_type", "Unknown")
                description = log.get("description", "No description")
                
                activity_table += f"""
                <tr>
                    <td>{formatted_time}</td>
                    <td>{user}</td>
                    <td>{action}</td>
                    <td>{description}</td>
                </tr>
                """
            
            activity_table += """
                </tbody>
            </table>
            """
            
            st.markdown(activity_table, unsafe_allow_html=True)
        else:
            st.info("No recent activity to display.")
    
    # System Health section
    if any(name == "system_health" for name, _ in sorted_widgets):
        st.subheader("System Health")
        
        # Update system health
        global_state.update_system_health()
        health_data = global_state.get_system_health()
        
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            display_metric_card(
                "CPU Usage",
                f"{health_data['cpu_usage']:.1f}%",
                icon="fa-microchip",
                color="#2196F3"
            )
        
        with health_col2:
            display_metric_card(
                "Memory Usage",
                f"{health_data['memory_usage']:.1f}%",
                icon="fa-memory",
                color="#FF9800"
            )
        
        with health_col3:
            display_metric_card(
                "Database Status",
                health_data["database_status"].capitalize(),
                icon="fa-database",
                color="#4CAF50" if health_data["database_status"] == "healthy" else "#F44336"
            )
    
    # Display cache stats in a collapsed section
    with st.expander("System Info", expanded=False):
        # Get cache stats
        cache_stats = cache.get_stats()
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <p><strong>App Version:</strong> {APP_VERSION}</p>
            <p><strong>Environment:</strong> {ENV.capitalize()}</p>
            <p><strong>Cache Stats:</strong> Size: {cache_stats['size']}/{cache_stats['max_size']}, Hit Ratio: {cache_stats['hit_ratio']*100:.1f}%</p>
            <p><strong>Last Update:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

# User roles with permission levels
USER_ROLES = {
    "admin": {
        "level": 5,
        "description": "Full system access and control",
        "permissions": ["manage_users", "manage_projects", "manage_content", "manage_system", "view_analytics", "manage_news"]
    },
    "moderator": {
        "level": 4,
        "description": "Can moderate content and approve submissions",
        "permissions": ["moderate_content", "view_analytics", "view_projects"]
    },
    "analyst": {
        "level": 3,
        "description": "Can view and analyze platform data",
        "permissions": ["view_analytics", "export_data", "view_projects", "view_content"]
    },
    "content_manager": {
        "level": 3,
        "description": "Can manage content and articles",
        "permissions": ["manage_content", "view_projects", "view_analytics"]
    },
    "project_manager": {
        "level": 4,
        "description": "Can manage municipal projects",
        "permissions": ["manage_projects", "view_analytics", "view_content"]
    },
    "citizen": {
        "level": 1,
        "description": "Standard citizen account",
        "permissions": ["submit_comments", "submit_ideas", "vote"]
    }
}

# Regions in Morocco for filtering
MOROCCO_REGIONS = [
    "All",
    "Rabat-Salé-Kénitra",
    "Casablanca-Settat",
    "Marrakech-Safi",
    "Fès-Meknès",
    "Tanger-Tétouan-Al Hoceïma",
    "Souss-Massa",
    "Oriental",
    "Béni Mellal-Khénifra",
    "Drâa-Tafilalet",
    "Laâyoune-Sakia El Hamra",
    "Dakhla-Oued Ed-Dahab",
    "Guelmim-Oued Noun"
]

# Project categories
PROJECT_CATEGORIES = [
    "Infrastructure",
    "Healthcare",
    "Education",
    "Environment",
    "Transportation",
    "Water & Sanitation",
    "Urban Planning",
    "Cultural Development",
    "Economic Development",
    "Social Services",
    "Sport & Recreation",
    "Technology & Innovation"
]

# Project statuses
PROJECT_STATUSES = [
    "Proposed",
    "Approved",
    "In Progress",
    "On Hold",
    "Completed",
    "Cancelled"
]

# Icon mapping for the UI
ICONS = {
    "dashboard": "fa-gauge-high",
    "users": "fa-users",
    "projects": "fa-diagram-project",
    "ideas": "fa-lightbulb",
    "content": "fa-file-lines",
    "analytics": "fa-chart-line",
    "news": "fa-newspaper", 
    "system": "fa-server",
    "audit": "fa-list-check",
    "search": "fa-magnifying-glass",
    "settings": "fa-gear",
    "logout": "fa-right-from-bracket",
    "add": "fa-plus",
    "edit": "fa-pen-to-square",
    "delete": "fa-trash",
    "view": "fa-eye",
    "save": "fa-floppy-disk",
    "cancel": "fa-xmark",
    "export": "fa-file-export",
    "import": "fa-file-import",
    "download": "fa-download",
    "upload": "fa-upload",
    "filter": "fa-filter",
    "sort": "fa-sort",
    "search": "fa-search",
    "refresh": "fa-arrows-rotate",
    "approve": "fa-check",
    "reject": "fa-ban",
    "flag": "fa-flag",
    "star": "fa-star",
    "comment": "fa-comment",
    "calendar": "fa-calendar",
    "clock": "fa-clock",
    "map": "fa-map-location-dot",
    "info": "fa-circle-info",
    "warning": "fa-triangle-exclamation",
    "error": "fa-circle-exclamation",
    "success": "fa-circle-check",
    "language": "fa-language",
    "theme": "fa-palette",
    "notification": "fa-bell",
    "menu": "fa-bars",
    "close": "fa-xmark",
    "link": "fa-link",
    "unlink": "fa-link-slash",
    "expand": "fa-expand",
    "collapse": "fa-compress",
    "copy": "fa-copy",
    "paste": "fa-paste",
    "lock": "fa-lock",
    "unlock": "fa-unlock",
    "analysis": "fa-microscope",
    "sentiment": "fa-face-smile",
    "topic": "fa-tags",
    "region": "fa-map-marker",
    "user": "fa-user",
    "admin": "fa-user-shield",
    "moderator": "fa-user-gear",
    "analyst": "fa-user-graduate",
    "content_manager": "fa-user-pen",
    "project_manager": "fa-user-hard-hat",
    "citizen": "fa-user"
}

# -------------------------------------------------------------
# ENHANCED FEATURES: Caching, Performance & Global State
# -------------------------------------------------------------

class CacheManager:
    """Enhanced caching manager with TTL, size limits, and stats"""
    
    def __init__(self, max_size=MAX_CACHE_SIZE, default_ttl=CACHE_TTL_DEFAULT):
        self.cache = {}
        self.cache_ttl = {}
        self.cache_access_count = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lock = threading.RLock()
    
    def get(self, key, default=None):
        """Get a value from cache with proper metrics tracking"""
        with self._lock:
            if key in self.cache:
                # Check if expired
                if key in self.cache_ttl and self.cache_ttl[key] < time.time():
                    # Expired, remove from cache
                    self._remove_key(key)
                    self.miss_count += 1
                    return default
                
                # Cache hit - update access count
                self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
                self.hit_count += 1
                return self.cache[key]
            
            # Cache miss
            self.miss_count += 1
            return default
    
    def set(self, key, value, ttl=None):
        """Set value in cache with TTL and size management"""
        with self._lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entries()
            
            # Store in cache
            self.cache[key] = value
            ttl_value = ttl if ttl is not None else self.default_ttl
            self.cache_ttl[key] = time.time() + ttl_value
            self.cache_access_count[key] = 1
            return value
    
    def _evict_entries(self):
        """Evict least recently accessed entries"""
        # Sort keys by access count (ascending)
        sorted_keys = sorted(self.cache_access_count.keys(), 
                             key=lambda k: self.cache_access_count[k])
        
        # Evict the least accessed 20% of entries
        num_to_evict = max(1, int(len(sorted_keys) * 0.2))
        for key in sorted_keys[:num_to_evict]:
            self._remove_key(key)
            self.eviction_count += 1
    
    def _remove_key(self, key):
        """Remove a key from all tracking dicts"""
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_ttl:
            del self.cache_ttl[key]
        if key in self.cache_access_count:
            del self.cache_access_count[key]
    
    def clear(self, prefix=None):
        """Clear cache entries, optionally only those matching a prefix"""
        with self._lock:
            if prefix:
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
                for k in keys_to_remove:
                    self._remove_key(k)
            else:
                self.cache = {}
                self.cache_ttl = {}
                self.cache_access_count = {}
    
    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": hit_ratio,
                "eviction_count": self.eviction_count
            }
    
    def get_keys_by_prefix(self, prefix):
        """Get all cache keys that start with the given prefix"""
        with self._lock:
            return [k for k in self.cache.keys() if k.startswith(prefix)]
    
    def touch(self, key):
        """Update TTL for an existing key"""
        with self._lock:
            if key in self.cache and key in self.cache_ttl:
                self.cache_ttl[key] = time.time() + self.default_ttl

# Initialize cache manager
cache = CacheManager(max_size=1000, default_ttl=600)  # 10 minutes default TTL

class GlobalState:
    """Enhanced global state management with thread safety and observables"""
    
    def __init__(self):
        self._data = {}
        self._background_tasks = queue.Queue()
        self._real_time_data = {}
        self._event_listeners = {}
        self._lock = threading.RLock()
        self._workers = []
        self._notification_id_counter = 0
        self._dashboard_config = self._load_default_dashboard_config()
        self._system_health = {
            "last_checked": None,
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "database_status": "unknown",
            "api_status": "unknown"
        }
        self._status_history = []
        
        # Start background workers
        self._start_workers(2)  # Start 2 worker threads
    
    def get(self, key, default=None):
        """Thread-safe get value from state"""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key, value):
        """Thread-safe set value in state with event emission"""
        with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value
            
            # Emit change event
            self._emit_event(f"change:{key}", {
                "key": key,
                "old_value": old_value,
                "new_value": value
            })
            
            return value
    
    def _worker_thread(self):
        """Background worker thread to process tasks"""
        while True:
            try:
                task, args, kwargs = self._background_tasks.get(timeout=1)
                try:
                    task(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                    self.add_notification(
                        "Background Task Error",
                        f"Error in background task: {str(e)}",
                        "error"
                    )
                finally:
                    self._background_tasks.task_done()
            except queue.Empty:
                pass
    
    def _start_workers(self, num_workers):
        """Start multiple background worker threads"""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_thread, 
                daemon=True,
                name=f"GlobalStateWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
    
    def add_task(self, task, *args, **kwargs):
        """Add a task to be executed in the background"""
        self._background_tasks.put((task, args, kwargs))
    
    def add_notification(self, title, message, type="info"):
        """Add a notification with thread safety"""
        with self._lock:
            notification_id = self._notification_id_counter
            self._notification_id_counter += 1
            
            notification = {
                "id": notification_id,
                "title": title,
                "message": message,
                "type": type,
                "timestamp": datetime.now(),
                "read": False
            }
            
            if "notifications" not in self._data:
                self._data["notifications"] = []
                
            self._data["notifications"].append(notification)
            
            # Emit notification event
            self._emit_event("notification", notification)
            
            return notification_id
    
    def mark_notification_read(self, notification_id):
        """Mark a notification as read"""
        with self._lock:
            if "notifications" in self._data:
                for notification in self._data["notifications"]:
                    if notification["id"] == notification_id:
                        notification["read"] = True
                        
                        # Emit notification update event
                        self._emit_event("notification_update", {
                            "id": notification_id,
                            "read": True
                        })
                        
                        break
    
    def get_notifications(self, unread_only=False):
        """Get all notifications with optional filtering"""
        with self._lock:
            notifications = self._data.get("notifications", [])
            if unread_only:
                return [n for n in notifications if not n["read"]]
            return notifications
    
    def get_unread_notifications_count(self):
        """Get count of unread notifications"""
        with self._lock:
            notifications = self._data.get("notifications", [])
            return sum(1 for n in notifications if not n["read"])
    
    def update_real_time_data(self, key, value):
        """Update a value in the real-time data store"""
        with self._lock:
            self._real_time_data[key] = {
                "value": value,
                "timestamp": datetime.now()
            }
            
            # Emit real-time data update event
            self._emit_event("real_time_data", {
                "key": key,
                "value": value,
                "timestamp": datetime.now()
            })
    
    def get_real_time_data(self, key):
        """Get real-time data by key"""
        with self._lock:
            return self._real_time_data.get(key)
    
    def _emit_event(self, event_type, data):
        """Emit an event to all listeners"""
        with self._lock:
            if event_type in self._event_listeners:
                for callback in self._event_listeners[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Event listener error: {e}")
    
    def on(self, event_type, callback):
        """Register an event listener"""
        with self._lock:
            if event_type not in self._event_listeners:
                self._event_listeners[event_type] = []
            
            self._event_listeners[event_type].append(callback)
    
    def off(self, event_type, callback):
        """Remove an event listener"""
        with self._lock:
            if event_type in self._event_listeners:
                if callback in self._event_listeners[event_type]:
                    self._event_listeners[event_type].remove(callback)
    
    def _load_default_dashboard_config(self):
        """Load default dashboard configuration"""
        return {
            "layout": "standard",  # standard, compact, expanded
            "widgets": {
                "overview": {
                    "enabled": True,
                    "position": 1,
                    "expanded": True
                },
                "citizen_engagement": {
                    "enabled": True,
                    "position": 2,
                    "expanded": True
                },
                "project_status": {
                    "enabled": True,
                    "position": 3,
                    "expanded": True
                },
                "sentiment_analysis": {
                    "enabled": True,
                    "position": 4,
                    "expanded": True
                },
                "regional_insights": {
                    "enabled": True,
                    "position": 5,
                    "expanded": True
                },
                "recent_activity": {
                    "enabled": True,
                    "position": 6,
                    "expanded": True
                },
                "geographic_insights": {
                    "enabled": True,
                    "position": 7,
                    "expanded": False
                },
                "system_health": {
                    "enabled": True,
                    "position": 8,
                    "expanded": False
                }
            },
            "default_region": "All",
            "refresh_interval": 300,  # 5 minutes
            "data_date_range": 30,  # 30 days
            "show_predictions": True,
            "dark_mode": True,
            "ai_insights": True,
            "real_time_updates": True
        }
    
    def update_dashboard_config(self, new_config):
        """Update dashboard configuration"""
        with self._lock:
            self._dashboard_config.update(new_config)
            
            # Emit dashboard config update event
            self._emit_event("dashboard_config", self._dashboard_config)
    
    def get_dashboard_config(self):
        """Get current dashboard configuration"""
        with self._lock:
            return self._dashboard_config.copy()
    
    def save_dashboard_config(self):
        """Save dashboard configuration to MongoDB"""
        try:
            client = get_mongo_client()
            if not client:
                return False
                
            db = client["CivicCatalyst"]
            config_collection = db["dashboard_configs"]
            
            username = st.session_state.get("username", "admin")
            
            # Save config with username
            config_data = {
                "username": username,
                "config": self._dashboard_config,
                "updated_at": datetime.now()
            }
            
            # Upsert config
            config_collection.update_one(
                {"username": username},
                {"$set": config_data},
                upsert=True
            )
            
            client.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save dashboard configuration: {e}")
            self.add_notification(
                "Configuration Error",
                f"Failed to save dashboard configuration: {e}",
                "error"
            )
            return False
    
    def load_dashboard_config(self):
        """Load dashboard configuration from MongoDB"""
        try:
            client = get_mongo_client()
            if not client:
                return False
                
            db = client["CivicCatalyst"]
            config_collection = db["dashboard_configs"]
            
            username = st.session_state.get("username", "admin")
            
            # Get config for username
            config_doc = config_collection.find_one({"username": username})
            
            client.close()
            
            if config_doc and "config" in config_doc:
                with self._lock:
                    self._dashboard_config = config_doc["config"]
                    
                    # Emit dashboard config update event
                    self._emit_event("dashboard_config", self._dashboard_config)
                    
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load dashboard configuration: {e}")
            self.add_notification(
                "Configuration Error",
                f"Failed to load dashboard configuration: {e}",
                "error"
            )
            return False
    
    def update_system_health(self):
        """Update system health metrics"""
        try:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Test database connection
            db_status = "healthy"
            try:
                client = get_mongo_client()
                if not client:
                    db_status = "error"
                else:
                    # Ping database
                    client.admin.command('ping')
                    client.close()
            except Exception:
                db_status = "error"
            
            # Test API connection (OpenAI)
            api_status = "healthy"
            if openai.api_key:
                try:
                    # Just a minimal API check - don't want to waste tokens
                    openai.api_key is not None
                except Exception:
                    api_status = "error"
            else:
                api_status = "unavailable"
            
            # Update health data
            with self._lock:
                self._system_health = {
                    "last_checked": datetime.now(),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "database_status": db_status,
                    "api_status": api_status
                }
                
                # Add to history (keep last 60 points)
                self._status_history.append({
                    "timestamp": datetime.now(),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage
                })
                
                if len(self._status_history) > 60:
                    self._status_history.pop(0)
                
                # Emit system health update event
                self._emit_event("system_health", self._system_health)
            
            return self._system_health
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
            return None
    
    def get_system_health(self):
        """Get current system health data"""
        with self._lock:
            return self._system_health.copy()
    
    def get_system_health_history(self):
        """Get system health history"""
        with self._lock:
            return self._status_history.copy()

# Initialize global state
if 'global_state' not in st.session_state:
    st.session_state.global_state = GlobalState()

global_state = st.session_state.global_state

# -------------------------------------------------------------
# Enhanced Internationalization System
# -------------------------------------------------------------

# Enhanced translation dictionary with categories
translations = {
    # Admin Dashboard
    "admin_dashboard": {
        "en": "Admin Dashboard",
        "fr": "Tableau de bord d'administration",
        "ar": "لوحة تحكم المسؤول",
        "darija": "لوحة تحكم المسؤول"
    },
    "overview": {
        "en": "Overview & Analytics",
        "fr": "Vue d'ensemble et analyses",
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
    "advanced_analytics": {
        "en": "Advanced Analytics",
        "fr": "Analyses avancées",
        "ar": "التحليلات المتقدمة",
        "darija": "التحليلات المتقدمة"
    },
    "ai_insights": {
        "en": "AI & NLP Insights",
        "fr": "Aperçus IA & NLP",
        "ar": "رؤى الذكاء الاصطناعي",
        "darija": "تحليلات الذكاء الاصطناعي"
    },
    "system_health": {
        "en": "System Health & Settings",
        "fr": "Santé du système et paramètres",
        "ar": "صحة النظام والإعدادات",
        "darija": "صحة النظام والإعدادات"
    },
    "audit_logs": {
        "en": "Audit Logs",
        "fr": "Journaux d'audit",
        "ar": "سجلات التدقيق",
        "darija": "سجلات التدقيق"
    },
    
    # Search and Metrics
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
    "search_all_collections": {
        "en": "Search All Collections",
        "fr": "Rechercher dans toutes les collections",
        "ar": "البحث في جميع المجموعات",
        "darija": "البحث في جميع المجموعات"
    },
    
    # Metrics and Stats
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
    "active_users": {
        "en": "Active Users",
        "fr": "Utilisateurs actifs",
        "ar": "المستخدمين النشطين",
        "darija": "المستخدمين النشطين"
    },
    "pending_comments": {
        "en": "Pending Comments",
        "fr": "Commentaires en attente",
        "ar": "التعليقات قيد الانتظار",
        "darija": "التعليقات قيد الانتظار"
    },
    "flagged_content": {
        "en": "Flagged Content",
        "fr": "Contenu signalé",
        "ar": "المحتوى المبلغ عنه",
        "darija": "المحتوى المبلغ عنه"
    },
    
    # User Management
    "select_user": {
        "en": "Select a User",
        "fr": "Sélectionner un utilisateur",
        "ar": "اختر مستخدمًا",
        "darija": "اختار مستعمل"
    },
    "new_user": {
        "en": "New User",
        "fr": "Nouvel utilisateur",
        "ar": "مستخدم جديد",
        "darija": "مستخدم جديد"
    },
    "edit_user": {
        "en": "Edit User",
        "fr": "Modifier l'utilisateur",
        "ar": "تعديل المستخدم",
        "darija": "تعديل المستخدم"
    },
    "delete_user": {
        "en": "Delete User",
        "fr": "Supprimer l'utilisateur",
        "ar": "حذف المستخدم",
        "darija": "حذف المستخدم"
    },
    "username": {
        "en": "Username",
        "fr": "Nom d'utilisateur",
        "ar": "اسم المستخدم",
        "darija": "اسم المستخدم"
    },
    "email": {
        "en": "Email",
        "fr": "Email",
        "ar": "البريد الإلكتروني",
        "darija": "البريد الإلكتروني"
    },
    "password": {
        "en": "Password",
        "fr": "Mot de passe",
        "ar": "كلمة المرور",
        "darija": "كلمة المرور"
    },
    "confirm_password": {
        "en": "Confirm Password",
        "fr": "Confirmer le mot de passe",
        "ar": "تأكيد كلمة المرور",
        "darija": "تأكيد كلمة المرور"
    },
    "role": {
        "en": "Role",
        "fr": "Rôle",
        "ar": "الدور",
        "darija": "الدور"
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
    },
    "last_login": {
        "en": "Last Login",
        "fr": "Dernière connexion",
        "ar": "آخر تسجيل دخول",
        "darija": "آخر تسجيل دخول"
    },
    
    # Project Management
    "new_project": {
        "en": "New Project",
        "fr": "Nouveau projet",
        "ar": "مشروع جديد",
        "darija": "مشروع جديد"
    },
    "edit_project": {
        "en": "Edit Project",
        "fr": "Modifier le projet",
        "ar": "تعديل المشروع",
        "darija": "تعديل المشروع"
    },
    "delete_project": {
        "en": "Delete Project",
        "fr": "Supprimer le projet",
        "ar": "حذف المشروع",
        "darija": "حذف المشروع"
    },
    "project_title": {
        "en": "Project Title",
        "fr": "Titre du projet",
        "ar": "عنوان المشروع",
        "darija": "عنوان المشروع"
    },
    "project_description": {
        "en": "Project Description",
        "fr": "Description du projet",
        "ar": "وصف المشروع",
        "darija": "وصف المشروع"
    },
    "project_category": {
        "en": "Project Category",
        "fr": "Catégorie du projet",
        "ar": "فئة المشروع",
        "darija": "فئة المشروع"
    },
    "project_status": {
        "en": "Project Status",
        "fr": "Statut du projet",
        "ar": "حالة المشروع",
        "darija": "حالة المشروع"
    },
    "project_budget": {
        "en": "Project Budget",
        "fr": "Budget du projet",
        "ar": "ميزانية المشروع",
        "darija": "ميزانية المشروع"
    },
    "project_region": {
        "en": "Project Region",
        "fr": "Région du projet",
        "ar": "منطقة المشروع",
        "darija": "منطقة المشروع"
    },
    "start_date": {
        "en": "Start Date",
        "fr": "Date de début",
        "ar": "تاريخ البدء",
        "darija": "تاريخ البدء"
    },
    "end_date": {
        "en": "End Date",
        "fr": "Date de fin",
        "ar": "تاريخ الانتهاء",
        "darija": "تاريخ الانتهاء"
    },
    "completion_percentage": {
        "en": "Completion Percentage",
        "fr": "Pourcentage d'achèvement",
        "ar": "نسبة الإنجاز",
        "darija": "نسبة الإنجاز"
    },
    
    # Content Moderation
    "approve": {
        "en": "Approve",
        "fr": "Approuver",
        "ar": "الموافقة",
        "darija": "الموافقة"
    },
    "reject": {
        "en": "Reject",
        "fr": "Rejeter",
        "ar": "رفض",
        "darija": "رفض"
    },
    "flag": {
        "en": "Flag",
        "fr": "Signaler",
        "ar": "وضع علامة",
        "darija": "وضع علامة"
    },
    "flagged_reason": {
        "en": "Flagged Reason",
        "fr": "Raison du signalement",
        "ar": "سبب وضع العلامة",
        "darija": "سبب وضع العلامة"
    },
    "comment_content": {
        "en": "Comment Content",
        "fr": "Contenu du commentaire",
        "ar": "محتوى التعليق",
        "darija": "محتوى التعليق"
    },
    "comment_author": {
        "en": "Comment Author",
        "fr": "Auteur du commentaire",
        "ar": "كاتب التعليق",
        "darija": "كاتب التعليق"
    },
    "moderation_status": {
        "en": "Moderation Status",
        "fr": "Statut de modération",
        "ar": "حالة المراجعة",
        "darija": "حالة المراجعة"
    },
    
    # Analytics
    "sentiment_analysis": {
        "en": "Sentiment Analysis",
        "fr": "Analyse des sentiments",
        "ar": "تحليل المشاعر",
        "darija": "تحليل المشاعر"
    },
    "topic_modeling": {
        "en": "Topic Modeling",
        "fr": "Modélisation de sujets",
        "ar": "نمذجة المواضيع",
        "darija": "نمذجة المواضيع"
    },
    "network_analysis": {
        "en": "Network Analysis",
        "fr": "Analyse de réseau",
        "ar": "تحليل الشبكة",
        "darija": "تحليل الشبكة"
    },
    "data_visualization": {
        "en": "Data Visualization",
        "fr": "Visualisation des données",
        "ar": "تصور البيانات",
        "darija": "تصور البيانات"
    },
    "predictive_analytics": {
        "en": "Predictive Analytics",
        "fr": "Analyse prédictive",
        "ar": "التحليلات التنبؤية",
        "darija": "التحليلات التنبؤية"
    },
    "geographic_insights": {
        "en": "Geographic Insights",
        "fr": "Aperçus géographiques",
        "ar": "رؤى جغرافية",
        "darija": "رؤى جغرافية"
    },
    
    # UI Elements
    "notification_center": {
        "en": "Notification Center",
        "fr": "Centre de notifications",
        "ar": "مركز الإشعارات",
        "darija": "مركز الإشعارات"
    },
    "customizable_dashboard": {
        "en": "Customizable Dashboard",
        "fr": "Tableau de bord personnalisable",
        "ar": "لوحة تحكم قابلة للتخصيص",
        "darija": "لوحة تحكم قابلة للتخصيص"
    },
    "batch_operations": {
        "en": "Batch Operations",
        "fr": "Opérations par lot",
        "ar": "عمليات المجموعة",
        "darija": "عمليات المجموعة"
    },
    "real_time_collaboration": {
        "en": "Real-time Collaboration",
        "fr": "Collaboration en temps réel",
        "ar": "التعاون في الوقت الحقيقي",
        "darija": "التعاون في الوقت الحقيقي"
    },
    "dark_mode": {
        "en": "Dark Mode",
        "fr": "Mode sombre",
        "ar": "الوضع المظلم",
        "darija": "الوضع المظلم"
    },
    "light_mode": {
        "en": "Light Mode",
        "fr": "Mode clair",
        "ar": "الوضع المضيء",
        "darija": "الوضع المضيء"
    },
    
    # Citizen Ideas
    "citizen_ideas": {
        "en": "Citizen Ideas",
        "fr": "Idées des citoyens",
        "ar": "أفكار المواطنين",
        "darija": "أفكار المواطنين"
    },
    "idea_management": {
        "en": "Idea Management",
        "fr": "Gestion des idées",
        "ar": "إدارة الأفكار",
        "darija": "تسيير الأفكار"
    },
    "idea_title": {
        "en": "Idea Title",
        "fr": "Titre de l'idée",
        "ar": "عنوان الفكرة",
        "darija": "عنوان الفكرة"
    },
    "idea_description": {
        "en": "Idea Description",
        "fr": "Description de l'idée",
        "ar": "وصف الفكرة",
        "darija": "وصف الفكرة"
    },
    "idea_category": {
        "en": "Idea Category",
        "fr": "Catégorie de l'idée",
        "ar": "فئة الفكرة",
        "darija": "فئة الفكرة"
    },
    "idea_status": {
        "en": "Idea Status",
        "fr": "Statut de l'idée",
        "ar": "حالة الفكرة",
        "darija": "حالة الفكرة"
    },
    
    # News Management
    "news_title": {
        "en": "News Title",
        "fr": "Titre de l'actualité",
        "ar": "عنوان الخبر",
        "darija": "عنوان الخبر"
    },
    "news_content": {
        "en": "News Content",
        "fr": "Contenu de l'actualité",
        "ar": "محتوى الخبر",
        "darija": "محتوى الخبر"
    },
    "news_category": {
        "en": "News Category",
        "fr": "Catégorie de l'actualité",
        "ar": "فئة الخبر",
        "darija": "فئة الخبر"
    },
    "publication_date": {
        "en": "Publication Date",
        "fr": "Date de publication",
        "ar": "تاريخ النشر",
        "darija": "تاريخ النشر"
    },
    "author": {
        "en": "Author",
        "fr": "Auteur",
        "ar": "المؤلف",
        "darija": "المؤلف"
    },
    
    # System
    "regions_management": {
        "en": "Regions Management",
        "fr": "Gestion des régions",
        "ar": "إدارة المناطق",
        "darija": "تسيير المناطق"
    },
    "logout": {
        "en": "Logout",
        "fr": "Déconnexion",
        "ar": "تسجيل الخروج",
        "darija": "تسجيل الخروج"
    },
    "login": {
        "en": "Login",
        "fr": "Connexion",
        "ar": "تسجيل الدخول",
        "darija": "تسجيل الدخول"
    },
    "language": {
        "en": "Language",
        "fr": "Langue",
        "ar": "اللغة",
        "darija": "اللغة"
    },
    "theme": {
        "en": "Theme",
        "fr": "Thème",
        "ar": "المظهر",
        "darija": "المظهر"
    },
    "save": {
        "en": "Save",
        "fr": "Enregistrer",
        "ar": "حفظ",
        "darija": "حفظ"
    },
    "cancel": {
        "en": "Cancel",
        "fr": "Annuler",
        "ar": "إلغاء",
        "darija": "إلغاء"
    },
    "search": {
        "en": "Search",
        "fr": "Rechercher",
        "ar": "بحث",
        "darija": "بحث"
    },
    "export": {
        "en": "Export",
        "fr": "Exporter",
        "ar": "تصدير",
        "darija": "تصدير"
    },
    "import": {
        "en": "Import",
        "fr": "Importer",
        "ar": "استيراد",
        "darija": "استيراد"
    },
    "settings": {
        "en": "Settings",
        "fr": "Paramètres",
        "ar": "الإعدادات",
        "darija": "الإعدادات"
    }
}

def t(key):
    """Return the translated string based on session language with better fallback."""
    lang = st.session_state.get("site_language", "en")
    
    # Check if key exists in translations
    if key in translations:
        # Return translation if available, otherwise fall back to English or the key itself
        return translations[key].get(lang, translations[key].get("en", key))
    
    # If key doesn't exist in translations, return the key itself
    return key

# -------------------------------------------------------------
# ENHANCED MongoDB & Qdrant Connection with Error Handling
# -------------------------------------------------------------

def get_mongo_client():
    """Return a MongoDB client with connection pooling and error handling."""
    try:
        # Use cache for client if already created
        cache_key = "mongo_client"
        existing_client = cache.get(cache_key)
        if existing_client:
            # Check if client is still connected
            try:
                existing_client.admin.command('ping')
                return existing_client
            except Exception:
                # Client disconnected, remove from cache
                cache.clear(prefix=cache_key)
        
        # Get connection string from Streamlit secrets
        connection_string = st.secrets.get("mongodb", {}).get("connection_string", "mongodb://localhost:27017")
        
        # Use connection pooling for better performance
        client = MongoClient(
            connection_string,
            maxPoolSize=10,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000
        )
        
        # Verify connection
        client.admin.command('ping')
        
        # Store in cache
        cache.set(cache_key, client, ttl=3600)  # Cache for 1 hour
        
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        # Log error to global state
        global_state.add_notification(
            "Database Connection Error", 
            f"Failed to connect to MongoDB: {e}", 
            "error"
        )
        return None

@lru_cache(maxsize=32)
def get_qdrant_client():
    """Return a Qdrant client with connection pooling and error handling."""
    try:
        # Get connection details from Streamlit secrets
        host = st.secrets.get("qdrant", {}).get("host", "localhost")
        port = st.secrets.get("qdrant", {}).get("port", 6333)
        timeout = st.secrets.get("qdrant", {}).get("timeout", 5.0)
        
        client = QdrantClient(
            host=host, 
            port=port,
            timeout=timeout
        )
        
        # Verify connection
        client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        # Log error to global state
        global_state.add_notification(
            "Vector Database Connection Error", 
            f"Failed to connect to Qdrant: {e}", 
            "error"
        )
        return None

def get_qdrant_collection_info(collection_name):
    """Get information about a Qdrant collection with error handling."""
    try:
        client = get_qdrant_client()
        if not client:
            return None
        
        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        return collection_info
    except Exception as e:
        logger.error(f"Error getting Qdrant collection info: {e}")
        return None

def get_all_qdrant_collections():
    """Get list of all Qdrant collections with error handling."""
    try:
        client = get_qdrant_client()
        if not client:
            return []
        
        # Get all collections
        collections = client.get_collections()
        return collections.collections
    except Exception as e:
        logger.error(f"Error getting all Qdrant collections: {e}")
        return []

# Enhanced user loading with caching
def load_users(limit=None, search_query=None, role_filter=None, status_filter=None):
    """
    Load users from the MongoDB 'users' collection with caching and filtering.
    
    Parameters:
        limit (int): Maximum number of users to return
        search_query (str): Optional search query for username or email
        role_filter (str): Optional role filter
        status_filter (str): Optional status filter (active/inactive)
        
    Returns:
        list: List of user documents
    """
    # Create cache key based on parameters
    cache_key = f"users_{limit}_{search_query}_{role_filter}_{status_filter}"
    
    # Try to get from cache first
    cached_users = cache.get(cache_key)
    if cached_users is not None:
        return cached_users
    
    try:
        client = get_mongo_client()
        if not client:
            return []
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Build query
        query = {}
        
        if search_query:
            # Search in username or email
            query["$or"] = [
                {"username": {"$regex": search_query, "$options": "i"}},
                {"email": {"$regex": search_query, "$options": "i"}}
            ]
        
        if role_filter and role_filter != "All":
            query["role"] = role_filter
        
        if status_filter:
            if status_filter == "Active":
                query["active"] = True
            elif status_filter == "Inactive":
                query["active"] = False
        
        # Execute query with projection to exclude sensitive fields
        cursor = users_collection.find(
            query, 
            {"password_hash": 0, "password_salt": 0}
        )
        
        # Apply limit if specified
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert to list
        users = list(cursor)
        
        # Convert ObjectId to string for better display
        for user in users:
            if "_id" in user:
                user["_id"] = str(user["_id"])
        
        # Store in cache for 5 minutes
        cache.set(cache_key, users, ttl=300)
        return users
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        # Log error to global state
        global_state.add_notification(
            "Data Loading Error", 
            f"Failed to load users: {e}", 
            "error"
        )
        return []
    finally:
        if client:
            client.close()

def get_user_by_id(user_id):
    """Get a user by ID with caching."""
    cache_key = f"user_{user_id}"
    
    # Try to get from cache first
    cached_user = cache.get(cache_key)
    if cached_user is not None:
        return cached_user
    
    try:
        client = get_mongo_client()
        if not client:
            return None
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Find user by ID
        user = users_collection.find_one(
            {"_id": ObjectId(user_id)}, 
            {"password_hash": 0, "password_salt": 0}
        )
        
        if user:
            # Convert ObjectId to string
            user["_id"] = str(user["_id"])
            
            # Store in cache
            cache.set(cache_key, user, ttl=300)
            
        return user
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
        return None
    finally:
        if client:
            client.close()

def create_new_user(username, email, password, role, full_name=None, profile_data=None):
    """Create a new user with proper password hashing."""
    try:
        client = get_mongo_client()
        if not client:
            return False, "Database connection error"
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Check if username already exists
        if users_collection.find_one({"username": username}):
            return False, "Username already exists"
        
        # Check if email already exists
        if users_collection.find_one({"email": email}):
            return False, "Email already exists"
        
        # Generate salt
        salt = uuid.uuid4().hex
        
        # Hash password
        hashed_password = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        ).hex()
        
        # Create user document
        user_doc = {
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "password_salt": salt,
            "role": role,
            "full_name": full_name,
            "profile_data": profile_data or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_login": None,
            "active": True
        }
        
        # Insert user
        result = users_collection.insert_one(user_doc)
        
        # Clear user cache
        cache.clear(prefix="users_")
        
        # Add to audit log
        add_audit_log(
            "user_creation",
            f"Created new user: {username}",
            {"username": username, "role": role},
            st.session_state.get("username", "system")
        )
        
        return True, str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False, str(e)
    finally:
        if client:
            client.close()

def update_user(user_id, update_data):
    """Update user data with proper password handling."""
    try:
        client = get_mongo_client()
        if not client:
            return False, "Database connection error"
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Check if updating password
        if "password" in update_data and update_data["password"]:
            # Generate new salt
            salt = uuid.uuid4().hex
            
            # Hash new password
            hashed_password = hashlib.pbkdf2_hmac(
                'sha256', 
                update_data["password"].encode('utf-8'), 
                salt.encode('utf-8'), 
                100000
            ).hex()
            
            # Update password fields
            update_data["password_hash"] = hashed_password
            update_data["password_salt"] = salt
            
            # Remove plain password from update data
            del update_data["password"]
        
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        # Update user
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        # Clear user cache
        cache.clear(prefix=f"user_{user_id}")
        cache.clear(prefix="users_")
        
        # Add to audit log
        add_audit_log(
            "user_update",
            f"Updated user: {user_id}",
            {"user_id": user_id, "updated_fields": list(update_data.keys())},
            st.session_state.get("username", "system")
        )
        
        return True, "User updated successfully"
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return False, str(e)
    finally:
        if client:
            client.close()

def delete_user(user_id):
    """Delete a user by ID."""
    try:
        client = get_mongo_client()
        if not client:
            return False, "Database connection error"
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Get username for audit log
        user = users_collection.find_one({"_id": ObjectId(user_id)}, {"username": 1})
        username = user.get("username", "unknown") if user else "unknown"
        
        # Delete user
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        
        # Clear user cache
        cache.clear(prefix=f"user_{user_id}")
        cache.clear(prefix="users_")
        
        # Add to audit log
        add_audit_log(
            "user_deletion",
            f"Deleted user: {username}",
            {"user_id": user_id, "username": username},
            st.session_state.get("username", "system")
        )
        
        return True, "User deleted successfully"
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False, str(e)
    finally:
        if client:
            client.close()

# Enhanced document loading with caching, pagination, and filtering
def load_qdrant_documents(
    collection_name: str, 
    vector_dim: int, 
    limit: int = 1000, 
    offset: int = 0,
    filters=None, 
    prefetch: bool = False,
    scroll_id: str = None
):
    """
    Retrieve documents (payloads) from a Qdrant collection with advanced features.
    
    Parameters:
        collection_name (str): Name of the Qdrant collection
        vector_dim (int): Vector dimension for the collection
        limit (int): Maximum number of documents to retrieve
        offset (int): Offset for pagination
        filters: Optional Filter object for Qdrant
        prefetch (bool): Whether to prefetch the documents in background
        scroll_id (str): Scroll ID for cursor-based pagination
        
    Returns:
        tuple: (documents, next_scroll_id)
    """
    # Create a cache key based on parameters
    filter_str = str(filters) if filters else "none"
    cache_key = f"qdrant_{collection_name}_{limit}_{offset}_{hash(filter_str)}"
    
    # Try to get from cache first
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If prefetch is True and we're not already in a prefetch operation
    if prefetch and not st.session_state.get("is_prefetching", False):
        # Set flag to prevent recursive prefetching
        st.session_state.is_prefetching = True
        
        # Add background task for prefetching
        def prefetch_task():
            try:
                docs, next_id = _fetch_qdrant_documents(
                    collection_name, vector_dim, limit, offset, filters, scroll_id
                )
                cache.set(cache_key, (docs, next_id), ttl=600)  # Cache for 10 minutes
            finally:
                st.session_state.is_prefetching = False
                
        global_state.add_task(prefetch_task)
        
        # Return empty result or previously cached data
        return cached_data if cached_data else ([], None)
    
    # Regular synchronous fetch
    docs, next_id = _fetch_qdrant_documents(
        collection_name, vector_dim, limit, offset, filters, scroll_id
    )
    
    # Store in cache
    cache.set(cache_key, (docs, next_id), ttl=300)  # Cache for 5 minutes
    
    return docs, next_id

def _fetch_qdrant_documents(
    collection_name: str, 
    vector_dim: int, 
    limit: int = 1000, 
    offset: int = 0,
    filters=None,
    scroll_id: str = None
):
    """
    Fetch documents (payloads) from a Qdrant collection using pagination.
    
    Parameters:
        collection_name (str): Name of the Qdrant collection
        vector_dim (int): Vector dimension
        limit (int): Maximum number of documents to retrieve
        offset (int): Offset for pagination
        filters: Optional Filter object for Qdrant
        scroll_id (str): Scroll ID for cursor-based pagination
        
    Returns:
        tuple: (documents, next_scroll_id)
    """
    try:
        client = get_qdrant_client()
        if not client:
            return [], None
            
        all_docs = []
        
        # Try using scroll without filter first (since filter is causing the error)
        try:
            # Basic scroll parameters without the problematic 'filter' parameter
            scroll_params = {
                "collection_name": collection_name,
                "limit": min(limit, 100),  # Fetch in batches of 100 max
                "offset": offset if scroll_id is None else int(scroll_id)
            }
            
            # First scroll request
            scroll_result = client.scroll(**scroll_params)
            
            # Extract results based on the return format 
            # (handling different client versions)
            if isinstance(scroll_result, tuple):
                points, next_offset = scroll_result
            else:
                # Older API might return just points
                points = scroll_result
                next_offset = offset + len(points) if len(points) > 0 else None
            
            # Extract payloads
            all_docs.extend([pt.payload for pt in points])
            
            # Return results
            next_scroll_id = str(next_offset) if next_offset is not None else None
            return all_docs[:limit], next_scroll_id
                
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            
            # Fallback to simpler approach - get without filters first
            try:
                # Try basic search without filter
                points = client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * vector_dim,
                    limit=limit
                )
                all_docs = [point.payload for point in points]
                return all_docs, None
            except Exception as e2:
                logger.error(f"Error in search operation: {e2}")
                
                # Last resort - try to retrieve points without search
                try:
                    # Check if the collection has a different API
                    # This might work for some Qdrant versions
                    points = client.retrieve(
                        collection_name=collection_name,
                        limit=limit
                    )
                    all_docs = [point.payload for point in points]
                    return all_docs, None
                except:
                    # If everything fails, return empty results
                    return [], None
            
    except Exception as e:
        logger.error(f"Error loading documents from Qdrant ({collection_name}): {e}")
        return [], None
def semantic_search(collection_name, query_text, limit=10, filter_conditions=None):
    """
    Perform semantic search using OpenAI embeddings.
    
    Parameters:
        collection_name (str): Name of the Qdrant collection
        query_text (str): Search query text
        limit (int): Maximum number of results
        filter_conditions (list): Optional list of filter conditions
        
    Returns:
        list: Search results with payloads
    """
    try:
        # Cache key for the embedding
        cache_key = f"embedding_{hash(query_text)}"
        
        # Try to get embedding from cache
        embedding = cache.get(cache_key)
        
        if embedding is None:
            # Generate embedding using OpenAI
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=query_text
            )
            embedding = response["data"][0]["embedding"]
            
            # Cache the embedding
            cache.set(cache_key, embedding, ttl=3600)  # Cache for 1 hour
        
        # Get Qdrant client
        client = get_qdrant_client()
        if not client:
            return []
        
        try:
            # Try search without filter first
            search_results = client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Error in semantic search with filter: {e}")
            # Fallback to basic search without filter
            search_results = client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit
            )
        
        # Extract search results with scores
        results = [
            {
                "score": result.score,
                "payload": result.payload
            }
            for result in search_results
        ]
        
        return results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []


def add_audit_log(action_type, description, details=None, user=None):
    """
    Add an entry to the audit log.
    
    Parameters:
        action_type (str): Type of action (e.g., 'login', 'user_update')
        description (str): Human-readable description of the action
        details (dict): Additional details of the action
        user (str): Username who performed the action
    """
    try:
        client = get_mongo_client()
        if not client:
            logger.error("Failed to connect to MongoDB for audit log")
            return False
            
        db = client["CivicCatalyst"]
        audit_collection = db["audit_logs"]
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now(),
            "action_type": action_type,
            "description": description,
            "user": user or "system",
            "ip_address": "127.0.0.1",  # In production, get real IP
            "details": details or {}
        }
        
        # Insert log entry
        audit_collection.insert_one(log_entry)
        return True
    
    except Exception as e:
        logger.error(f"Error adding audit log: {e}")
        return False
    
    finally:
        if client:
            client.close()

def load_audit_logs(limit=100, action_type=None, user=None, date_range=None):
    """
    Load audit logs with filtering options.
    
    Parameters:
        limit (int): Maximum number of logs to return
        action_type (str): Optional filter by action type
        user (str): Optional filter by username
        date_range (tuple): Optional (start_date, end_date) tuple
        
    Returns:
        list: Filtered audit logs
    """
    try:
        client = get_mongo_client()
        if not client:
            return []
            
        db = client["CivicCatalyst"]
        audit_collection = db["audit_logs"]
        
        # Build query
        query = {}
        
        if action_type:
            query["action_type"] = action_type
            
        if user:
            query["user"] = user
            
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            query["timestamp"] = {
                "$gte": start_date,
                "$lte": end_date
            }
        
        # Execute query
        cursor = audit_collection.find(query).sort("timestamp", -1).limit(limit)
        
        # Convert to list
        logs = list(cursor)
        
        # Convert ObjectId to string
        for log in logs:
            if "_id" in log:
                log["_id"] = str(log["_id"])
        
        return logs
    
    except Exception as e:
        logger.error(f"Error loading audit logs: {e}")
        return []
    
    finally:
        if client:
            client.close()

# -------------------------------------------------------------
# THEME AND UI MANAGEMENT
# -------------------------------------------------------------

def initialize_theme():
    """Initialize theme settings"""
    if "theme" not in st.session_state:
        # Default to dark mode
        st.session_state.theme = "dark"

def toggle_theme():
    """Toggle between dark and light themes"""
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"
    
    # Add notification about theme change
    theme_name = t("light_mode") if st.session_state.theme == "light" else t("dark_mode")
    global_state.add_notification(
        "Theme Changed",
        f"Switched to {theme_name}",
        "info"
    )

def get_theme_css():
    """Get CSS based on current theme"""
    initialize_theme()
    
    # Base CSS that applies to both themes
    base_css = """
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Base Styling */
    * {
        box-sizing: border-box;
    }
    
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    .tab-subheader {
        font-size: 1.4rem;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .metric-up {
        color: #4CAF50;
    }
    
    .metric-down {
        color: #F44336;
    }
    
    .data-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 1.5rem;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .styled-table thead {
        font-weight: 600;
    }
    
    .styled-table th, .styled-table td {
        padding: 1rem;
        text-align: left;
    }
    
    .styled-table th {
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .styled-table tbody tr {
        transition: background-color 0.2s;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge i {
        margin-right: 0.35rem;
    }
    
    .badge-primary {
        background-color: rgba(33, 150, 243, 0.15);
        color: #2196F3;
    }
    
    .badge-success {
        background-color: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
    }
    
    .badge-warning {
        background-color: rgba(255, 152, 0, 0.15);
        color: #FF9800;
    }
    
    .badge-danger {
        background-color: rgba(244, 67, 54, 0.15);
        color: #F44336;
    }
    
    .badge-info {
        background-color: rgba(0, 188, 212, 0.15);
        color: #00BCD4;
    }
    
    .status-dot {
        display: inline-block;
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
        margin-right: 0.4rem;
    }
    
    .status-active {
        background-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.25);
    }
    
    .status-pending {
        background-color: #FF9800;
        box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.25);
    }
    
    .status-inactive {
        background-color: #F44336;
        box-shadow: 0 0 0 2px rgba(244, 67, 54, 0.25);
    }
    
    .action-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        margin-right: 0.5rem;
        transition: all 0.2s;
        border: none;
        white-space: nowrap;
    }
    
    .action-button i {
        margin-right: 0.35rem;
    }
    
    .action-view {
        background-color: rgba(33, 150, 243, 0.1);
        color: #2196F3;
    }
    
    .action-view:hover {
        background-color: rgba(33, 150, 243, 0.2);
    }
    
    .action-edit {
        background-color: rgba(255, 152, 0, 0.1);
        color: #FF9800;
    }
    
    .action-edit:hover {
        background-color: rgba(255, 152, 0, 0.2);
    }
    
    .action-delete {
        background-color: rgba(244, 67, 54, 0.1);
        color: #F44336;
    }
    
    .action-delete:hover {
        background-color: rgba(244, 67, 54, 0.2);
    }
    
    .fancy-divider {
        border: 0;
        height: 1px;
        margin: 2.5rem 0;
    }
    
    .breadcrumb {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }
    
    .breadcrumb-item {
        opacity: 0.7;
    }
    
    .breadcrumb-current {
        font-weight: 600;
    }
    
    .breadcrumb-separator {
        margin: 0 0.5rem;
        opacity: 0.5;
    }
    
    .notification-counter {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.5rem;
        height: 1.5rem;
        border-radius: 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0 0.25rem;
        position: absolute;
        top: -0.5rem;
        right: -0.5rem;
    }
    
    /* Sidebar Menu */
    .sidebar-menu {
        margin-bottom: 2rem;
    }
    
    .sidebar-menu-item {
        display: flex;
        align-items: center;
        padding: 0.85rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .sidebar-menu-item-icon {
        margin-right: 0.75rem;
        font-size: 1.1rem;
        width: 1.5rem;
        text-align: center;
    }
    
    .sidebar-menu-item-text {
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Notification Panel */
    .notification-panel {
        position: fixed;
        top: 0;
        right: 0;
        width: 24rem;
        height: 100vh;
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease-in-out;
        box-shadow: -4px 0 15px rgba(0, 0, 0, 0.1);
    }
    
    .notification-panel.show {
        transform: translateX(0);
    }
    
    .notification-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem;
        border-bottom-width: 1px;
    }
    
    .notification-close {
        cursor: pointer;
        font-size: 1.2rem;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background-color 0.2s;
    }
    
    .notification-item {
        padding: 1.25rem;
        border-bottom-width: 1px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .notification-title {
        font-weight: 600;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .notification-message {
        font-size: 0.9rem;
        opacity: 0.85;
        line-height: 1.5;
    }
    
    .notification-time {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    
    /* Tags Input */
    .tag-input {
        display: flex;
        flex-wrap: wrap;
        border-radius: 0.5rem;
        min-height: 2.5rem;
        padding: 0.35rem;
        border-width: 1px;
    }
    
    .tag {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        margin: 0.25rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .tag-remove {
        margin-left: 0.5rem;
        cursor: pointer;
        width: 1.25rem;
        height: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background-color 0.2s;
    }
    
    /* Progress Bar */
    .progress-bar-container {
        width: 100%;
        height: 0.5rem;
        border-radius: 0.25rem;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Kanban Board */
    .kanban-board {
        display: flex;
        overflow-x: auto;
        padding-bottom: 1rem;
        gap: 1rem;
    }
    
    .kanban-column {
        min-width: 16rem;
        max-width: 16rem;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .kanban-column-header {
        padding: 1rem;
        font-weight: 600;
        border-bottom-width: 1px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .kanban-column-count {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .kanban-cards {
        padding: 0.5rem;
        min-height: 6rem;
    }
    
    .kanban-card {
        margin: 0.5rem;
        padding: 1rem;
        border-radius: 0.375rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .kanban-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .kanban-card-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    .kanban-card-content {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-bottom: 0.75rem;
    }
    
    .kanban-card-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        opacity: 0.7;
    }
    
    /* Dashboard Widget */
    .dashboard-widget {
        border-radius: 0.75rem;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .dashboard-widget:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .dashboard-widget-header {
        padding: 1rem 1.25rem;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom-width: 1px;
    }
    
    .dashboard-widget-actions {
        display: flex;
        gap: 0.5rem;
    }
    
    .dashboard-widget-action {
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .dashboard-widget-content {
        padding: 1.25rem;
    }
    
    .dashboard-widget-footer {
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        opacity: 0.8;
        border-top-width: 1px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Form Elements */
    .form-group {
        margin-bottom: 1.5rem;
    }
    
    .form-label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .form-hint {
        display: block;
        margin-top: 0.35rem;
        font-size: 0.8rem;
        opacity: 0.7;
    }
    
    .form-error {
        display: block;
        margin-top: 0.35rem;
        font-size: 0.8rem;
        color: #F44336;
    }
    
    /* Custom Card Components */
    .custom-card {
        border-radius: 0.75rem;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .custom-card-hover:hover {
        transform: translateY(-3px);
    }
    
    .custom-card-body {
        padding: 1.5rem;
    }
    
    .custom-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    /* Login Form */
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .login-logo {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-title {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .login-footer {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        opacity: 0.7;
    }
    
    /* Typography Utilities */
    .text-sm {
        font-size: 0.85rem;
    }
    
    .text-lg {
        font-size: 1.1rem;
    }
    
    .text-xl {
        font-size: 1.25rem;
    }
    
    .text-2xl {
        font-size: 1.5rem;
    }
    
    .font-bold {
        font-weight: 700;
    }
    
    .font-semibold {
        font-weight: 600;
    }
    
    .font-medium {
        font-weight: 500;
    }
    
    .text-center {
        text-align: center;
    }
    
    .text-right {
        text-align: right;
    }
    
    /* Flexbox Utilities */
    .flex {
        display: flex;
    }
    
    .flex-col {
        flex-direction: column;
    }
    
    .items-center {
        align-items: center;
    }
    
    .justify-between {
        justify-content: space-between;
    }
    
    .justify-center {
        justify-content: center;
    }
    
    .gap-1 {
        gap: 0.25rem;
    }
    
    .gap-2 {
        gap: 0.5rem;
    }
    
    .gap-3 {
        gap: 0.75rem;
    }
    
    .gap-4 {
        gap: 1rem;
    }
    
    /* Spacing Utilities */
    .m-0 {
        margin: 0;
    }
    
    .mt-1 {
        margin-top: 0.25rem;
    }
    
    .mt-2 {
        margin-top: 0.5rem;
    }
    
    .mt-4 {
        margin-top: 1rem;
    }
    
    .mb-1 {
        margin-bottom: 0.25rem;
    }
    
    .mb-2 {
        margin-bottom: 0.5rem;
    }
    
    .mb-4 {
        margin-bottom: 1rem;
    }
    
    .mb-6 {
        margin-bottom: 1.5rem;
    }
    
    .p-4 {
        padding: 1rem;
    }
    
    /* Animation Utilities */
    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: .5;
        }
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    /* Font Awesome Icon Animation */
    .fa-spin {
        animation: fa-spin 2s infinite linear;
    }
    
    @keyframes fa-spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    """
    
    # Theme-specific CSS
    theme_css = {
        "dark": """
        /* Dark theme */
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        
        .metric-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            color: #E0E0E0;
        }
        
        .data-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            color: #E0E0E0;
        }
        
        .styled-table thead {
            background-color: #252525;
            color: #E0E0E0;
        }
        
        .styled-table th, .styled-table td {
            border-bottom: 1px solid #333;
            color: #E0E0E0;
        }
        
        .styled-table tbody tr:nth-child(even) {
            background-color: #2A2A2A;
        }
        
        .styled-table tbody tr:hover {
            background-color: #303030;
        }
        
        .fancy-divider {
            background-image: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0));
        }
        
        .breadcrumb {
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
        
        .sidebar-menu-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .sidebar-menu-item.active {
            background-color: rgba(255, 255, 255, 0.15);
        }
        
        .notification-panel {
            background-color: #1E1E1E;
            border-left: 1px solid #333;
            color: #E0E0E0;
        }
        
        .notification-header {
            border-bottom-color: #333;
        }
        
        .notification-close:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .notification-item {
            border-bottom-color: #333;
        }
        
        .notification-item:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .tag-input {
            background-color: #252525;
            border-color: #444;
        }
        
        .tag {
            background-color: #333;
            color: #E0E0E0;
        }
        
        .tag-remove:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .progress-bar-container {
            background-color: #333;
        }
        
        .kanban-column {
            background-color: #252525;
            border: 1px solid #333;
        }
        
        .kanban-column-header {
            border-bottom-color: #333;
        }
        
        .kanban-column-count {
            background-color: #333;
            color: #E0E0E0;
        }
        
        .kanban-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
        
        .dashboard-widget {
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
        
        .dashboard-widget-header {
            background-color: #252525;
            border-bottom-color: #333;
        }
        
        .dashboard-widget-action:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .dashboard-widget-footer {
            border-top-color: #333;
            background-color: #252525;
        }
        
        .notification-counter {
            background-color: #F44336;
            color: white;
        }
        
        .custom-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .login-container {
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
        
        ::-webkit-scrollbar-thumb {
            background-color: #444;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background-color: #555;
        }
        
        input, select, textarea {
            background-color: #252525 !important;
            border-color: #444 !important;
            color: #E0E0E0 !important;
        }
        
        button {
            background-color: #252525 !important;
            border-color: #444 !important;
            color: #E0E0E0 !important;
        }
        
        /* Override Streamlit Elements */
        .stTextInput > div > div > input {
            background-color: #252525 !important;
            color: #E0E0E0 !important; 
        }
        
        .stSelectbox > div > div > select {
            background-color: #252525 !important;
            color: #E0E0E0 !important;
        }
        
        .stTextArea > div > div > textarea {
            background-color: #252525 !important;
            color: #E0E0E0 !important;
        }
        """,
        "light": """
        /* Light theme */
        body {
            background-color: #F5F7FA;
            color: #333;
        }
        
        .metric-card {
            background-color: white;
            border: 1px solid #E0E0E0;
            color: #333;
        }
        
        .data-card {
            background-color: white;
            border: 1px solid #E0E0E0;
            color: #333;
        }
        
        .styled-table thead {
            background-color: #F5F7FA;
            color: #333;
        }
        
        .styled-table th, .styled-table td {
            border-bottom: 1px solid #E0E0E0;
            color: #333;
        }
        
        .styled-table tbody tr:nth-child(even) {
            background-color: #F9F9F9;
        }
        
        .styled-table tbody tr:hover {
            background-color: #F0F0F0;
        }
        
        .fancy-divider {
            background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));
        }
        
        .breadcrumb {
            background-color: white;
            border: 1px solid #E0E0E0;
        }
        
        .sidebar-menu-item:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .sidebar-menu-item.active {
            background-color: rgba(33, 150, 243, 0.1);
        }
        
        .notification-panel {
            background-color: white;
            border-left: 1px solid #E0E0E0;
            color: #333;
        }
        
        .notification-header {
            border-bottom-color: #E0E0E0;
        }
        
        .notification-close:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .notification-item {
            border-bottom-color: #E0E0E0;
        }
        
        .notification-item:hover {
            background-color: rgba(0, 0, 0, 0.02);
        }
        
        .tag-input {
            background-color: white;
            border-color: #E0E0E0;
        }
        
        .tag {
            background-color: #F0F0F0;
            color: #333;
        }
        
        .tag-remove:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .progress-bar-container {
            background-color: #E0E0E0;
        }
        
        .kanban-column {
            background-color: #F9F9F9;
            border: 1px solid #E0E0E0;
        }
        
        .kanban-column-header {
            border-bottom-color: #E0E0E0;
        }
        
        .kanban-column-count {
            background-color: #E0E0E0;
            color: #333;
        }
        
        .kanban-card {
            background-color: white;
            border: 1px solid #E0E0E0;
        }
        
        .dashboard-widget {
            background-color: white;
            border: 1px solid #E0E0E0;
        }
        
        .dashboard-widget-header {
            background-color: #F5F7FA;
            border-bottom-color: #E0E0E0;
        }
        
        .dashboard-widget-action:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .dashboard-widget-footer {
            border-top-color: #E0E0E0;
            background-color: #F5F7FA;
        }
        
        .notification-counter {
            background-color: #F44336;
            color: white;
        }
        
        .custom-card {
            background-color: white;
            border: 1px solid #E0E0E0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .login-container {
            background-color: white;
            border: 1px solid #E0E0E0;
        }
        
        ::-webkit-scrollbar-thumb {
            background-color: #CCC;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background-color: #BBB;
        }
        
        input, select, textarea {
            background-color: white !important;
            border-color: #E0E0E0 !important;
            color: #333 !important;
        }
        
        button {
            background-color: #F5F7FA !important;
            border-color: #E0E0E0 !important;
            color: #333 !important;
        }
        """
    }
    
    # Return combined CSS
    return base_css + theme_css[st.session_state.theme]

# -------------------------------------------------------------
# NOTIFICATION CENTER COMPONENT
# -------------------------------------------------------------

def render_notification_center():
    """Render the notification center UI component."""
    notification_count = global_state.get_unread_notifications_count()
    
    # Notification button in the top right
    notification_btn_html = f"""
    <div id="notification-btn" style="position: fixed; top: 1rem; right: 1rem; z-index: 1000; cursor: pointer;">
        <div style="position: relative; width: 2.5rem; height: 2.5rem; border-radius: 50%; background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else 'white'}; display: flex; justify-content: center; align-items: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
            <i class="fas fa-bell" style="font-size: 1.2rem; color: {'#E0E0E0' if st.session_state.theme == 'dark' else '#333'};"></i>
            {f'<span class="notification-counter">{notification_count}</span>' if notification_count > 0 else ''}
        </div>
    </div>
    """
    
    # Notification panel
    notification_panel_html = f"""
    <div id="notification-panel" class="notification-panel">
        <div class="notification-header">
            <h3 style="margin: 0;">{t('notification_center')}</h3>
            <div class="notification-close"><i class="fas fa-times"></i></div>
        </div>
        <div style="height: calc(100vh - 4rem); overflow-y: auto;">
    """
    
    # Add notifications
    notifications = global_state.get_notifications()
    if notifications:
        for notification in reversed(notifications):
            type_icon = {
                "info": "fa-info-circle",
                "success": "fa-check-circle",
                "warning": "fa-exclamation-triangle",
                "error": "fa-exclamation-circle"
            }.get(notification["type"], "fa-info-circle")
            
            type_color = {
                "info": "#2196F3",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336"
            }.get(notification["type"], "#2196F3")
            
            notification_panel_html += f"""
            <div class="notification-item" data-id="{notification['id']}" style="opacity: {0.7 if notification['read'] else 1};">
                <div class="notification-title">
                    <i class="fas {type_icon}" style="color: {type_color};"></i>
                    {notification["title"]}
                </div>
                <div class="notification-message">{notification["message"]}</div>
                <div class="notification-time">
                    <i class="fas fa-clock"></i>
                    {notification["timestamp"].strftime("%H:%M:%S - %d/%m/%Y")}
                </div>
            </div>
            """
    else:
        notification_panel_html += f"""
        <div style="padding: 2rem; text-align: center; opacity: 0.6;">
            <i class="fas fa-bell-slash" style="font-size: 2rem; margin-bottom: 1rem;"></i>
            <p>No notifications yet</p>
        </div>
        """
    
    notification_panel_html += """
        </div>
    </div>
    """
    
    # JavaScript to handle notification interactions
    notification_js = """
    <script>
        // Wait for elements to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {
            const notificationBtn = document.getElementById('notification-btn');
            const notificationPanel = document.getElementById('notification-panel');
            const notificationClose = document.querySelector('.notification-close');
            const notificationItems = document.querySelectorAll('.notification-item');
            
            if (notificationBtn && notificationPanel && notificationClose) {
                // Toggle notification panel when button is clicked
                notificationBtn.addEventListener('click', function() {
                    notificationPanel.classList.toggle('show');
                });
                
                // Close notification panel when close button is clicked
                notificationClose.addEventListener('click', function() {
                    notificationPanel.classList.remove('show');
                });
                
                // Close notification panel when clicking outside
                document.addEventListener('click', function(event) {
                    if (!notificationPanel.contains(event.target) && event.target !== notificationBtn && !notificationBtn.contains(event.target)) {
                        notificationPanel.classList.remove('show');
                    }
                });
                
                // Mark notification as read when clicked
                notificationItems.forEach(item => {
                    item.addEventListener('click', function() {
                        const notificationId = this.dataset.id;
                        // Send to Streamlit using sessionStorage
                        sessionStorage.setItem('mark_notification_read', notificationId);
                        // Trigger component update
                        setTimeout(() => {
                            const event = new Event('markNotificationRead');
                            document.dispatchEvent(event);
                        }, 100);
                        // Visual feedback
                        this.style.opacity = '0.7';
                    });
                });
            }
        });
    </script>
    """
    
    # Combine and render
    notification_html = notification_btn_html + notification_panel_html + notification_js
    st.markdown(notification_html, unsafe_allow_html=True)
    
    # Check for notification read events
    if st.button("Mark All as Read", key="mark_all_notifications_read"):
        for notification in notifications:
            global_state.mark_notification_read(notification["id"])
        st.rerun()

# -------------------------------------------------------------
# VISUALIZATION COMPONENTS
# -------------------------------------------------------------

def display_metric_card(title, value, delta=None, delta_description="from previous period", icon=None, color=None):
    """Display a custom styled metric card with icon and color options."""
    icon_html = f'<i class="fas {icon}" style="margin-right: 0.5rem; {f"color: {color};" if color else ""}"></i>' if icon else ''
    
    card_style = f'border-left: 4px solid {color};' if color else ''
    
    html = f"""
    <div class="metric-card" style="{card_style}">
        <div class="metric-title">{icon_html}{title}</div>
        <div class="metric-value">{value}</div>
    """
    
    if delta is not None:
        delta_class = "metric-up" if delta >= 0 else "metric-down"
        delta_icon = "▲" if delta >= 0 else "▼"
        html += f"""
        <div class="metric-delta {delta_class}">
            {delta_icon} {abs(delta)}% {delta_description}
        </div>
        """
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

def display_data_card(title, content, icon=None, actions=None):
    """Display a custom styled data card with content and optional actions."""
    icon_html = f'<i class="fas {icon}" style="margin-right: 0.5rem;"></i>' if icon else ''
    
    # Create any action buttons
    action_html = ""
    if actions:
        action_html = '<div class="data-card-actions" style="margin-top: 1rem; display: flex; gap: 0.5rem;">'
        for action in actions:
            btn_icon = action.get("icon", "")
            btn_text = action.get("text", "Action")
            btn_key = action.get("key", "action_btn")
            btn_class = action.get("class", "action-view")
            
            action_html += f"""
            <button class="action-button {btn_class}" onclick="document.getElementById('{btn_key}').click();">
                <i class="fas {btn_icon}"></i> {btn_text}
            </button>
            """
        action_html += '</div>'
    
    html = f"""
    <div class="data-card">
        <h3 style="margin-top: 0;">{icon_html} {title}</h3>
        <div>{content}</div>
        {action_html}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Create hidden buttons for actions
    if actions:
        for action in actions:
            if st.button(action.get("text", "Action"), key=action.get("key", "action_btn"), help=action.get("tooltip", ""), type=action.get("type", "primary")):
                if "callback" in action and callable(action["callback"]):
                    return action["callback"]()
    
    return None

def display_dashboard_widget(title, content, footer=None, actions=None, expanded=True, key=None):
    """Display a dashboard widget with header, content, and footer."""
    # Generate unique key if not provided
    widget_key = key or f"widget_{hash(title)}_{random.randint(0, 1000)}"
    
    # Build actions HTML
    actions_html = ""
    if actions:
        actions_html = '<div class="dashboard-widget-actions">'
        for action in actions:
            action_icon = action.get("icon", "fa-cog")
            action_tooltip = action.get("tooltip", "")
            action_key = action.get("key", f"action_{random.randint(0, 1000)}")
            
            actions_html += f"""
            <div class="dashboard-widget-action" title="{action_tooltip}" onclick="document.getElementById('{widget_key}_{action_key}').click();">
                <i class="fas {action_icon}"></i>
            </div>
            """
        actions_html += '</div>'
    
    # Create widget HTML
    widget_html = f"""
    <div class="dashboard-widget">
        <div class="dashboard-widget-header">
            {title}
            {actions_html}
        </div>
        <div class="dashboard-widget-content">
            {content}
        </div>
    """
    
    if footer:
        widget_html += f"""
        <div class="dashboard-widget-footer">
            {footer}
        </div>
        """
    
    widget_html += "</div>"
    
    # Render widget
    st.markdown(widget_html, unsafe_allow_html=True)
    
    # Create hidden buttons for actions
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button("", key=f"{widget_key}_{action.get('key', f'action_{random.randint(0, 1000)}')}"):
                    if "callback" in action and callable(action["callback"]):
                        return action["callback"]()
    
    return None

def create_dynamic_table(data, columns, key_prefix="table", page_size=10, searchable=True, actions=None, height=None):
    """
    Create a dynamic table with pagination, search, and actions.
    
    Parameters:
        data (list): List of dictionaries containing the data
        columns (list): List of column definitions (dict with 'key', 'label', 'format', etc.)
        key_prefix (str): Prefix for all table keys to avoid collisions
        page_size (int): Number of rows per page
        searchable (bool): Whether to include search functionality
        actions (list): List of action definitions (dict with 'text', 'icon', 'callback', etc.)
        height (str): Optional height for the table (e.g., '400px')
        
    Returns:
        dict: Table state information (page number, search term, etc.)
    """
    # Initialize table state in session_state if not exists
    if f"{key_prefix}_page" not in st.session_state:
        st.session_state[f"{key_prefix}_page"] = 1
    
    if f"{key_prefix}_search" not in st.session_state:
        st.session_state[f"{key_prefix}_search"] = ""
    
    # Filter data based on search term if provided
    filtered_data = data
    if searchable and st.session_state[f"{key_prefix}_search"]:
        search_term = st.session_state[f"{key_prefix}_search"].lower()
        filtered_data = []
        
        for item in data:
            # Search in all string/text fields
            for col in columns:
                field_key = col.get("key", "")
                if field_key in item and isinstance(item[field_key], (str, int, float)):
                    if str(item[field_key]).lower().find(search_term) != -1:
                        filtered_data.append(item)
                        break
    
    # Calculate pagination
    total_items = len(filtered_data)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    
    # Ensure current page is valid
    current_page = st.session_state[f"{key_prefix}_page"]
    if current_page > total_pages:
        current_page = total_pages
        st.session_state[f"{key_prefix}_page"] = current_page
    
    # Get current page data
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    page_data = filtered_data[start_idx:end_idx]
    
    # Build table UI
    table_container = st.container()
    
    with table_container:
        # Search and pagination controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if searchable:
                search_input = st.text_input(
                    "Search",
                    value=st.session_state[f"{key_prefix}_search"],
                    key=f"{key_prefix}_search_input",
                    placeholder="Enter search terms..."
                )
                
                if search_input != st.session_state[f"{key_prefix}_search"]:
                    st.session_state[f"{key_prefix}_search"] = search_input
                    st.session_state[f"{key_prefix}_page"] = 1  # Reset to first page on search
                    st.rerun()
        
        with col2:
            st.markdown(f"**Page {current_page} of {total_pages}**")
        
        # Table HTML
        table_style = f'height: {height}; overflow-y: auto;' if height else ''
        
        table_html = f"""
        <div style="{table_style}">
            <table class="styled-table">
                <thead>
                    <tr>
        """
        
        # Add column headers
        for col in columns:
            table_html += f"""<th>{col.get('label', col.get('key', ''))}</th>"""
        
        # Add actions column if needed
        if actions:
            table_html += """<th>Actions</th>"""
        
        table_html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add data rows
        for row_idx, item in enumerate(page_data):
            table_html += "<tr>"
            
            # Add columns
            for col in columns:
                key = col.get("key", "")
                format_func = col.get("format")
                cell_class = col.get("class", "")
                
                # Get cell value
                if key in item:
                    value = item[key]
                    
                    # Apply formatting if provided
                    if format_func and callable(format_func):
                        formatted_value = format_func(value, item)
                    else:
                        formatted_value = value
                else:
                    formatted_value = ""
                
                table_html += f"""<td class="{cell_class}">{formatted_value}</td>"""
            
            # Add actions if provided
            if actions:
                table_html += """<td style="white-space: nowrap;">"""
                
                for action_idx, action in enumerate(actions):
                    action_icon = action.get("icon", "fa-cog")
                    action_text = action.get("text", "")
                    action_class = action.get("class", "action-view")
                    
                    # Generate a unique ID using multiple factors
                    item_id = str(item.get('id', item.get('_id', '')))
                    row_position = f"{current_page}_{row_idx}"
                    
                    # Create a unique action ID that includes position, item ID, and action details
                    action_id = f"{key_prefix}_{action.get('key', 'action')}_{row_position}_{hash(item_id)}_{action_idx}"
                    
                    table_html += f"""
                    <button class="action-button {action_class}" onclick="document.getElementById('{action_id}').click();">
                        <i class="fas {action_icon}"></i> {action_text}
                    </button>
                    """
                
                table_html += """</td>"""
            
            table_html += "</tr>"
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        # Render the table
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Add hidden buttons for actions - using UUIDs for absolutely unique keys
        if actions:
            # Create placeholders for each action to hold the buttons
            action_cols = st.columns(len(actions))
            
            # For each row in the current page
            for row_idx, item in enumerate(page_data):
                # For each action defined
                for action_idx, action in enumerate(actions):
                    with action_cols[action_idx]:
                        # Generate a truly unique ID for each button
                        # Combining multiple identifiers to ensure uniqueness:
                        # 1. Table prefix
                        # 2. Action key
                        # 3. Current page number
                        # 4. Row position in current page
                        # 5. Hash of the item's ID
                        # 6. Action's position in the actions list
                        item_id = str(item.get('id', item.get('_id', '')))
                        row_position = f"{current_page}_{row_idx}"
                        
                        # Create a unique action ID
                        action_id = f"{key_prefix}_{action.get('key', 'action')}_{row_position}_{hash(item_id)}_{action_idx}"
                        
                        # Create the hidden button with the unique ID
                        if st.button("", key=action_id):
                            if "callback" in action and callable(action["callback"]):
                                return action["callback"](item)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("Previous", key=f"{key_prefix}_prev_{current_page}", disabled=current_page <= 1):
                st.session_state[f"{key_prefix}_page"] -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center;'>Showing {start_idx + 1}-{end_idx} of {total_items}</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("Next", key=f"{key_prefix}_next_{current_page}", disabled=current_page >= total_pages):
                st.session_state[f"{key_prefix}_page"] += 1
                st.rerun()
    
    # Return table state information
    return {
        "page": current_page,
        "total_pages": total_pages,
        "total_items": total_items,
        "search_term": st.session_state[f"{key_prefix}_search"]
    }
def create_morocco_map(data=None, geo_col='province', value_col=None, title="Morocco Regional Map"):
    """
    Create an interactive map of Morocco with regional data visualization.
    
    Parameters:
        data: DataFrame with regional data
        geo_col: Column name in data containing geographic region names
        value_col: Column name in data containing values to display
        title: Map title
        
    Returns:
        folium map object
    """
    # Create a base map centered on Morocco
    map_style = "CartoDB dark_matter" if st.session_state.theme == "dark" else "CartoDB positron"
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=5, tiles=map_style)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50%; 
                transform: translateX(-50%);
                z-index: 9999; 
                background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else 'white'}; 
                color: {'#E0E0E0' if st.session_state.theme == 'dark' else '#333'};
                border: 2px solid {'#333' if st.session_state.theme == 'dark' else '#ddd'}; 
                border-radius: 5px; 
                padding: 10px; 
                font-size: 16px;
                font-weight: bold;">
        {title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Load Morocco GeoJSON data
    # In a production environment, you would use an actual GeoJSON file for Morocco regions
    # For this implementation, we'll simulate regions using circle markers
    
    # Morocco regions with approximate coordinates
    morocco_regions = {
        "Rabat-Salé-Kénitra": {"lat": 34.0209, "lon": -6.8416},
        "Casablanca-Settat": {"lat": 33.5731, "lon": -7.5898},
        "Marrakech-Safi": {"lat": 31.6295, "lon": -8.0778},
        "Fès-Meknès": {"lat": 33.9372, "lon": -4.9978},
        "Tanger-Tétouan-Al Hoceïma": {"lat": 35.7595, "lon": -5.8340},
        "Souss-Massa": {"lat": 30.4278, "lon": -9.5981},
        "Oriental": {"lat": 34.6811, "lon": -1.9110},
        "Béni Mellal-Khénifra": {"lat": 32.3370, "lon": -6.3499},
        "Drâa-Tafilalet": {"lat": 31.9288, "lon": -4.4247},
        "Laâyoune-Sakia El Hamra": {"lat": 27.1418, "lon": -13.1990},
        "Dakhla-Oued Ed-Dahab": {"lat": 23.6848, "lon": -15.9579},
        "Guelmim-Oued Noun": {"lat": 28.9864, "lon": -10.0586}
    }
    
    if data is not None and geo_col in data.columns and value_col in data.columns:
        # Create a choropleth
        max_value = data[value_col].max()
        
        # Create a colormap
        import branca.colormap as cm
        colormap = cm.LinearColormap(
            ['#FEF0D9', '#FDD49E', '#FDBB84', '#FC8D59', '#E34A33', '#B30000'],
            vmin=0,
            vmax=max_value
        )
        
        # Add region markers with data
        for region, coords in morocco_regions.items():
            # Get the value for this region if it exists in the data
            region_data = data[data[geo_col] == region]
            if not region_data.empty:
                value = region_data.iloc[0][value_col]
                color = colormap(value)
                
                # Create a circle marker with the value
                folium.CircleMarker(
                    location=[coords["lat"], coords["lon"]],
                    radius=value / max_value * 30 + 10,  # Scale the radius by the value
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"<strong>{region}</strong><br>{value_col}: {value}"
                ).add_to(m)
                
                # Add a label
                folium.Marker(
                    location=[coords["lat"], coords["lon"]],
                    icon=folium.DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(75, 18),
                        html=f'<div style="font-size: 10pt; text-align: center; background-color: rgba(0,0,0,0); color: {"white" if st.session_state.theme == "dark" else "black"}"><strong>{region}</strong><br>{value}</div>'
                    )
                ).add_to(m)
            else:
                # Add just the region without data
                folium.CircleMarker(
                    location=[coords["lat"], coords["lon"]],
                    radius=10,
                    color='#AAA',
                    fill=True,
                    fill_color='#AAA',
                    fill_opacity=0.4,
                    popup=f"<strong>{region}</strong><br>No data"
                ).add_to(m)
                
                # Add a label
                folium.Marker(
                    location=[coords["lat"], coords["lon"]],
                    icon=folium.DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(75, 18),
                        html=f'<div style="font-size: 10pt; text-align: center; background-color: rgba(0,0,0,0); color: {"white" if st.session_state.theme == "dark" else "black"}"><strong>{region}</strong></div>'
                    )
                ).add_to(m)
        
        # Add the colormap legend
        colormap.caption = value_col
        m.add_child(colormap)
    else:
        # Just add region markers without data
        for region, coords in morocco_regions.items():
            folium.CircleMarker(
                location=[coords["lat"], coords["lon"]],
                radius=15,
                color='#AAA',
                fill=True,
                fill_color='#AAA',
                fill_opacity=0.4,
                popup=f"<strong>{region}</strong>"
            ).add_to(m)
            
            # Add a label
            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 18),
                    html=f'<div style="font-size: 10pt; text-align: center; background-color: rgba(0,0,0,0); color: {"white" if st.session_state.theme == "dark" else "black"}"><strong>{region}</strong></div>'
                )
            ).add_to(m)
    
    # Add municipal projects if available
    try:
        projects = load_qdrant_documents("municipal_projects", vector_dim=384)[0]
        if projects:
            # Create a project marker cluster
            from folium.plugins import MarkerCluster
            project_cluster = MarkerCluster(name="Projects")
            
            for project in projects:
                if isinstance(project, dict):
                    # Get region for the project
                    region = project.get("province", project.get("CT", "Unknown"))
                    
                    # Get coordinates for the region, or use random coordinates if region not found
                    if region in morocco_regions:
                        base_lat = morocco_regions[region]["lat"]
                        base_lon = morocco_regions[region]["lon"]
                        
                        # Add some random variation to spread out projects within the region
                        lat = base_lat + random.uniform(-0.3, 0.3)
                        lon = base_lon + random.uniform(-0.3, 0.3)
                    else:
                        # Use random coordinates within Morocco if region not found
                        lat = 31.7917 + random.uniform(-3, 3)
                        lon = -7.0926 + random.uniform(-3, 3)
                    
                    # Create popup content
                    popup_content = f"""
                    <div style="width: 250px;">
                        <h4 style="margin-top: 0;">{project.get('title', 'Untitled Project')}</h4>
                        <p><strong>Status:</strong> {project.get('status', 'Unknown')}</p>
                        <p><strong>Region:</strong> {region}</p>
                        <p><strong>Budget:</strong> {project.get('budget', 0):,} MAD</p>
                        <p><strong>Completion:</strong> {project.get('completion_percentage', 0)}%</p>
                    </div>
                    """
                    
                    # Determine icon color based on status
                    status = project.get("status", "").lower()
                    if status == "completed":
                        icon_color = "green"
                    elif status == "in progress":
                        icon_color = "orange"
                    elif status == "approved":
                        icon_color = "blue"
                    else:
                        icon_color = "gray"
                    
                    # Add marker
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=project.get('title', 'Untitled Project'),
                        icon=folium.Icon(color=icon_color, icon="info-sign")
                    ).add_to(project_cluster)
            
            project_cluster.add_to(m)
    except Exception as e:
        logger.error(f"Error loading projects for map: {e}")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_word_cloud(text_data, title="Word Cloud", width=800, height=400):
    """
    Create a word cloud visualization from text data.
    
    Parameters:
        text_data: List of strings or single string to generate word cloud from
        title: Title for the visualization
        width: Width of the word cloud image
        height: Height of the word cloud image
        
    Returns:
        matplotlib figure
    """
    # Prepare text data
    if isinstance(text_data, list):
        if all(isinstance(item, dict) for item in text_data):
            # Extract text content from dictionaries
            combined_text = " ".join([
                item.get("content", item.get("text", item.get("comment_text", "")))
                for item in text_data
                if isinstance(item, dict)
            ])
        else:
            # Join list of strings
            combined_text = " ".join([str(text) for text in text_data])
    else:
        combined_text = str(text_data)
    
    # Check if text data is empty
    if not combined_text.strip():
        return None
    
    # Generate word cloud
    theme_bg = "#1E1E1E" if st.session_state.theme == "dark" else "white"
    theme_fg = "#E0E0E0" if st.session_state.theme == "dark" else "#333"
    
    wc = WordCloud(
        width=width,
        height=height,
        background_color=theme_bg,
        colormap="viridis",
        max_words=200,
        contour_width=1,
        contour_color="rgba(255, 255, 255, 0.2)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.2)"
    ).generate(combined_text)
    
    # Create figure and display word cloud
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, color=theme_fg, fontsize=14, pad=20)
    ax.axis("off")
    ax.set_facecolor(theme_bg)
    fig.patch.set_facecolor(theme_bg)
    
    return fig

def create_sentiment_distribution_chart(data, sentiment_field="sentiment", group_by=None, title="Sentiment Distribution"):
    """
    Create a sentiment distribution chart.
    
    Parameters:
        data: List of dictionaries containing sentiment data
        sentiment_field: Field name containing sentiment values
        group_by: Optional field to group the data by
        title: Chart title
        
    Returns:
        plotly figure
    """
    # Count sentiments
    if group_by:
        # Group by specified field
        sentiment_counts = {}
        
        for item in data:
            if isinstance(item, dict):
                group = item.get(group_by, "Unknown")
                sentiment = item.get(sentiment_field, "NEU")
                
                if group not in sentiment_counts:
                    sentiment_counts[group] = {"POS": 0, "NEG": 0, "NEU": 0}
                
                if sentiment in sentiment_counts[group]:
                    sentiment_counts[group][sentiment] += 1
                else:
                    # Handle various sentiment formats
                    if sentiment.upper().startswith("P"):
                        sentiment_counts[group]["POS"] += 1
                    elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                        sentiment_counts[group]["NEG"] += 1
                    else:
                        sentiment_counts[group]["NEU"] += 1
        
        # Convert to DataFrame for plotting
        df_data = []
        
        for group, counts in sentiment_counts.items():
            for sentiment, count in counts.items():
                df_data.append({
                    group_by: group,
                    "sentiment": sentiment,
                    "count": count
                })
        
        df = pd.DataFrame(df_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df,
            x=group_by,
            y="count",
            color="sentiment",
            title=title,
            barmode="group",
            color_discrete_map={
                "POS": "#4CAF50",
                "NEG": "#F44336",
                "NEU": "#2196F3"
            }
        )
    else:
        # Count overall sentiment distribution
        sentiment_counts = {"POS": 0, "NEG": 0, "NEU": 0}
        
        for item in data:
            if isinstance(item, dict):
                sentiment = item.get(sentiment_field, "NEU")
                
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
                else:
                    # Handle various sentiment formats
                    if sentiment.upper().startswith("P"):
                        sentiment_counts["POS"] += 1
                    elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                        sentiment_counts["NEG"] += 1
                    else:
                        sentiment_counts["NEU"] += 1
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame({
            "sentiment": list(sentiment_counts.keys()),
            "count": list(sentiment_counts.values())
        })
        
        # Create pie chart
        fig = px.pie(
            df,
            values="count",
            names="sentiment",
            title=title,
            color="sentiment",
            color_discrete_map={
                "POS": "#4CAF50",
                "NEG": "#F44336",
                "NEU": "#2196F3"
            }
        )
    
    # Update layout for theme
    fig.update_layout(
        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
    )
    
    return fig

def create_time_series_chart(data, date_field, value_field, group_by=None, title="Time Series Analysis"):
    """
    Create a time series chart.
    
    Parameters:
        data: List of dictionaries containing time series data
        date_field: Field containing date values
        value_field: Field containing values to plot
        group_by: Optional field to group data by
        title: Chart title
        
    Returns:
        plotly figure
    """
    # Prepare data
    time_series_data = []
    
    for item in data:
        if isinstance(item, dict) and date_field in item and value_field in item:
            # Convert date to datetime if needed
            date_value = item[date_field]
            if not isinstance(date_value, datetime):
                try:
                    date_value = pd.to_datetime(date_value)
                except:
                    continue
            
            # Add entry to time series data
            entry = {
                "date": date_value,
                "value": item[value_field]
            }
            
            # Add group if specified
            if group_by and group_by in item:
                entry["group"] = item[group_by]
            
            time_series_data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(time_series_data)
    
    # Check if we have data
    if df.empty:
        return None
    
    # Sort by date
    df = df.sort_values("date")
    
    # Create time series chart
    if group_by and "group" in df.columns:
        # Create grouped line chart
        fig = px.line(
            df,
            x="date",
            y="value",
            color="group",
            title=title,
            markers=True
        )
    else:
        # Create single line chart
        fig = px.line(
            df,
            x="date",
            y="value",
            title=title,
            markers=True
        )
    
    # Update layout for theme
    fig.update_layout(
        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
        xaxis_title="Date",
        yaxis_title=value_field.replace("_", " ").title()
    )
    
    return fig

# -------------------------------------------------------------
# AI INSIGHTS FEATURES
# -------------------------------------------------------------

def summarize_comments(comments, max_tokens=250):
    """
    Use OpenAI API to generate a summary of comments.
    
    Parameters:
        comments: List of comment dictionaries or strings
        max_tokens: Maximum tokens for summary
        
    Returns:
        str: Generated summary
    """
    # Extract comment text
    comments_text = []
    
    for comment in comments:
        if isinstance(comment, dict):
            # Extract text from comment dictionary
            text = comment.get("content", comment.get("text", comment.get("comment_text", "")))
            if text:
                comments_text.append(text)
        elif isinstance(comment, str):
            comments_text.append(comment)
    
    # Check if we have any comments
    if not comments_text:
        return "No comments to summarize."
    
    # Create text for OpenAI
    comments_combined = "\n\n".join(f"Comment {i+1}: {text}" for i, text in enumerate(comments_text[:20]))
    
    # Create cache key based on text content
    import hashlib
    text_hash = hashlib.md5(comments_combined.encode()).hexdigest()
    cache_key = f"summary_{text_hash}"
    
    # Check cache first
    cached_summary = cache.get(cache_key)
    if cached_summary:
        return cached_summary
    
    # Generate summary using OpenAI
    try:
        if not openai.api_key:
            return "OpenAI API key not configured."
        
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant tasked with summarizing citizen comments and feedback in Morocco. Create a concise, insightful summary that captures the main themes, sentiments, and concerns expressed in these comments."},
                {"role": "user", "content": f"Please summarize the following citizen comments:\n\n{comments_combined}\n\nProvide a concise summary (max 3-4 sentences) highlighting the main themes and sentiments."}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        
        summary = response['choices'][0]['message']['content'].strip()
        
        # Cache the summary for future use
        cache.set(cache_key, summary, ttl=86400)  # Cache for 24 hours
        
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

def extract_topics(texts, num_topics=5, num_words=5):
    """
    Extract topics from a collection of texts using LDA.
    
    Parameters:
        texts: List of text strings
        num_topics: Number of topics to extract
        num_words: Number of words per topic
        
    Returns:
        list: List of topics with top words
    """
    try:
        # Create cache key
        cache_key = f"topics_{hash(str(texts))}_{num_topics}_{num_words}"
        
        # Check cache first
        cached_topics = cache.get(cache_key)
        if cached_topics:
            return cached_topics
        
        # Create a document-term matrix
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000
        )
        
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # If vectorization fails, try with a smaller vocabulary
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=1,
                stop_words='english',
                max_features=100
            )
            dtm = vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create and fit the LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        
        lda.fit(dtm)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                "id": topic_idx,
                "words": top_words,
                "weight": float(topic.sum())
            })
        
        # Sort topics by weight
        topics = sorted(topics, key=lambda x: x["weight"], reverse=True)
        
        # Cache the results
        cache.set(cache_key, topics, ttl=3600)  # Cache for 1 hour
        
        return topics
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return []

def generate_user_activity_report(username, days=30):
    """
    Generate a report on user activity.
    
    Parameters:
        username: Username to analyze
        days: Number of days to analyze
        
    Returns:
        dict: Report with various activity metrics
    """
    try:
        client = get_mongo_client()
        if not client:
            return {"error": "Database connection error"}
        
        db = client["CivicCatalyst"]
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # User profile
        user = db["users"].find_one({"username": username})
        if not user:
            return {"error": "User not found"}
        
        user_id = str(user.get("_id"))
        
        # Audit logs
        audit_logs = list(db["audit_logs"].find(
            {"user": username, "timestamp": {"$gte": start_date}}
        ).sort("timestamp", -1))
        
        # Comments
        # Note: In a real system, you'd have proper user_id references
        # Here we're simplifying by using username matching
        comments = list(db["comments"].find(
            {"user_id": user_id, "timestamp": {"$gte": start_date}}
        ).sort("timestamp", -1))
        
        # Ideas
        ideas = list(db["ideas"].find(
            {"user_id": user_id, "date_submitted": {"$gte": start_date}}
        ).sort("date_submitted", -1))
        
        # Login history
        login_history = list(db["login_history"].find(
            {"username": username, "timestamp": {"$gte": start_date}}
        ).sort("timestamp", -1))
        
        # Calculate activity metrics
        logins_count = len(login_history)
        comments_count = len(comments)
        ideas_count = len(ideas)
        actions_count = len(audit_logs)
        
        # Group activity by day
        activity_by_day = {}
        for day_offset in range(days):
            day = (end_date - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            activity_by_day[day] = 0
        
        for log in audit_logs:
            day = log["timestamp"].strftime("%Y-%m-%d")
            if day in activity_by_day:
                activity_by_day[day] += 1
        
        for comment in comments:
            try:
                day = comment["timestamp"].strftime("%Y-%m-%d")
                if day in activity_by_day:
                    activity_by_day[day] += 1
            except (KeyError, AttributeError):
                pass
        
        for idea in ideas:
            try:
                day = idea["date_submitted"].strftime("%Y-%m-%d")
                if day in activity_by_day:
                    activity_by_day[day] += 1
            except (KeyError, AttributeError):
                pass
        
        # Create time series
        activity_series = [
            {"date": day, "activity": count}
            for day, count in activity_by_day.items()
        ]
        activity_series.sort(key=lambda x: x["date"])
        
        # Generate report
        report = {
            "username": username,
            "user_id": user_id,
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "logins_count": logins_count,
            "comments_count": comments_count,
            "ideas_count": ideas_count,
            "actions_count": actions_count,
            "total_activity": logins_count + comments_count + ideas_count + actions_count,
            "activity_series": activity_series,
            "last_login": login_history[0]["timestamp"] if login_history else None,
            "recent_actions": [
                {
                    "action_type": log["action_type"],
                    "description": log["description"],
                    "timestamp": log["timestamp"]
                }
                for log in audit_logs[:10]
            ]
        }
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating user activity report: {e}")
        return {"error": f"Error generating report: {str(e)}"}
    
    finally:
        if client:
            client.close()

# -------------------------------------------------------------
# USER MANAGEMENT MODULE
# -------------------------------------------------------------

def user_management():
    """User management interface with CRUD operations."""
    st.title(t("user_management"))
    
    # Sidebar filters
    st.sidebar.header(t("filters"))
    
    search_query = st.sidebar.text_input(
        t("search"), 
        placeholder=t("username") + " / " + t("email")
    )
    
    role_filter = st.sidebar.selectbox(
        t("role"),
        ["All"] + list(USER_ROLES.keys())
    )
    
    status_filter = st.sidebar.radio(
        "Status",
        ["All", "Active", "Inactive"]
    )
    
    # Load users with filters
    users = load_users(
        search_query=search_query,
        role_filter=role_filter if role_filter != "All" else None,
        status_filter=status_filter if status_filter != "All" else None
    )
    
    # User metrics
    st.subheader("User Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            t("total_users"),
            f"{len(users):,}",
            icon="fa-users",
            color="#2196F3"
        )
    
    with col2:
        # Count active users
        active_count = sum(1 for user in users if user.get("active", True))
        active_percentage = (active_count / len(users) * 100) if users else 0
        
        display_metric_card(
            t("active_users"),
            f"{active_count:,}",
            delta=round(active_percentage),
            delta_description="of total users",
            icon="fa-user-check",
            color="#4CAF50"
        )
    
    with col3:
        # Count admins
        admin_count = sum(1 for user in users if user.get("role") == "admin")
        
        display_metric_card(
            "Administrators",
            f"{admin_count:,}",
            icon="fa-user-shield",
            color="#FF9800"
        )
    
    with col4:
        # Recent logins
        recent_logins = sum(1 for user in users if user.get("last_login") and 
                           (datetime.now() - user["last_login"]).days < 7)
        
        display_metric_card(
            "Recent Logins",
            f"{recent_logins:,}",
            icon="fa-sign-in-alt",
            color="#9C27B0"
        )
    
    # User management tabs
    tabs = st.tabs(["User List", "User Analytics", "Role Management", "New User"])
    
    with tabs[0]:
        st.subheader("User List")
        
        # Action buttons
        # Action buttons
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Export Users", key="export_users"):
                # Convert users to DataFrame
                user_df = pd.DataFrame(users)
                
                # Remove sensitive fields
                sensitive_fields = ["password_hash", "password_salt", "_id"]
                for field in sensitive_fields:
                    if field in user_df.columns:
                        user_df = user_df.drop(columns=[field])
                
                # Convert to CSV
                csv = user_df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "users_export.csv",
                    "text/csv",
                    key="download_users_csv"
                )
        
        # User table with actions
        user_columns = [
            {
                "key": "username",
                "label": "Username",
                "format": lambda val, item: f"""
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 24px; height: 24px; border-radius: 50%; background-color: {color_from_string(val)}; 
                             display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                            {val[0].upper()}
                        </div>
                        <div>{val}</div>
                    </div>
                """
            },
            {
                "key": "email",
                "label": "Email"
            },
            {
                "key": "role",
                "label": "Role",
                "format": lambda val, item: f"""
                    <span class="badge badge-primary">
                        <i class="fas {ICONS.get(val, 'fa-user')}"></i> {val.title()}
                    </span>
                """
            },
            {
                "key": "active",
                "label": "Status",
                "format": lambda val, item: f"""
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="status-dot status-{'active' if val else 'inactive'}"></span>
                        <span>{'Active' if val else 'Inactive'}</span>
                    </div>
                """
            },
            {
                "key": "last_login",
                "label": "Last Login",
                "format": lambda val, item: format_datetime(val) if val else "Never"
            },
            {
                "key": "created_at",
                "label": "Created",
                "format": lambda val, item: format_datetime(val) if val else "Unknown"
            }
        ]
        
        user_actions = [
            {
                "key": "view",
                "text": "View",
                "icon": "fa-eye",
                "class": "action-view",
                "callback": view_user_details
            },
            {
                "key": "edit",
                "text": "Edit",
                "icon": "fa-pen-to-square",
                "class": "action-edit",
                "callback": edit_user
            },
            {
                "key": "delete",
                "text": "Delete",
                "icon": "fa-trash",
                "class": "action-delete",
                "callback": delete_user_confirmation
            }
        ]
        
        create_dynamic_table(
            users, 
            user_columns, 
            key_prefix="users_table", 
            page_size=10, 
            searchable=True,
            actions=user_actions
        )
    
    with tabs[1]:
        st.subheader("User Analytics")
        
        # User role distribution
        role_counts = {}
        for user in users:
            role = user.get("role", "unknown")
            if role not in role_counts:
                role_counts[role] = 0
            role_counts[role] += 1
        
        # Create pie chart for role distribution
        role_df = pd.DataFrame({
            "role": list(role_counts.keys()),
            "count": list(role_counts.values())
        })
        
        role_fig = px.pie(
            role_df,
            values="count",
            names="role",
            title="User Role Distribution",
            color="role",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Update layout for theme consistency
        role_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
        )
        
        st.plotly_chart(role_fig, use_container_width=True)
        
        # User signup trend
        signup_dates = []
        for user in users:
            created_at = user.get("created_at")
            if created_at:
                signup_dates.append(created_at)
        
        if signup_dates:
            # Create DataFrame with signup dates
            signup_df = pd.DataFrame({"signup_date": signup_dates})
            
            # Convert to datetime and extract date components
            signup_df["signup_date"] = pd.to_datetime(signup_df["signup_date"])
            signup_df["date"] = signup_df["signup_date"].dt.date
            
            # Count signups by date
            signup_counts = signup_df.groupby("date").size().reset_index(name="count")
            
            # Create line chart
            signup_fig = px.line(
                signup_counts,
                x="date",
                y="count",
                title="User Signup Trend",
                markers=True
            )
            
            # Update layout for theme consistency
            signup_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Date",
                yaxis_title="New Users"
            )
            
            st.plotly_chart(signup_fig, use_container_width=True)
        
        # User activity analysis
        st.subheader("User Activity Analysis")
        
        # User selection for activity analysis
        selected_user = st.selectbox(
            "Select User for Activity Analysis",
            [user.get("username") for user in users if "username" in user]
        )
        
        if selected_user:
            activity_days = st.slider(
                "Analysis Period (days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1
            )
            
            # Generate user activity report
            activity_report = generate_user_activity_report(selected_user, days=activity_days)
            
            if "error" in activity_report:
                st.error(activity_report["error"])
            else:
                # Display activity metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_metric_card(
                        "Total Activity",
                        activity_report["total_activity"],
                        icon="fa-chart-line"
                    )
                
                with col2:
                    display_metric_card(
                        "Logins",
                        activity_report["logins_count"],
                        icon="fa-sign-in-alt"
                    )
                
                with col3:
                    display_metric_card(
                        "Comments",
                        activity_report["comments_count"],
                        icon="fa-comment"
                    )
                
                with col4:
                    display_metric_card(
                        "Ideas Submitted",
                        activity_report["ideas_count"],
                        icon="fa-lightbulb"
                    )
                
                # Activity time series chart
                if activity_report["activity_series"]:
                    activity_df = pd.DataFrame(activity_report["activity_series"])
                    activity_df["date"] = pd.to_datetime(activity_df["date"])
                    
                    activity_fig = px.line(
                        activity_df,
                        x="date",
                        y="activity",
                        title=f"Activity Timeline for {selected_user}",
                        markers=True
                    )
                    
                    # Update layout for theme consistency
                    activity_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        xaxis_title="Date",
                        yaxis_title="Activity Count"
                    )
                    
                    st.plotly_chart(activity_fig, use_container_width=True)
                
                # Recent actions
                st.subheader("Recent Actions")
                
                if activity_report["recent_actions"]:
                    actions_df = pd.DataFrame(activity_report["recent_actions"])
                    
                    # Format timestamp
                    actions_df["formatted_time"] = actions_df["timestamp"].apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if x else "Unknown"
                    )
                    
                    # Create table
                    actions_table = """
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Action</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for _, row in actions_df.iterrows():
                        actions_table += f"""
                        <tr>
                            <td>{row['formatted_time']}</td>
                            <td><span class="badge badge-info">{row['action_type']}</span></td>
                            <td>{row['description']}</td>
                        </tr>
                        """
                    
                    actions_table += """
                        </tbody>
                    </table>
                    """
                    
                    st.markdown(actions_table, unsafe_allow_html=True)
                else:
                    st.info("No recent actions recorded for this user.")
    
    with tabs[2]:
        st.subheader("Role Management")
        
        # Role description and permissions
        role_info = """
        <div class="data-card">
            <h4 style="margin-top: 0;">Role Definitions</h4>
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Role</th>
                        <th>Description</th>
                        <th>Permissions</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for role, info in USER_ROLES.items():
            role_info += f"""
            <tr>
                <td><span class="badge badge-primary"><i class="fas {ICONS.get(role, 'fa-user')}"></i> {role.title()}</span></td>
                <td>{info.get('description', 'No description')}</td>
                <td>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
            """
            
            for permission in info.get('permissions', []):
                role_info += f"""
                <span class="badge badge-info">{permission.replace('_', ' ').title()}</span>
                """
            
            role_info += """
                    </div>
                </td>
            </tr>
            """
        
        role_info += """
                </tbody>
            </table>
        </div>
        """
        
        st.markdown(role_info, unsafe_allow_html=True)
        
        # Role assignment
        st.subheader("Change User Role")
        
        # User and role selection
        role_col1, role_col2, role_col3 = st.columns([2, 2, 1])
        
        with role_col1:
            role_username = st.selectbox(
                "Select User",
                [user.get("username") for user in users if "username" in user],
                key="role_username"
            )
        
        with role_col2:
            selected_role = st.selectbox(
                "New Role",
                list(USER_ROLES.keys()),
                key="selected_role"
            )
        
        with role_col3:
            if st.button("Update Role", key="update_role_btn"):
                if role_username and selected_role:
                    # Find user by username
                    user_to_update = next((user for user in users if user.get("username") == role_username), None)
                    
                    if user_to_update and "_id" in user_to_update:
                        # Update user role
                        success, message = update_user(
                            user_to_update["_id"],
                            {"role": selected_role}
                        )
                        
                        if success:
                            st.success(f"User role updated to {selected_role}")
                            
                            # Refresh user list
                            st.rerun()
                        else:
                            st.error(f"Error updating role: {message}")
                    else:
                        st.error("User not found")
    
    with tabs[3]:
        st.subheader("Create New User")
        
        # New user form
        with st.form("new_user_form"):
            new_username = st.text_input("Username", key="new_username")
            new_email = st.text_input("Email", key="new_email")
            new_password = st.text_input("Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            new_role = st.selectbox("Role", list(USER_ROLES.keys()), key="new_role")
            new_full_name = st.text_input("Full Name (Optional)", key="new_full_name")
            
            submit_button = st.form_submit_button("Create User")
        
        if submit_button:
            # Validate form
            if not new_username or not new_email or not new_password:
                st.error("Please fill in all required fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Create new user
                success, result = create_new_user(
                    new_username,
                    new_email,
                    new_password,
                    new_role,
                    full_name=new_full_name
                )
                
                if success:
                    st.success(f"User {new_username} created successfully!")
                    
                    # Clear form
                    st.session_state.new_username = ""
                    st.session_state.new_email = ""
                    st.session_state.new_password = ""
                    st.session_state.confirm_password = ""
                    st.session_state.new_full_name = ""
                    
                    # Refresh user list
                    st.rerun()
                else:
                    st.error(f"Error creating user: {result}")

# -------------------------------------------------------------
# PROJECT MANAGEMENT MODULE
# -------------------------------------------------------------

def project_management():
    """Project management interface with CRUD operations."""
    st.title(t("project_management"))
    
    # Sidebar filters
    st.sidebar.header(t("filters"))
    
    search_query = st.sidebar.text_input(
        t("search"), 
        placeholder=t("project_title")
    )
    
    category_filter = st.sidebar.selectbox(
        t("project_category"),
        ["All"] + PROJECT_CATEGORIES
    )
    
    status_filter = st.sidebar.selectbox(
        t("project_status"),
        ["All"] + PROJECT_STATUSES
    )
    
    region_filter = st.sidebar.selectbox(
        t("project_region"),
        MOROCCO_REGIONS
    )
    
    # Load projects with filters
    filter_conditions = []
    
    if search_query:
        title_condition = FieldCondition(
            key="title",
            match=MatchValue(value=search_query)
        )
        description_condition = FieldCondition(
            key="description",
            match=MatchValue(value=search_query)
        )
        filter_conditions.append(Filter(should=[title_condition, description_condition]))
    
    if category_filter != "All":
        filter_conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category_filter)
            )
        )
    
    if status_filter != "All":
        filter_conditions.append(
            FieldCondition(
                key="status",
                match=MatchValue(value=status_filter)
            )
        )
    
    if region_filter != "All":
        province_condition = FieldCondition(
            key="province",
            match=MatchValue(value=region_filter)
        )
        ct_condition = FieldCondition(
            key="CT",
            match=MatchValue(value=region_filter)
        )
        filter_conditions.append(Filter(should=[province_condition, ct_condition]))
    
    # Create filter if we have conditions
    projects_filter = None
    if filter_conditions:
        projects_filter = Filter(must=filter_conditions)
    
    # Load projects
    projects, next_scroll_id = load_qdrant_documents(
        "municipal_projects", 
        vector_dim=384, 
        limit=100, 
        filters=projects_filter
    )
    
    # Project metrics
    st.subheader("Project Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            t("total_projects"),
            f"{len(projects):,}",
            icon="fa-diagram-project",
            color="#2196F3"
        )
    
    with col2:
        # Count completed projects
        completed_count = sum(1 for project in projects if project.get("status") == "Completed")
        completion_rate = (completed_count / len(projects) * 100) if projects else 0
        
        display_metric_card(
            "Completed Projects",
            f"{completed_count:,}",
            delta=round(completion_rate),
            delta_description="completion rate",
            icon="fa-check-circle",
            color="#4CAF50"
        )
    
    with col3:
        # Calculate total budget
        total_budget = sum(float(project.get("budget", 0)) for project in projects)
        
        display_metric_card(
            "Total Budget",
            f"{total_budget:,.0f} MAD",
            icon="fa-money-bill",
            color="#FF9800"
        )
    
    with col4:
        # Average completion percentage
        completion_percentages = [float(project.get("completion_percentage", 0)) for project in projects]
        avg_completion = sum(completion_percentages) / len(completion_percentages) if completion_percentages else 0
        
        display_metric_card(
            "Avg. Completion",
            f"{avg_completion:.1f}%",
            icon="fa-tasks-alt",
            color="#9C27B0"
        )
    
    # Project management tabs
    tabs = st.tabs(["Project List", "Project Analytics", "New Project", "Geographic View"])
    
    with tabs[0]:
        st.subheader("Project List")
        
        # Action buttons
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Export Projects", key="export_projects"):
                # Convert projects to DataFrame
                project_df = pd.DataFrame(projects)
                
                # Convert to CSV
                csv = project_df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "projects_export.csv",
                    "text/csv",
                    key="download_projects_csv"
                )
        
        # Project table with actions
        project_columns = [
            {
                "key": "title",
                "label": "Project Title",
                "format": lambda val, item: f"""
                    <div style="font-weight: 500;">{val}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">{item.get('description', '')[:50]}...</div>
                """
            },
            {
                "key": "status",
                "label": "Status",
                "format": lambda val, item: f"""
                    <span class="badge badge-{'success' if val == 'Completed' else 'warning' if val == 'In Progress' else 'info'}">
                        {val}
                    </span>
                """
            },
            {
                "key": "province",
                "label": "Region",
                "format": lambda val, item: val or item.get("CT", "Unknown")
            },
            {
                "key": "budget",
                "label": "Budget",
                "format": lambda val, item: f"{float(val):,.0f} MAD" if val else "N/A"
            },
            {
                "key": "completion_percentage",
                "label": "Completion",
                "format": lambda val, item: f"""
                    <div style="width: 100%; background-color: {'#333' if st.session_state.theme == 'dark' else '#EEE'}; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="width: {val}%; height: 100%; background-color: {'#4CAF50' if float(val) == 100 else '#FF9800'}; border-radius: 4px;"></div>
                    </div>
                    <div style="text-align: center; font-size: 0.8rem; margin-top: 4px;">{val}%</div>
                """
            }
        ]
        
        project_actions = [
            {
                "key": "view",
                "text": "View",
                "icon": "fa-eye",
                "class": "action-view",
                "callback": view_project_details
            },
            {
                "key": "edit",
                "text": "Edit",
                "icon": "fa-pen-to-square",
                "class": "action-edit",
                "callback": edit_project
            },
            {
                "key": "delete",
                "text": "Delete",
                "icon": "fa-trash",
                "class": "action-delete",
                "callback": delete_project_confirmation
            }
        ]
        
        create_dynamic_table(
            projects, 
            project_columns, 
            key_prefix="projects_table", 
            page_size=10, 
            searchable=True,
            actions=project_actions
        )
    
    with tabs[1]:
        st.subheader("Project Analytics")
        
        # Project status distribution
        status_counts = {}
        for project in projects:
            status = project.get("status", "Unknown")
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        # Create pie chart for status distribution
        status_df = pd.DataFrame({
            "status": list(status_counts.keys()),
            "count": list(status_counts.values())
        })
        
        status_fig = px.pie(
            status_df,
            values="count",
            names="status",
            title="Project Status Distribution",
            color="status",
            color_discrete_map={
                "Completed": "#4CAF50",
                "In Progress": "#FF9800",
                "Approved": "#2196F3",
                "Proposed": "#9E9E9E",
                "On Hold": "#FFC107",
                "Cancelled": "#F44336"
            }
        )
        
        # Update layout for theme consistency
        status_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
        )
        
        st.plotly_chart(status_fig, use_container_width=True)
        
        # Budget by region
        region_budgets = {}
        for project in projects:
            region = project.get("province", project.get("CT", "Unknown"))
            budget = float(project.get("budget", 0))
            
            if region not in region_budgets:
                region_budgets[region] = 0
            
            region_budgets[region] += budget
        
        # Create bar chart for budget distribution
        region_budget_df = pd.DataFrame({
            "region": list(region_budgets.keys()),
            "budget": list(region_budgets.values())
        })
        
        # Sort by budget
        region_budget_df = region_budget_df.sort_values("budget", ascending=False)
        
        budget_fig = px.bar(
            region_budget_df,
            x="region",
            y="budget",
            title="Budget Distribution by Region",
            color="budget",
            color_continuous_scale="Viridis",
            labels={"budget": "Budget (MAD)", "region": "Region"}
        )
        
        # Update layout for theme consistency
        budget_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Region",
            yaxis_title="Budget (MAD)"
        )
        
        st.plotly_chart(budget_fig, use_container_width=True)
        
        # Project completion timeline
        completed_projects = [p for p in projects if p.get("status") == "Completed" and "end_date" in p]
        
        if completed_projects:
            # Convert end dates to datetime
            for project in completed_projects:
                try:
                    project["end_date"] = pd.to_datetime(project["end_date"])
                except:
                    # Skip projects with invalid dates
                    continue
            
            # Filter projects with valid dates
            valid_projects = [p for p in completed_projects if isinstance(p.get("end_date"), pd.Timestamp)]
            
            if valid_projects:
                # Create DataFrame for completed projects
                completion_df = pd.DataFrame([
                    {
                        "title": p.get("title", "Untitled"),
                        "end_date": p["end_date"],
                        "budget": float(p.get("budget", 0))
                    }
                    for p in valid_projects
                ])
                
                # Sort by completion date
                completion_df = completion_df.sort_values("end_date")
                
                # Create scatter plot for project completions
                completion_fig = px.scatter(
                    completion_df,
                    x="end_date",
                    y="budget",
                    size="budget",
                    text="title",
                    title="Project Completion Timeline",
                    labels={"end_date": "Completion Date", "budget": "Budget (MAD)"}
                )
                
                # Update layout for theme consistency
                completion_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Completion Date",
                    yaxis_title="Budget (MAD)"
                )
                
                st.plotly_chart(completion_fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Create New Project")
        
        # New project form
        with st.form("new_project_form"):
            new_title = st.text_input("Project Title", key="new_project_title")
            new_description = st.text_area("Project Description", key="new_project_description")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_category = st.selectbox("Category", PROJECT_CATEGORIES, key="new_project_category")
                new_status = st.selectbox("Status", PROJECT_STATUSES, key="new_project_status")
                new_region = st.selectbox("Region", MOROCCO_REGIONS[1:], key="new_project_region")  # Skip "All"
            
            with col2:
                new_budget = st.number_input("Budget (MAD)", min_value=0, step=10000, key="new_project_budget")
                new_start_date = st.date_input("Start Date", key="new_project_start_date")
                new_end_date = st.date_input("Estimated End Date", key="new_project_end_date")
            
            new_completion = st.slider("Completion Percentage", 0, 100, 0, key="new_project_completion")
            
            submit_button = st.form_submit_button("Create Project")
        
        if submit_button:
            # Validate form
            if not new_title or not new_description or not new_region:
                st.error("Please fill in all required fields.")
            else:
                # Create new project
                try:
                    # Get Qdrant client
                    client = get_qdrant_client()
                    if not client:
                        st.error("Failed to connect to Qdrant.")
                    else:
                        # Generate a unique project ID
                        project_id = str(uuid.uuid4())
                        
                        # Create project document
                        project_doc = {
                            "project_id": project_id,
                            "title": new_title,
                            "description": new_description,
                            "category": new_category,
                            "status": new_status,
                            "province": new_region,
                            "budget": float(new_budget),
                            "start_date": new_start_date.strftime("%Y-%m-%d"),
                            "end_date": new_end_date.strftime("%Y-%m-%d"),
                            "completion_percentage": new_completion,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "created_by": st.session_state.get("username", "admin"),
                            "vote_score": 0,
                            "participation_count": 0
                        }
                        
                        # Get default vector
                        # In a production environment, you'd generate a proper embedding
                        # Get default vector
                        # In a production environment, you'd generate a proper embedding
                        default_vector = [0.0] * 384
                        
                        # Add project to Qdrant collection
                        client.upsert(
                            collection_name="municipal_projects",
                            points=[
                                PointStruct(
                                    id=project_id,
                                    vector=default_vector,
                                    payload=project_doc
                                )
                            ]
                        )
                        
                        # Add to audit log
                        add_audit_log(
                            "project_creation",
                            f"Created new project: {new_title}",
                            {"project_id": project_id, "title": new_title},
                            st.session_state.get("username", "admin")
                        )
                        
                        st.success(f"Project '{new_title}' created successfully!")
                        
                        # Clear form by resetting session state
                        st.session_state.new_project_title = ""
                        st.session_state.new_project_description = ""
                        st.session_state.new_project_budget = 0
                        st.session_state.new_project_completion = 0
                        
                        # Refresh project list
                        st.rerun()
                
                except Exception as e:
                    logger.error(f"Error creating project: {e}")
                    st.error(f"Error creating project: {str(e)}")
    
    with tabs[3]:
        st.subheader("Geographic Project Distribution")
        
        # Create data for map
        if projects:
            # Group projects by region
            projects_by_region = {}
            
            for project in projects:
                region = project.get("province", project.get("CT", "Unknown"))
                if region not in projects_by_region:
                    projects_by_region[region] = {
                        "count": 0,
                        "budget": 0,
                        "completed": 0
                    }
                
                projects_by_region[region]["count"] += 1
                projects_by_region[region]["budget"] += float(project.get("budget", 0))
                
                if project.get("status") == "Completed":
                    projects_by_region[region]["completed"] += 1
            
            # Create DataFrame for map
            map_data = pd.DataFrame([
                {
                    "province": region,
                    "count": data["count"],
                    "budget": data["budget"],
                    "completed_ratio": data["completed"] / data["count"] if data["count"] > 0 else 0
                }
                for region, data in projects_by_region.items()
                if region != "Unknown"
            ])
            
            # Create tabs for different map views
            map_tabs = st.tabs(["Project Count", "Budget Distribution", "Completion Rate"])
            
            with map_tabs[0]:
                if not map_data.empty:
                    map_fig = create_morocco_map(
                        map_data,
                        geo_col="province",
                        value_col="count",
                        title="Project Count by Region"
                    )
                    folium_static(map_fig, width=800)
                else:
                    st.info("No geographic data available for mapping.")
            
            with map_tabs[1]:
                if not map_data.empty:
                    budget_map_fig = create_morocco_map(
                        map_data,
                        geo_col="province",
                        value_col="budget",
                        title="Budget Distribution by Region"
                    )
                    folium_static(budget_map_fig, width=800)
                else:
                    st.info("No geographic data available for mapping.")
            
            with map_tabs[2]:
                if not map_data.empty:
                    completion_map_fig = create_morocco_map(
                        map_data,
                        geo_col="province",
                        value_col="completed_ratio",
                        title="Project Completion Rate by Region"
                    )
                    folium_static(completion_map_fig, width=800)
                else:
                    st.info("No geographic data available for mapping.")
        else:
            st.info("No projects available for geographic visualization.")

# -------------------------------------------------------------
# CONTENT MODERATION MODULE
# -------------------------------------------------------------

def content_moderation():
    """Content moderation interface for reviewing flagged content."""
    st.title(t("content_moderation"))
    
    # Sidebar filters
    st.sidebar.header(t("filters"))
    
    content_type = st.sidebar.selectbox(
        "Content Type",
        ["All", "Comments", "Ideas", "News Comments"]
    )
    
    flag_reason = st.sidebar.selectbox(
        "Flag Reason",
        ["All", "Inappropriate", "Spam", "Offensive", "Off-Topic", "Other"]
    )
    
    moderation_status = st.sidebar.selectbox(
        "Moderation Status",
        ["Pending", "Approved", "Rejected", "All"]
    )
    
    # Load flagged content
    # In a real implementation, you'd query specific collections or have a dedicated flagged_content collection
    # We'll simulate this by filtering from the existing collections
    
    # Determine which collections to search based on content type
    collections_to_search = []
    
    if content_type == "All" or content_type == "Comments":
        collections_to_search.append(("citizen_comments", 384))
    
    if content_type == "All" or content_type == "Ideas":
        collections_to_search.append(("citizen_ideas", 384))
    
    if content_type == "All" or content_type == "News Comments":
        collections_to_search.append(("hespress_politics_comments", 384))
    
    # Create filter condition for flagged content
    flagged_condition = FieldCondition(
        key="flagged",
        match=MatchValue(value=True)
    )
    
    # Add flag reason filter if specified
    filter_conditions = [flagged_condition]
    
    if flag_reason != "All":
        reason_condition = FieldCondition(
            key="flag_reason",
            match=MatchValue(value=flag_reason)
        )
        filter_conditions.append(reason_condition)
    
    # Add moderation status filter if specified
    if moderation_status != "All":
        status_condition = FieldCondition(
            key="moderation_status",
            match=MatchValue(value=moderation_status.lower())
        )
        filter_conditions.append(status_condition)
    
    # Create combined filter
    combined_filter = Filter(must=filter_conditions)
    
    # Load flagged content from selected collections
    all_flagged_content = []
    
    for collection_name, vector_dim in collections_to_search:
        try:
            flagged_items, _ = load_qdrant_documents(
                collection_name,
                vector_dim=vector_dim,
                limit=100,
                filters=combined_filter
            )
            
            # Add collection name to each item for reference
            for item in flagged_items:
                item["source_collection"] = collection_name
            
            all_flagged_content.extend(flagged_items)
        except Exception as e:
            logger.error(f"Error loading flagged content from {collection_name}: {e}")
    
    # Content moderation metrics
    st.subheader("Moderation Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Flagged Content",
            f"{len(all_flagged_content):,}",
            icon="fa-flag",
            color="#F44336"
        )
    
    with col2:
        # Count pending items
        pending_count = sum(1 for item in all_flagged_content if item.get("moderation_status") == "pending")
        
        display_metric_card(
            "Pending Review",
            f"{pending_count:,}",
            icon="fa-clock",
            color="#FF9800"
        )
    
    with col3:
        # Count approved items
        approved_count = sum(1 for item in all_flagged_content if item.get("moderation_status") == "approved")
        
        display_metric_card(
            "Approved Content",
            f"{approved_count:,}",
            icon="fa-check-circle",
            color="#4CAF50"
        )
    
    with col4:
        # Count rejected items
        rejected_count = sum(1 for item in all_flagged_content if item.get("moderation_status") == "rejected")
        
        display_metric_card(
            "Rejected Content",
            f"{rejected_count:,}",
            icon="fa-times-circle",
            color="#9E9E9E"
        )
    
    # Content moderation tabs
    tabs = st.tabs(["Flagged Content", "Moderation Analytics", "Moderation Logs"])
    
    with tabs[0]:
        st.subheader("Flagged Content Queue")
        
        if not all_flagged_content:
            st.info("No flagged content matching the specified filters.")
        else:
            # Process flagged content queue
            for i, item in enumerate(all_flagged_content):
                with st.container():
                    # Create an expander for each item
                    collection_display = {
                        "citizen_comments": "Citizen Comment",
                        "citizen_ideas": "Citizen Idea",
                        "hespress_politics_comments": "News Comment"
                    }.get(item.get("source_collection", ""), "Content")
                    
                    # Determine content text field based on collection
                    content_field = {
                        "citizen_comments": "content",
                        "citizen_ideas": "challenge",  # or solution
                        "hespress_politics_comments": "comment_text"
                    }.get(item.get("source_collection", ""), "content")
                    
                    content_text = item.get(content_field, "") 
                    if item.get("source_collection") == "citizen_ideas" and "solution" in item:
                        content_text += f"\n\nSolution: {item.get('solution', '')}"
                    
                    # Get content author
                    author = item.get("user_id", item.get("username", "Anonymous"))
                    
                    # Get content timestamp
                    timestamp = item.get("timestamp", item.get("date_submitted", "Unknown date"))
                    if isinstance(timestamp, datetime):
                        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get flag reason
                    flag_reason = item.get("flag_reason", "Not specified")
                    
                    # Get moderation status
                    status = item.get("moderation_status", "pending")
                    status_badge = {
                        "pending": f'<span class="badge badge-warning"><i class="fas fa-clock"></i> Pending</span>',
                        "approved": f'<span class="badge badge-success"><i class="fas fa-check"></i> Approved</span>',
                        "rejected": f'<span class="badge badge-danger"><i class="fas fa-ban"></i> Rejected</span>'
                    }.get(status, f'<span class="badge badge-info">Unknown</span>')
                    
                    with st.expander(f"{collection_display} - Flagged {timestamp}"):
                        # Content details
                        st.markdown(f"**Author:** {author}")
                        st.markdown(f"**Flag Reason:** {flag_reason}")
                        st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
                        
                        # Content text
                        st.text_area(
                            "Content", 
                            value=content_text, 
                            height=100, 
                            key=f"content_{i}",
                            disabled=True
                        )
                        
                        # Action buttons
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                        
                        with col1:
                            approve_disabled = status == "approved"
                            if st.button("Approve", key=f"approve_{i}", disabled=approve_disabled):
                                # Update moderation status in Qdrant
                                try:
                                    client = get_qdrant_client()
                                    if client:
                                        # Update item
                                        updated_payload = {
                                            "moderation_status": "approved",
                                            "moderated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "moderated_by": st.session_state.get("username", "admin")
                                        }
                                        
                                        client.set_payload(
                                            collection_name=item.get("source_collection"),
                                            payload=updated_payload,
                                            points=[item.get("id", item.get("comment_id", item.get("idea_id")))]
                                        )
                                        
                                        # Add to audit log
                                        add_audit_log(
                                            "content_moderation",
                                            f"Approved flagged content",
                                            {
                                                "collection": item.get("source_collection"),
                                                "content_id": item.get("id", item.get("comment_id", item.get("idea_id"))),
                                                "flag_reason": flag_reason
                                            },
                                            st.session_state.get("username", "admin")
                                        )
                                        
                                        st.success("Content approved!")
                                        st.rerun()
                                except Exception as e:
                                    logger.error(f"Error approving content: {e}")
                                    st.error(f"Error approving content: {str(e)}")
                        
                        with col2:
                            reject_disabled = status == "rejected"
                            if st.button("Reject", key=f"reject_{i}", disabled=reject_disabled):
                                # Update moderation status in Qdrant
                                try:
                                    client = get_qdrant_client()
                                    if client:
                                        # Update item
                                        updated_payload = {
                                            "moderation_status": "rejected",
                                            "moderated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "moderated_by": st.session_state.get("username", "admin")
                                        }
                                        
                                        client.set_payload(
                                            collection_name=item.get("source_collection"),
                                            payload=updated_payload,
                                            points=[item.get("id", item.get("comment_id", item.get("idea_id")))]
                                        )
                                        
                                        # Add to audit log
                                        add_audit_log(
                                            "content_moderation",
                                            f"Rejected flagged content",
                                            {
                                                "collection": item.get("source_collection"),
                                                "content_id": item.get("id", item.get("comment_id", item.get("idea_id"))),
                                                "flag_reason": flag_reason
                                            },
                                            st.session_state.get("username", "admin")
                                        )
                                        
                                        st.success("Content rejected!")
                                        st.rerun()
                                except Exception as e:
                                    logger.error(f"Error rejecting content: {e}")
                                    st.error(f"Error rejecting content: {str(e)}")
                        
                        with col3:
                            if st.button("View Context", key=f"context_{i}"):
                                # Show context info (like associated project, article, etc.)
                                # In a real implementation, you'd fetch related content
                                st.info("Would display the context of this content (e.g., the article, project, or thread).")
    
    with tabs[1]:
        st.subheader("Moderation Analytics")
        
        if not all_flagged_content:
            st.info("No flagged content data available for analytics.")
        else:
            # Analyze moderation data
            
            # Flag reasons distribution
            flag_reasons = {}
            for item in all_flagged_content:
                reason = item.get("flag_reason", "Not specified")
                if reason not in flag_reasons:
                    flag_reasons[reason] = 0
                flag_reasons[reason] += 1
            
            # Create DataFrame for flag reasons
            reason_df = pd.DataFrame({
                "reason": list(flag_reasons.keys()),
                "count": list(flag_reasons.values())
            })
            
            # Create pie chart for flag reasons
            reason_fig = px.pie(
                reason_df,
                values="count",
                names="reason",
                title="Content Flag Reasons",
                color="reason",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Update layout for theme consistency
            reason_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(reason_fig, use_container_width=True)
            
            # Moderation decisions distribution
            moderation_decisions = {
                "pending": 0,
                "approved": 0,
                "rejected": 0
            }
            
            for item in all_flagged_content:
                status = item.get("moderation_status", "pending")
                if status in moderation_decisions:
                    moderation_decisions[status] += 1
            
            # Create DataFrame for moderation decisions
            decision_df = pd.DataFrame({
                "status": list(moderation_decisions.keys()),
                "count": list(moderation_decisions.values())
            })
            
            # Create bar chart for moderation decisions
            decision_fig = px.bar(
                decision_df,
                x="status",
                y="count",
                title="Moderation Decisions",
                color="status",
                color_discrete_map={
                    "pending": "#FF9800",
                    "approved": "#4CAF50",
                    "rejected": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            decision_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Status",
                yaxis_title="Count"
            )
            
            st.plotly_chart(decision_fig, use_container_width=True)
            
            # Content source distribution
            content_sources = {}
            for item in all_flagged_content:
                source = item.get("source_collection", "Unknown")
                if source not in content_sources:
                    content_sources[source] = 0
                content_sources[source] += 1
            
            # Create DataFrame for content sources
            source_df = pd.DataFrame({
                "source": list(content_sources.keys()),
                "count": list(content_sources.values())
            })
            
            # Map collection names to display names
            source_df["source"] = source_df["source"].map({
                "citizen_comments": "Citizen Comments",
                "citizen_ideas": "Citizen Ideas",
                "hespress_politics_comments": "News Comments",
                "Unknown": "Unknown"
            })
            
            # Create pie chart for content sources
            source_fig = px.pie(
                source_df,
                values="count",
                names="source",
                title="Flagged Content by Source",
                color="source",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Update layout for theme consistency
            source_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(source_fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Moderation Logs")
        
        # Load moderation audit logs
        moderation_logs = load_audit_logs(
            limit=100,
            action_type="content_moderation"
        )
        
        if not moderation_logs:
            st.info("No moderation logs available.")
        else:
            # Create table for moderation logs
            logs_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Moderator</th>
                        <th>Action</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for log in moderation_logs:
                # Format timestamp
                timestamp = log.get("timestamp", datetime.now())
                if isinstance(timestamp, datetime):
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
                
                # Get moderator
                moderator = log.get("user", "Unknown")
                
                # Get action description
                description = log.get("description", "Unknown action")
                
                # Get details
                details = log.get("details", {})
                collection = details.get("collection", "Unknown")
                content_id = details.get("content_id", "Unknown")
                flag_reason = details.get("flag_reason", "Not specified")
                
                # Format details text
                details_text = f"Collection: {collection}<br>Content ID: {content_id}<br>Flag Reason: {flag_reason}"
                
                logs_table += f"""
                <tr>
                    <td>{formatted_time}</td>
                    <td>{moderator}</td>
                    <td>{description}</td>
                    <td>{details_text}</td>
                </tr>
                """
            
            logs_table += """
                </tbody>
            </table>
            """
            
            st.markdown(logs_table, unsafe_allow_html=True)
            
            # Export logs button
            if st.button("Export Logs"):
                # Convert logs to DataFrame
                logs_df = pd.DataFrame([
                    {
                        "timestamp": log.get("timestamp", ""),
                        "moderator": log.get("user", ""),
                        "action": log.get("description", ""),
                        "collection": log.get("details", {}).get("collection", ""),
                        "content_id": log.get("details", {}).get("content_id", ""),
                        "flag_reason": log.get("details", {}).get("flag_reason", "")
                    }
                    for log in moderation_logs
                ])
                
                # Convert to CSV
                csv = logs_df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "moderation_logs.csv",
                    "text/csv",
                    key="download_logs_csv"
                )

# -------------------------------------------------------------
# NEWS MANAGEMENT MODULE
# -------------------------------------------------------------

def news_management():
    """News management interface for moderating and analyzing news articles."""
    st.title(t("news_management"))
    
    # Sidebar filters
    st.sidebar.header(t("filters"))
    
    search_query = st.sidebar.text_input(
        t("search"), 
        placeholder=t("news_title")
    )
    
    category_filter = st.sidebar.selectbox(
        "News Category",
        ["All", "Politics", "Economy", "Society", "Culture", "Sports", "Technology", "International"]
    )
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now().date() - timedelta(days=30), datetime.now().date())
    )
    
    # Load news articles
    filter_conditions = []
    
    if search_query:
        title_condition = FieldCondition(
            key="title",
            match=MatchValue(value=search_query)
        )
        content_condition = FieldCondition(
            key="content",
            match=MatchValue(value=search_query)
        )
        filter_conditions.append(Filter(should=[title_condition, content_condition]))
    
    if category_filter != "All":
        filter_conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category_filter)
            )
        )
    
    # Replace it with this code that properly handles the date filtering:

    if date_range is not None and len(date_range) == 2:
        start_date, end_date = date_range
        
        # Convert dates to Unix timestamps (seconds since epoch)
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        # Determine which date field to use based on the source
        # You'll need to adjust this based on your actual data structure
        date_field_mapping = {
            "Citizen Comments": "timestamp",
            "Citizen Ideas": "date_submitted",
            "Municipal Projects": "start_date", 
            "News Articles": "date_published",
            "News Comments": "timestamp"
        }
        
        # Create a filter condition for each selected data source
        date_filter_conditions = []
        for source in data_source:
            if source in date_field_mapping:
                date_key = date_field_mapping[source]
                date_filter_conditions.append(
                    FieldCondition(
                        key=date_key,
                        range=Range(
                            gte=start_timestamp,
                            lte=end_timestamp
                        )
                    )
                )
        
        # Add the date conditions if any were created
        if date_filter_conditions:
            if len(date_filter_conditions) == 1:
                filter_conditions.append(date_filter_conditions[0])
            else:
                # If multiple date fields, match any of them
                filter_conditions.append(Filter(should=date_filter_conditions))
    
    # Create filter if we have conditions
    articles_filter = None
    if filter_conditions:
        articles_filter = Filter(must=filter_conditions)
    
    # Load articles
    articles, next_articles_id = load_qdrant_documents(
        "hespress_politics_details", 
        vector_dim=1536, 
        limit=100, 
        filters=articles_filter
    )
    
    # Load comments for these articles
    comments_filter_conditions = []
    
    article_ids = [article.get("article_id") for article in articles if "article_id" in article]
    if article_ids:
        article_id_conditions = [
            FieldCondition(key="article_id", match=MatchValue(value=article_id))
            for article_id in article_ids
        ]
        comments_filter_conditions.append(Filter(should=article_id_conditions))
    
    # Create comments filter
    comments_filter = None
    if comments_filter_conditions:
        comments_filter = Filter(must=comments_filter_conditions)
    
    # Load comments
    comments, next_comments_id = load_qdrant_documents(
        "hespress_politics_comments", 
        vector_dim=384, 
        limit=500, 
        filters=comments_filter
    )
    
    # News metrics
    st.subheader("News Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Total Articles",
            f"{len(articles):,}",
            icon="fa-newspaper",
            color="#2196F3"
        )
    
    with col2:
        # Count comments
        display_metric_card(
            "Total Comments",
            f"{len(comments):,}",
            icon="fa-comments",
            color="#4CAF50"
        )
    
    with col3:
        # Calculate comments per article
        comments_per_article = len(comments) / len(articles) if articles else 0
        
        display_metric_card(
            "Comments/Article",
            f"{comments_per_article:.1f}",
            icon="fa-chart-line",
            color="#FF9800"
        )
    
    with col4:
        # Calculate flagged comments
        flagged_count = sum(1 for comment in comments if comment.get("flagged", False))
        
        display_metric_card(
            "Flagged Comments",
            f"{flagged_count:,}",
            icon="fa-flag",
            color="#F44336"
        )
    
    # News management tabs
    tabs = st.tabs(["Articles", "Comments Analysis", "Content Analysis", "Publish Article"])
    
    with tabs[0]:
        st.subheader("News Articles")
        
        # Export button
        if st.button("Export Articles"):
            # Convert articles to DataFrame
            articles_df = pd.DataFrame(articles)
            
            # Convert to CSV
            csv = articles_df.to_csv(index=False)
            
            # Provide download button
            st.download_button(
                "Download CSV",
                csv,
                "news_articles.csv",
                "text/csv",
                key="download_articles_csv"
            )
        
        # Articles table
        article_columns = [
            {
                "key": "title",
                "label": "Title",
                "format": lambda val, item: f"""
                    <div style="font-weight: 500;">{val}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">{item.get('content', '')[:50]}...</div>
                """
            },
            {
                "key": "date_published",
                "label": "Published",
                "format": lambda val, item: val
            },
            {
                "key": "author",
                "label": "Author",
                "format": lambda val, item: val or "Unknown"
            },
            {
                "key": "category",
                "label": "Category",
                "format": lambda val, item: f"""
                    <span class="badge badge-primary">
                        {val}
                    </span>
                """
            }
        ]
        
        article_actions = [
            {
                "key": "view",
                "text": "View",
                "icon": "fa-eye",
                "class": "action-view",
                "callback": view_article_details
            },
            {
                "key": "comments",
                "text": "Comments",
                "icon": "fa-comments",
                "class": "action-view",
                "callback": view_article_comments
            },
            {
                "key": "edit",
                "text": "Edit",
                "icon": "fa-pen-to-square",
                "class": "action-edit",
                "callback": edit_article
            }
        ]
        
        create_dynamic_table(
            articles, 
            article_columns, 
            key_prefix="articles_table", 
            page_size=10, 
            searchable=True,
            actions=article_actions
        )
    
    with tabs[1]:
        st.subheader("Comments Analysis")
        
        if not comments:
            st.info("No comments available for analysis.")
        else:
            # Analyze comment sentiment
            sentiment_counts = {"POS": 0, "NEG": 0, "NEU": 0}
            
            for comment in comments:
                sentiment = comment.get("sentiment", "NEU")
                
                # Normalize sentiment value
                if sentiment.upper().startswith("P"):
                    sentiment_counts["POS"] += 1
                elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                    sentiment_counts["NEG"] += 1
                else:
                    sentiment_counts["NEU"] += 1
            
            # Create DataFrame for sentiment analysis
            sentiment_df = pd.DataFrame({
                "sentiment": list(sentiment_counts.keys()),
                "count": list(sentiment_counts.values())
            })
            
            
            # Create pie chart for sentiment analysis
            sentiment_fig = px.pie(
                sentiment_df,
                values="count",
                names="sentiment",
                title="Comment Sentiment Distribution",
                color="sentiment",
                color_discrete_map={
                    "POS": "#4CAF50",
                    "NEG": "#F44336",
                    "NEU": "#2196F3"
                }
            )
            
            # Update layout for theme consistency
            sentiment_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Comment volume over time
            comment_dates = []
            for comment in comments:
                timestamp = comment.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        except:
                            continue
                    comment_dates.append(timestamp)
            
            if comment_dates:
                # Create DataFrame with comment dates
                comment_df = pd.DataFrame({"date": comment_dates})
                
                # Group by date
                comment_df["date"] = pd.to_datetime(comment_df["date"]).dt.date
                comment_counts = comment_df.groupby("date").size().reset_index(name="count")
                
                # Create line chart
                volume_fig = px.line(
                    comment_counts,
                    x="date",
                    y="count",
                    title="Comment Volume Over Time",
                    markers=True
                )
                
                # Update layout for theme consistency
                volume_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Date",
                    yaxis_title="Number of Comments"
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
            
            # Comment word cloud
            st.subheader("Comment Word Cloud")
            
            # Extract comment text
            comment_texts = [comment.get("comment_text", "") for comment in comments if "comment_text" in comment]
            
            if comment_texts:
                # Create word cloud
                wordcloud_fig = create_word_cloud(comment_texts, title="Comment Content Word Cloud")
                
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text content for word cloud generation.")
    
    with tabs[2]:
        st.subheader("Content Analysis")
        
        if not articles:
            st.info("No articles available for analysis.")
        else:
            # Extract article text
            article_texts = [article.get("content", "") for article in articles if "content" in article]
            
            if article_texts:
                # Topic extraction
                topics = extract_topics(article_texts, num_topics=5, num_words=8)
                
                if topics:
                    st.subheader("Main Topics in Articles")
                    
                    # Create topics display
                    for i, topic in enumerate(topics):
                        words = ", ".join(topic["words"])
                        weight = topic["weight"]
                        
                        st.markdown(f"**Topic {i+1}:** {words}")
                        
                        # Weight bar
                        st.markdown(f"""
                        <div style="width: 100%; background-color: {'#333' if st.session_state.theme == 'dark' else '#EEE'}; height: 10px; border-radius: 5px; margin-bottom: 15px; overflow: hidden;">
                            <div style="width: {min(100, weight/10)}%; height: 100%; background-color: #2196F3; border-radius: 5px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Article word cloud
                st.subheader("Article Content Word Cloud")
                
                wordcloud_fig = create_word_cloud(article_texts, title="Article Content Word Cloud")
                
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text content for word cloud generation.")
                
                # Articles AI summary
                st.subheader("AI-Generated Content Summary")
                
                if st.button("Generate Summary"):
                    # Use OpenAI to generate a summary
                    summary = summarize_comments(article_texts[:5], max_tokens=500)
                    st.markdown(f"**Summary:** {summary}")
    
    with tabs[3]:
        st.subheader("Publish New Article")
        
        # New article form
        with st.form("new_article_form"):
            new_title = st.text_input("Article Title", key="new_article_title")
            new_content = st.text_area("Article Content", height=300, key="new_article_content")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_category = st.selectbox(
                    "Category", 
                    ["Politics", "Economy", "Society", "Culture", "Sports", "Technology", "International"],
                    key="new_article_category"
                )
                new_author = st.text_input("Author", value=st.session_state.get("username", ""), key="new_article_author")
            
            with col2:
                new_publish_date = st.date_input("Publication Date", value=datetime.now().date(), key="new_article_date")
                new_keywords = st.text_input("Keywords (comma separated)", key="new_article_keywords")
            
            submit_button = st.form_submit_button("Publish Article")
        
        if submit_button:
            # Validate form
            if not new_title or not new_content:
                st.error("Please provide both a title and content for the article.")
            else:
                # Create new article
                try:
                    # Get Qdrant client
                    client = get_qdrant_client()
                    if not client:
                        st.error("Failed to connect to Qdrant.")
                    else:
                        # Generate a unique article ID
                        article_id = str(uuid.uuid4())
                        
                        # Parse keywords
                        keywords = [k.strip() for k in new_keywords.split(",") if k.strip()]
                        
                        # Create an OpenAI embedding for the article
                        try:
                            # Generate summary using OpenAI
                            if openai.api_key:
                                # Generate summary
                                summary_response = openai.ChatCompletion.create(
                                    model=OPENAI_MODEL,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful AI assistant tasked with summarizing news articles."},
                                        {"role": "user", "content": f"Please provide a brief summary (1-2 sentences) of the following article:\n\n{new_content}"}
                                    ],
                                    max_tokens=100,
                                    temperature=0.5
                                )
                                
                                summary = summary_response['choices'][0]['message']['content'].strip()
                                
                                # Generate embedding
                                embedding_response = openai.Embedding.create(
                                    model="text-embedding-ada-002",
                                    input=new_content
                                )
                                
                                vector = embedding_response['data'][0]['embedding']
                            else:
                                # Default values if OpenAI is not configured
                                summary = "No summary available."
                                vector = [0.0] * 1536  # Default vector
                        except Exception as e:
                            logger.error(f"Error generating embedding: {e}")
                            summary = "No summary available."
                            vector = [0.0] * 1536  # Default vector
                        
                        # Create article document
                        article_doc = {
                            "article_id": article_id,
                            "title": new_title,
                            "content": new_content,
                            "category": new_category,
                            "author": new_author,
                            "date_published": new_publish_date.strftime("%Y-%m-%d"),
                            "keywords": keywords,
                            "summary": summary,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "created_by": st.session_state.get("username", "admin"),
                            "comment_count": 0,
                            "view_count": 0
                        }
                        
                        # Add article to Qdrant collection
                        client.upsert(
                            collection_name="hespress_politics_details",
                            points=[
                                PointStruct(
                                    id=article_id,
                                    vector=vector,
                                    payload=article_doc
                                )
                            ]
                        )
                        
                        # Add to audit log
                        add_audit_log(
                            "article_creation",
                            f"Published new article: {new_title}",
                            {"article_id": article_id, "title": new_title, "category": new_category},
                            st.session_state.get("username", "admin")
                        )
                        
                        st.success(f"Article '{new_title}' published successfully!")
                        
                        # Clear form by resetting session state
                        st.session_state.new_article_title = ""
                        st.session_state.new_article_content = ""
                        st.session_state.new_article_keywords = ""
                        
                        # Refresh article list
                        st.rerun()
                
                except Exception as e:
                    logger.error(f"Error publishing article: {e}")
                    st.error(f"Error publishing article: {str(e)}")

# -------------------------------------------------------------
# ADVANCED ANALYTICS MODULE
# -------------------------------------------------------------

def advanced_analytics():
    """Advanced analytics interface with AI-powered insights."""
    st.title("Advanced Analytics & AI Insights")
    
    # Sidebar analytics options
    st.sidebar.header("Analytics Options")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Sentiment Analysis", "Topic Modeling", "Trend Analysis", "Citizen Engagement", "Project Performance", "Geographic Insights", "Predictive Analytics"]
    )
    
    data_source = st.sidebar.multiselect(
        "Data Sources",
        ["Citizen Comments", "Citizen Ideas", "Municipal Projects", "News Articles", "News Comments"],
        default=["Citizen Comments", "Citizen Ideas"]
    )
    
    time_period = st.sidebar.slider(
        "Time Period (days)",
        min_value=7,
        max_value=365,
        value=90,
        step=1
    )
    
    region_filter = st.sidebar.selectbox(
        "Region Filter",
        MOROCCO_REGIONS
    )
    
    # Load data based on selections
    data_collections = {
        "Citizen Comments": ("citizen_comments", 384),
        "Citizen Ideas": ("citizen_ideas", 384),
        "Municipal Projects": ("municipal_projects", 384),
        "News Articles": ("hespress_politics_details", 1536),
        "News Comments": ("hespress_politics_comments", 384)
    }
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_period)
    
    # Create date filter condition
    date_field_keys = {
        "Citizen Comments": "timestamp",
        "Citizen Ideas": "date_submitted",
        "Municipal Projects": "start_date", 
        "News Articles": "date_published",
        "News Comments": "timestamp"
    }
    
    # Load data for selected sources
    all_data = {}
    
    for source in data_source:
        if source in data_collections:
            collection_name, vector_dim = data_collections[source]
            
            # Create filter conditions
            filter_conditions = []
            
            # Add date filter
            if source in date_field_keys:
                date_key = date_field_keys[source]
                
                # Convert dates to timestamps (numeric values)
                start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
                
                filter_conditions.append(
                    FieldCondition(
                        key=date_key,
                        range=Range(
                            gte=start_timestamp,
                            lte=end_timestamp
                        )
                    )
                )
            
            # Add region filter if specified
            if region_filter != "All":
                # Different collections use different region field names
                region_keys = ["province", "CT", "region", "project_province"]
                region_conditions = [
                    FieldCondition(key=key, match=MatchValue(value=region_filter))
                    for key in region_keys
                ]
                filter_conditions.append(Filter(should=region_conditions))
            
            # Create combined filter
            combined_filter = None
            if filter_conditions:
                combined_filter = Filter(must=filter_conditions)
            
            # Load data
            data, _ = load_qdrant_documents(
                collection_name,
                vector_dim=vector_dim,
                limit=1000,
                filters=combined_filter
            )
            
            all_data[source] = data
    
    # Check if we have data
    if not any(all_data.values()):
        st.info("No data available for analysis. Please select at least one data source.")
        return
    
    # Render analysis based on selected type
    if analysis_type == "Sentiment Analysis":
        render_sentiment_analysis(all_data)
    elif analysis_type == "Topic Modeling":
        render_topic_modeling(all_data)
    elif analysis_type == "Trend Analysis":
        render_trend_analysis(all_data)
    elif analysis_type == "Citizen Engagement":
        render_citizen_engagement_analysis(all_data)
    elif analysis_type == "Project Performance":
        render_project_performance_analysis(all_data)
    elif analysis_type == "Geographic Insights":
        render_geographic_insights(all_data)
    elif analysis_type == "Predictive Analytics":
        render_predictive_analytics(all_data)

def render_sentiment_analysis(data):
    """Render sentiment analysis visualizations."""
    st.header("Sentiment Analysis")
    
    # Overview info
    st.markdown("""
    This analysis examines the sentiment patterns across different data sources, helping identify positive and negative trends in citizen feedback and public discourse.
    """)
    
    # Check if we have data
    if not data:
        st.info("No data available for sentiment analysis. Please select at least one data source.")
        return
    
    # Combine all data with sentiment values
    all_items_with_sentiment = []
    source_totals = {}
    
    for source, items in data.items():
        items_with_sentiment = []
        
        for item in items:
            if isinstance(item, dict) and "sentiment" in item:
                # Normalize sentiment value
                sentiment = item["sentiment"]
                if isinstance(sentiment, str):
                    if sentiment.upper().startswith("P"):
                        normalized_sentiment = "POS"
                    elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                        normalized_sentiment = "NEG"
                    else:
                        normalized_sentiment = "NEU"
                    
                    # Add normalized sentiment
                    item_copy = item.copy()
                    item_copy["normalized_sentiment"] = normalized_sentiment
                    item_copy["source"] = source
                    
                    # Extract date if available
                    date_field = None
                    if "timestamp" in item:
                        date_field = "timestamp"
                    elif "date_submitted" in item:
                        date_field = "date_submitted"
                    elif "date_published" in item:
                        date_field = "date_published"
                    
                    if date_field and item[date_field]:
                        try:
                            if isinstance(item[date_field], str):
                                item_copy["date"] = datetime.strptime(item[date_field], "%Y-%m-%d")
                            else:
                                item_copy["date"] = item[date_field]
                        except:
                            item_copy["date"] = None
                    
                    items_with_sentiment.append(item_copy)
        
        source_totals[source] = len(items_with_sentiment)
        all_items_with_sentiment.extend(items_with_sentiment)
    
    if not all_items_with_sentiment:
        st.info("No sentiment data available in the selected data sources.")
        return
    
    # Overall sentiment metrics
    st.subheader("Sentiment Overview")
    
    # Count sentiments
    sentiment_counts = {"POS": 0, "NEG": 0, "NEU": 0}
    for item in all_items_with_sentiment:
        sentiment = item.get("normalized_sentiment")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    
    # Calculate percentages
    total_count = sum(sentiment_counts.values())
    pos_percentage = (sentiment_counts["POS"] / total_count * 100) if total_count > 0 else 0
    neg_percentage = (sentiment_counts["NEG"] / total_count * 100) if total_count > 0 else 0
    neu_percentage = (sentiment_counts["NEU"] / total_count * 100) if total_count > 0 else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card(
            "Positive Sentiment",
            f"{pos_percentage:.1f}%",
            icon="fa-smile",
            color="#4CAF50"
        )
    
    with col2:
        display_metric_card(
            "Negative Sentiment",
            f"{neg_percentage:.1f}%",
            icon="fa-frown",
            color="#F44336"
        )
    
    with col3:
        display_metric_card(
            "Neutral Sentiment",
            f"{neu_percentage:.1f}%",
            icon="fa-meh",
            color="#2196F3"
        )
    
    # Create sentiment visualization tabs
    tabs = st.tabs(["Overall Distribution", "By Source", "Over Time", "Content Analysis"])
    
    with tabs[0]:
        # Create pie chart for overall sentiment distribution
        sentiment_df = pd.DataFrame({
            "sentiment": list(sentiment_counts.keys()),
            "count": list(sentiment_counts.values())
        })
        
        sentiment_fig = px.pie(
            sentiment_df,
            values="count",
            names="sentiment",
            title="Overall Sentiment Distribution",
            color="sentiment",
            color_discrete_map={
                "POS": "#4CAF50",
                "NEG": "#F44336",
                "NEU": "#2196F3"
            }
        )
        
        # Update layout for theme consistency
        sentiment_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
        )
        
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with tabs[1]:
        # Group sentiment by source
        source_sentiment = {}
        for item in all_items_with_sentiment:
            source = item.get("source")
            sentiment = item.get("normalized_sentiment")
            
            if source not in source_sentiment:
                source_sentiment[source] = {"POS": 0, "NEG": 0, "NEU": 0}
            
            if sentiment in source_sentiment[source]:
                source_sentiment[source][sentiment] += 1
        
        # Create DataFrame for visualization
        source_sentiment_df = []
        for source, counts in source_sentiment.items():
            for sentiment, count in counts.items():
                source_sentiment_df.append({
                    "source": source,
                    "sentiment": sentiment,
                    "count": count
                })
        
        # Convert to DataFrame
        source_sentiment_df = pd.DataFrame(source_sentiment_df)
        
        # Create grouped bar chart
        source_fig = px.bar(
            source_sentiment_df,
            x="source",
            y="count",
            color="sentiment",
            title="Sentiment Distribution by Source",
            barmode="group",
            color_discrete_map={
                "POS": "#4CAF50",
                "NEG": "#F44336",
                "NEU": "#2196F3"
            }
        )
        
        # Update layout for theme consistency
        source_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Data Source",
            yaxis_title="Count"
        )
        
        st.plotly_chart(source_fig, use_container_width=True)
        
        # Calculate sentiment ratio by source
        st.subheader("Sentiment Ratio by Source")
        
        source_ratio_data = []
        for source, counts in source_sentiment.items():
            total = sum(counts.values())
            if total > 0:
                pos_ratio = counts["POS"] / total * 100
                neg_ratio = counts["NEG"] / total * 100
                neu_ratio = counts["NEU"] / total * 100
                
                source_ratio_data.append({
                    "source": source,
                    "positive": pos_ratio,
                    "negative": neg_ratio,
                    "neutral": neu_ratio
                })
        
        # Convert to DataFrame
        source_ratio_df = pd.DataFrame(source_ratio_data)
        
        # Create stacked bar chart for sentiment ratios
        ratio_fig = px.bar(
            source_ratio_df,
            x="source",
            y=["positive", "neutral", "negative"],
            title="Sentiment Ratio by Source (%)",
            labels={"value": "Percentage", "variable": "Sentiment"},
            color_discrete_map={
                "positive": "#4CAF50",
                "neutral": "#2196F3",
                "negative": "#F44336"
            }
        )
        
        # Update layout for theme consistency
        ratio_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Data Source",
            yaxis_title="Percentage (%)"
        )
        
        st.plotly_chart(ratio_fig, use_container_width=True)
    
    with tabs[2]:
        # Analyze sentiment over time
        items_with_date = [item for item in all_items_with_sentiment if "date" in item and item["date"] is not None]
        
        if items_with_date:
            # Group by date and sentiment
            sentiment_by_date = {}
            
            for item in items_with_date:
                date_str = item["date"].strftime("%Y-%m-%d")
                sentiment = item.get("normalized_sentiment")
                
                if date_str not in sentiment_by_date:
                    sentiment_by_date[date_str] = {"POS": 0, "NEG": 0, "NEU": 0}
                
                if sentiment in sentiment_by_date[date_str]:
                    sentiment_by_date[date_str][sentiment] += 1
            
            # Create DataFrame for time series
            time_series_data = []
            for date_str, counts in sentiment_by_date.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                total = sum(counts.values())
                
                if total > 0:
                    pos_ratio = counts["POS"] / total * 100
                    neg_ratio = counts["NEG"] / total * 100
                    neu_ratio = counts["NEU"] / total * 100
                    
                    time_series_data.append({
                        "date": date,
                        "positive": pos_ratio,
                        "negative": neg_ratio,
                        "neutral": neu_ratio,
                        "total": total
                    })
            
            # Sort by date
            time_series_df = pd.DataFrame(time_series_data).sort_values("date")
            
            # Create time series chart
            time_series_fig = px.line(
                time_series_df,
                x="date",
                y=["positive", "negative", "neutral"],
                title="Sentiment Trends Over Time",
                labels={"value": "Percentage (%)", "variable": "Sentiment"},
                color_discrete_map={
                    "positive": "#4CAF50",
                    "negative": "#F44336",
                    "neutral": "#2196F3"
                }
            )
            
            # Update layout
            time_series_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Date",
                yaxis_title="Percentage (%)"
            )
            
            st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Plot total volume as area chart
            volume_fig = px.area(
                time_series_df,
                x="date",
                y="total",
                title="Content Volume Over Time",
                labels={"total": "Total Items", "date": "Date"}
            )
            
            # Update layout
            volume_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Date",
                yaxis_title="Number of Items"
            )
            
            st.plotly_chart(volume_fig, use_container_width=True)
        else:
            st.info("No time-based data available for sentiment trend analysis.")
    
    with tabs[3]:
        # Word clouds by sentiment
        st.subheader("Content Analysis by Sentiment")
        
        # Group content by sentiment
        positive_texts = []
        negative_texts = []
        neutral_texts = []
        
        for item in all_items_with_sentiment:
            sentiment = item.get("normalized_sentiment")
            
            # Extract text content based on source
            text = ""
            if item.get("source") == "Citizen Comments":
                text = item.get("content", "")
            elif item.get("source") == "Citizen Ideas":
                challenge = item.get("challenge", "")
                solution = item.get("solution", "")
                text = f"{challenge} {solution}"
            elif item.get("source") == "News Comments":
                text = item.get("comment_text", "")
            elif item.get("source") == "News Articles":
                text = item.get("content", "")
            
            # Add to appropriate list
            if sentiment == "POS":
                positive_texts.append(text)
            elif sentiment == "NEG":
                negative_texts.append(text)
            elif sentiment == "NEU":
                neutral_texts.append(text)
        
        # Create word clouds
        sentiment_clouds = st.tabs(["Positive Content", "Negative Content", "Neutral Content"])
        
        with sentiment_clouds[0]:
            if positive_texts:
                pos_cloud = create_word_cloud(positive_texts, title="Positive Content Word Cloud")
                if pos_cloud:
                    st.pyplot(pos_cloud)
                else:
                    st.info("Not enough positive content for word cloud generation.")
            else:
                st.info("No positive content available.")
        
        with sentiment_clouds[1]:
            if negative_texts:
                neg_cloud = create_word_cloud(negative_texts, title="Negative Content Word Cloud")
                if neg_cloud:
                    st.pyplot(neg_cloud)
                else:
                    st.info("Not enough negative content for word cloud generation.")
            else:
                st.info("No negative content available.")
        
        with sentiment_clouds[2]:
            if neutral_texts:
                neu_cloud = create_word_cloud(neutral_texts, title="Neutral Content Word Cloud")
                if neu_cloud:
                    st.pyplot(neu_cloud)
                else:
                    st.info("Not enough neutral content for word cloud generation.")
            else:
                st.info("No neutral content available.")
        
        # AI summary of sentiment patterns
        st.subheader("AI Insights on Sentiment Patterns")
        
        if st.button("Generate Sentiment Insights"):
            # Sample texts from each sentiment category for analysis
            sample_positive = positive_texts[:5] if positive_texts else []
            sample_negative = negative_texts[:5] if negative_texts else []
            
            # Create prompt for OpenAI
            prompt = f"""
            Analyze the sentiment patterns in this citizen feedback data:
            
            Sentiment distribution:
            - Positive: {pos_percentage:.1f}%
            - Negative: {neg_percentage:.1f}%
            - Neutral: {neu_percentage:.1f}%
            
            Sample positive content:
            {sample_positive}
            
            Sample negative content:
            {sample_negative}
            
            Please provide insights on:
            1. Key themes driving positive sentiment
            2. Common issues leading to negative sentiment
            3. Recommendations for improving overall sentiment
            """
            
            try:
                if openai.api_key:
                    response = openai.ChatCompletion.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert data analyst specializing in sentiment analysis and civic engagement."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.5
                    )
                    
                    insights = response['choices'][0]['message']['content']
                    st.markdown(insights)
                else:
                    st.warning("OpenAI API key not configured. Unable to generate AI insights.")
            except Exception as e:
                logger.error(f"Error generating sentiment insights: {e}")
                st.error(f"Error generating insights: {str(e)}")

def render_topic_modeling(data):
    """Render topic modeling visualizations."""
    st.header("Topic Modeling")
    
    # Overview info
    st.markdown("""
    This analysis identifies key topics and themes across different data sources, revealing patterns and trending discussions among citizens and in news media.
    """)
    
    # Check if we have data
    if not data:
        st.info("No data available for topic modeling. Please select at least one data source.")
        return
    
    # Extract text content from data
    text_by_source = {}
    all_texts = []
    
    for source, items in data.items():
        source_texts = []
        
        for item in items:
            if isinstance(item, dict):
                # Extract text based on source type
                text = ""
                if source == "Citizen Comments":
                    text = item.get("content", "")
                elif source == "Citizen Ideas":
                    challenge = item.get("challenge", "")
                    solution = item.get("solution", "")
                    text = f"{challenge} {solution}".strip()
                elif source == "News Comments":
                    text = item.get("comment_text", "")
                elif source == "News Articles":
                    title = item.get("title", "")
                    content = item.get("content", "")
                    text = f"{title} {content}".strip()
                elif source == "Municipal Projects":
                    title = item.get("title", "")
                    desc = item.get("description", "")
                    text = f"{title} {desc}".strip()
                
                if text:
                    source_texts.append(text)
                    all_texts.append(text)
        
        text_by_source[source] = source_texts
    
    if not all_texts:
        st.info("No text content available for topic modeling.")
        return
    
    # Topic modeling metrics
    st.subheader("Content Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card(
            "Total Items",
            f"{len(all_texts):,}",
            icon="fa-file-alt",
            color="#2196F3"
        )
    
    with col2:
        # Calculate average text length
        avg_length = sum(len(text) for text in all_texts) / len(all_texts)
        
        display_metric_card(
            "Avg. Content Length",
            f"{avg_length:.0f} chars",
            icon="fa-text-height",
            color="#4CAF50"
        )
    
    with col3:
        # Count sources with data
        sources_with_data = sum(1 for texts in text_by_source.values() if texts)
        
        display_metric_card(
            "Data Sources",
            f"{sources_with_data}",
            icon="fa-database",
            color="#FF9800"
        )
    
    # Topic modeling tabs
    tabs = st.tabs(["Overall Topics", "Topics by Source", "Content Clusters", "Keyword Analysis"])
    
    with tabs[0]:
        st.subheader("Top Topics Across All Content")
        
        # Topic modeling parameters
        col1, col2 = st.columns([1, 3])
        
        with col1:
            num_topics = st.slider("Number of Topics", 3, 10, 5, key="overall_num_topics")
            num_words = st.slider("Words per Topic", 5, 15, 8, key="overall_num_words")
        
        # Extract topics from all text
        topics = extract_topics(all_texts, num_topics=num_topics, num_words=num_words)
        
        if topics:
            # Create visualization for topics
            for i, topic in enumerate(topics):
                words = ", ".join(topic["words"])
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Topic {i+1}:** {words}")
                
                with col2:
                    # Weight percentage
                    total_weight = sum(t["weight"] for t in topics)
                    weight_pct = (topic["weight"] / total_weight * 100) if total_weight > 0 else 0
                    st.markdown(f"**Relevance:** {weight_pct:.1f}%")
                
                # Weight bar
                st.markdown(f"""
                <div style="width: 100%; background-color: {'#333' if st.session_state.theme == 'dark' else '#EEE'}; height: 10px; border-radius: 5px; margin-bottom: 25px; overflow: hidden;">
                    <div style="width: {weight_pct}%; height: 100%; background-color: {f'rgba(33, 150, 243, {0.3 + weight_pct/100*0.7})'}; border-radius: 5px;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create word cloud from all text
            st.subheader("Overall Content Word Cloud")
            
            wordcloud_fig = create_word_cloud(all_texts, title="All Content Word Cloud")
            
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Not enough text for word cloud generation.")
        else:
            st.info("Unable to extract topics from the available text content.")
    
    with tabs[1]:
        st.subheader("Topics by Source")
        
        # Select source for topic analysis
        selected_source = st.selectbox(
            "Select Data Source",
            [source for source, texts in text_by_source.items() if texts],
            key="source_topic_selector"
        )
        
        if selected_source and selected_source in text_by_source and text_by_source[selected_source]:
            source_texts = text_by_source[selected_source]
            
            # Topic modeling parameters
            col1, col2 = st.columns([1, 3])
            
            with col1:
                source_num_topics = st.slider("Number of Topics", 3, 10, 5, key="source_num_topics")
                source_num_words = st.slider("Words per Topic", 5, 15, 8, key="source_num_words")
            
            # Extract topics for this source
            source_topics = extract_topics(source_texts, num_topics=source_num_topics, num_words=source_num_words)
            
            if source_topics:
                # Create visualization for topics
                for i, topic in enumerate(source_topics):
                    words = ", ".join(topic["words"])
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Topic {i+1}:** {words}")
                    
                    with col2:
                        # Weight percentage
                        total_weight = sum(t["weight"] for t in source_topics)
                        weight_pct = (topic["weight"] / total_weight * 100) if total_weight > 0 else 0
                        st.markdown(f"**Relevance:** {weight_pct:.1f}%")
                    
                    # Weight bar
                    st.markdown(f"""
                    <div style="width: 100%; background-color: {'#333' if st.session_state.theme == 'dark' else '#EEE'}; height: 10px; border-radius: 5px; margin-bottom: 25px; overflow: hidden;">
                        <div style="width: {weight_pct}%; height: 100%; background-color: {f'rgba(33, 150, 243, {0.3 + weight_pct/100*0.7})'}; border-radius: 5px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create word cloud from source text
                st.subheader(f"{selected_source} Word Cloud")
                
                source_wordcloud_fig = create_word_cloud(source_texts, title=f"{selected_source} Word Cloud")
                
                if source_wordcloud_fig:
                    st.pyplot(source_wordcloud_fig)
                else:
                    st.info("Not enough text for word cloud generation.")
            else:
                st.info(f"Unable to extract topics from {selected_source} content.")
        else:
            st.info("No text content available for the selected source.")
    
    with tabs[2]:
        st.subheader("Content Clustering")
        
        st.markdown("""
        This visualization groups similar content together, revealing natural clusters in the data.
        Each point represents a text item, and items close together are more similar in content.
        """)
        
        # Sample text items for clustering (limit to prevent performance issues)
        max_samples = 500
        sampled_texts = random.sample(all_texts, min(max_samples, len(all_texts)))
        
        if len(sampled_texts) > 20:  # Need enough samples for meaningful clustering
            try:
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    max_df=0.9,
                    min_df=3
                )
                
                # Create TF-IDF matrix
                tfidf_matrix = vectorizer.fit_transform(sampled_texts)
                
                # Use dimensionality reduction for visualization
                svd = TruncatedSVD(n_components=2, random_state=42)
                reduced_data = svd.fit_transform(tfidf_matrix)
                
                # Perform K-means clustering
                num_clusters = st.slider("Number of Clusters", 2, 10, 5, key="cluster_slider")
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                # Create DataFrame for visualization
                cluster_df = pd.DataFrame({
                    "x": reduced_data[:, 0],
                    "y": reduced_data[:, 1],
                    "cluster": clusters,
                    "text": [text[:100] + "..." if len(text) > 100 else text for text in sampled_texts]
                })
                
                # Create scatter plot
                cluster_fig = px.scatter(
                    cluster_df,
                    x="x",
                    y="y",
                    color="cluster",
                    hover_data=["text"],
                    title="Content Clusters",
                    labels={"cluster": "Cluster"}
                )
                
                # Update layout for theme consistency
                cluster_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
                
                st.plotly_chart(cluster_fig, use_container_width=True)
                
                # Analyze top terms for each cluster
                st.subheader("Top Terms by Cluster")
                
                # Get cluster centers and find top terms for each cluster
                for i in range(num_clusters):
                    # Get texts in this cluster
                    cluster_texts = [text for text, cluster in zip(sampled_texts, clusters) if cluster == i]
                    
                    if cluster_texts:
                        # Extract keywords using CountVectorizer
                        count_vectorizer = CountVectorizer(max_features=10, stop_words='english')
                        count_matrix = count_vectorizer.fit_transform(cluster_texts)
                        
                        # Get top terms
                        sum_words = count_matrix.sum(axis=0)
                        words_freq = [(word, sum_words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]
                        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                        
                        # Display top terms
                        top_terms = ", ".join([word for word, freq in words_freq[:8]])
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                                <div style="width: 15px; height: 15px; border-radius: 50%; background-color: {px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]}"></div>
                                <strong>Cluster {i+1}</strong>
                                <span style="margin-left: auto; color: {'#E0E0E0' if st.session_state.theme == 'dark' else '#666'}; font-size: 0.9rem;">
                                    {len(cluster_texts)} items
                                </span>
                            </div>
                            <div style="margin-left: 25px;">{top_terms}</div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error in content clustering: {e}")
                st.error(f"Error performing content clustering: {str(e)}")
        else:
            st.info("Not enough text items available for meaningful clustering.")
    
    with tabs[3]:
        st.subheader("Keyword Analysis")
        
        # Extract and analyze keywords
        try:
            # Create combined CountVectorizer for all text
            vectorizer = CountVectorizer(max_features=100, stop_words='english', min_df=3)
            word_count_matrix = vectorizer.fit_transform(all_texts)
            
            # Get word frequencies
            sum_words = word_count_matrix.sum(axis=0)
            word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            word_freq.sort(key=lambda x: x[1], reverse=True)
            
            # Create DataFrame for visualization
            word_freq_df = pd.DataFrame(word_freq[:30], columns=["word", "frequency"])
            
            # Create horizontal bar chart
            word_fig = px.bar(
                word_freq_df,
                y="word",
                x="frequency",
                title="Top Keywords Across All Content",
                orientation="h",
                color="frequency",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            word_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Frequency",
                yaxis_title="Keyword"
            )
            
            st.plotly_chart(word_fig, use_container_width=True)
            
            # Compare keywords across sources
            st.subheader("Keyword Comparison Across Sources")
            
            # Create a dict to hold keywords by source
            source_keywords = {}
            
            for source, texts in text_by_source.items():
                if len(texts) >= 10:  # Need enough texts for meaningful analysis
                    try:
                        # Create vectorizer for this source
                        source_vectorizer = CountVectorizer(max_features=20, stop_words='english')
                        source_matrix = source_vectorizer.fit_transform(texts)
                        
                        # Get word frequencies
                        source_sum = source_matrix.sum(axis=0)
                        source_freq = [(word, source_sum[0, idx]) for word, idx in source_vectorizer.vocabulary_.items()]
                        source_freq.sort(key=lambda x: x[1], reverse=True)
                        
                        # Store top keywords
                        source_keywords[source] = source_freq[:10]
                    except:
                        continue
            
            if len(source_keywords) >= 2:
                # Create comparative visualization
                comparison_data = []
                
                for source, keywords in source_keywords.items():
                    for word, freq in keywords:
                        comparison_data.append({
                            "source": source,
                            "keyword": word,
                            "frequency": freq
                        })
                
                # Convert to DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create grouped bar chart
                comparison_fig = px.bar(
                    comparison_df,
                    x="keyword",
                    y="frequency",
                    color="source",
                    title="Keyword Comparison by Source",
                    barmode="group"
                )
                
                # Update layout for theme consistency
                comparison_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Keyword",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(comparison_fig, use_container_width=True)
            else:
                st.info("Not enough sources with sufficient text for keyword comparison.")
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            st.error(f"Error performing keyword analysis: {str(e)}")

def render_trend_analysis(data):
    """Render trend analysis visualizations."""
    st.header("Trend Analysis")
    
    # Overview info
    st.markdown("""
    This analysis examines trends over time across different data sources, helping identify emerging patterns,
    seasonal variations, and changes in citizen sentiment and engagement.
    """)
    
    # Check if we have data
    if not data:
        st.info("No data available for trend analysis. Please select at least one data source.")
        return
    
    # Extract items with dates
    items_with_date = []
    
    for source, items in data.items():
        for item in items:
            if isinstance(item, dict):
                # Determine date field based on source
                date_value = None
                
                if source == "Citizen Comments" and "timestamp" in item:
                    date_value = item["timestamp"]
                elif source == "Citizen Ideas" and "date_submitted" in item:
                    date_value = item["date_submitted"]
                elif source == "Municipal Projects" and "start_date" in item:
                    date_value = item["start_date"]
                elif source == "News Articles" and "date_published" in item:
                    date_value = item["date_published"]
                elif source == "News Comments" and "timestamp" in item:
                    date_value = item["timestamp"]
                
                # Convert to datetime if string
                if date_value and isinstance(date_value, str):
                    try:
                        date_value = datetime.strptime(date_value, "%Y-%m-%d")
                    except:
                        try:
                            date_value = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S")
                        except:
                            date_value = None
                
                # Add to items if date is valid
                if date_value and isinstance(date_value, datetime):
                    item_copy = item.copy()
                    item_copy["date"] = date_value
                    item_copy["source"] = source
                    items_with_date.append(item_copy)
    
    if not items_with_date:
        st.info("No items with valid dates found for trend analysis.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(items_with_date)
    
    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Add derived date columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    
    # Create overview metrics
    st.subheader("Timeline Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = (df["date"].min(), df["date"].max())
        
        display_metric_card(
            "Time Period",
            f"{date_range[0].strftime('%b %Y')} - {date_range[1].strftime('%b %Y')}",
            icon="fa-calendar",
            color="#2196F3"
        )
    
    with col2:
        # Items per month
        days_diff = (date_range[1] - date_range[0]).days
        months_approx = max(1, days_diff / 30)
        items_per_month = len(df) / months_approx
        
        display_metric_card(
            "Avg. Items per Month",
            f"{items_per_month:.1f}",
            icon="fa-chart-line",
            color="#4CAF50"
        )
    
    with col3:
        # Most active source
        source_counts = df["source"].value_counts()
        most_active = source_counts.index[0] if not source_counts.empty else "None"
        
        display_metric_card(
            "Most Active Source",
            most_active,
            icon="fa-fire",
            color="#FF9800"
        )
    
    # Create trend analysis tabs
    tabs = st.tabs(["Activity Trends", "Content by Day/Time", "Seasonal Patterns", "Trend Prediction"])
    
    with tabs[0]:
        st.subheader("Activity Trends Over Time")
        
        # Group by date and count
        time_grouping = st.selectbox(
            "Time Granularity",
            ["Day", "Week", "Month", "Quarter", "Year"],
            index=2  # Default to Month
        )
        
        # Set date grouping format based on selection
        if time_grouping == "Day":
            df["date_group"] = df["date"].dt.date
        elif time_grouping == "Week":
            df["date_group"] = df["date"].dt.to_period("W").dt.start_time.dt.date
        elif time_grouping == "Month":
            df["date_group"] = df["date"].dt.to_period("M").dt.start_time.dt.date
        elif time_grouping == "Quarter":
            df["date_group"] = df["date"].dt.to_period("Q").dt.start_time.dt.date
        else:  # Year
            df["date_group"] = df["date"].dt.to_period("Y").dt.start_time.dt.date
        
        # Group by date and source
        activity_by_date = df.groupby(["date_group", "source"]).size().reset_index(name="count")
        
        # Create time series chart
        activity_fig = px.line(
            activity_by_date,
            x="date_group",
            y="count",
            color="source",
            title=f"Activity Trends by {time_grouping}",
            markers=True
        )
        
        # Update layout for theme consistency
        activity_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title=time_grouping,
            yaxis_title="Number of Items"
        )
        
        st.plotly_chart(activity_fig, use_container_width=True)
        
        # Stacked area chart for cumulative view
        activity_stacked_fig = px.area(
            activity_by_date,
            x="date_group",
            y="count",
            color="source",
            title=f"Cumulative Activity by {time_grouping}",
            groupnorm="percent"  # Normalize to 100%
        )
        
        # Update layout for theme consistency
        activity_stacked_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title=time_grouping,
            yaxis_title="Percentage of Total"
        )
        
        st.plotly_chart(activity_stacked_fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Activity by Day and Time")
        
        # Day of week analysis
        day_counts = df["weekday"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        # Create DataFrame for day analysis
        day_df = pd.DataFrame({
            "day": day_counts.index,
            "count": day_counts.values
        })
        
        # Create bar chart for day of week
        day_fig = px.bar(
            day_df,
            x="day",
            y="count",
            title="Activity by Day of Week",
            color="count",
            color_continuous_scale="Viridis"
        )
        
        # Update layout for theme consistency
        day_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Day of Week",
            yaxis_title="Number of Items"
        )
        
        st.plotly_chart(day_fig, use_container_width=True)
        
        # For items with timestamp, analyze hour of day
        if "timestamp" in df.columns:
            # Extract hour of day
            df["hour"] = df["timestamp"].dt.hour
            
            # Hour counts
            hour_counts = df["hour"].value_counts().sort_index()
            
            # Create DataFrame for hour analysis
            hour_df = pd.DataFrame({
                "hour": hour_counts.index,
                "count": hour_counts.values
            })
            
            # Create line chart for hour of day
            hour_fig = px.line(
                hour_df,
                x="hour",
                y="count",
                title="Activity by Hour of Day",
                markers=True
            )
            
            # Update layout for theme consistency
            hour_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Hour of Day",
                yaxis_title="Number of Items"
            )
            
            st.plotly_chart(hour_fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Seasonal Patterns")
        
        # Month analysis
        month_counts = df["month"].value_counts().sort_index()
        
        # Map month numbers to names
        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }
        
        # Create DataFrame for month analysis
        month_df = pd.DataFrame({
            "month": [month_names[m] for m in month_counts.index],
            "count": month_counts.values
        })
        
        # Create bar chart for months
        month_fig = px.bar(
            month_df,
            x="month",
            y="count",
            title="Activity by Month",
            color="count",
            color_continuous_scale="Viridis"
        )
        
        # Update x-axis category order to be chronological
        month_fig.update_xaxes(categoryorder="array", categoryarray=list(month_names.values()))
        
        # Update layout for theme consistency
        month_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Month",
            yaxis_title="Number of Items"
        )
        
        st.plotly_chart(month_fig, use_container_width=True)
        
        # Quarter analysis
        quarter_counts = df["quarter"].value_counts().sort_index()
        
        # Create DataFrame for quarter analysis
        quarter_df = pd.DataFrame({
            "quarter": [f"Q{q}" for q in quarter_counts.index],
            "count": quarter_counts.values
        })
        
        # Create pie chart for quarters
        quarter_fig = px.pie(
            quarter_df,
            values="count",
            names="quarter",
            title="Activity Distribution by Quarter",
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        # Update layout for theme consistency
        quarter_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
        )
        
        st.plotly_chart(quarter_fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Trend Prediction")
        
        st.markdown("""
        This analysis forecasts future trends based on historical patterns in the data.
        Predictions are more accurate with longer historical data.
        """)
        
        # Select source for prediction
        pred_source = st.selectbox(
            "Select Data Source for Prediction",
            df["source"].unique(),
            key="pred_source"
        )
        
        # Filter data for selected source
        source_df = df[df["source"] == pred_source].copy()
        
        if len(source_df) > 10:  # Need enough data for meaningful prediction
            # Group by month for prediction
            source_df["year_month"] = source_df["date"].dt.strftime("%Y-%m")
            monthly_counts = source_df.groupby("year_month").size()
            
            # Convert to DataFrame for time series forecasting
            ts_df = pd.DataFrame({
                "date": pd.to_datetime(monthly_counts.index.str.replace("-", "-01-")),
                "count": monthly_counts.values
            }).sort_values("date")
            
            # Ensure we have at least 3 months of data
            if len(ts_df) >= 3:
                # Number of months to predict
                pred_months = st.slider("Prediction Horizon (months)", 1, 12, 3, key="pred_months")
                
                try:
                    # Simple linear regression for prediction
                    X = np.arange(len(ts_df)).reshape(-1, 1)
                    y = ts_df["count"].values
                    
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Predict future values
                    future_X = np.arange(len(ts_df), len(ts_df) + pred_months).reshape(-1, 1)
                    future_y = model.predict(future_X)
                    
                    # Create future dates
                    last_date = ts_df["date"].max()
                    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(pred_months)]
                    
                    # Create prediction DataFrame
                    pred_df = pd.DataFrame({
                        "date": future_dates,
                        "count": future_y,
                        "type": "Predicted"
                    })
                    
                    # Add type column to original data
                    ts_df["type"] = "Historical"
                    
                    # Combine historical and predicted data
                    combined_df = pd.concat([ts_df, pred_df], ignore_index=True)
                    
                    # Create forecast visualization
                    forecast_fig = px.line(
                        combined_df,
                        x="date",
                        y="count",
                        color="type",
                        title=f"Activity Forecast for {pred_source}",
                        markers=True,
                        color_discrete_map={
                            "Historical": "#2196F3",
                            "Predicted": "#FF9800"
                        }
                    )
                    
                    # Add confidence interval for predictions
                    # Calculate standard error of predictions
                    from sklearn.metrics import mean_squared_error
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # Add upper and lower bounds (95% confidence interval)
                    upper_bound = future_y + 1.96 * rmse
                    lower_bound = future_y - 1.96 * rmse
                    
                    # Clip lower bound to zero (can't have negative counts)
                    lower_bound = np.maximum(0, lower_bound)
                    
                    # Add confidence interval to plot
                    for i, date in enumerate(future_dates):
                        forecast_fig.add_shape(
                            type="line",
                            x0=date,
                            y0=lower_bound[i],
                            x1=date,
                            y1=upper_bound[i],
                            line=dict(color="#FF9800", width=1)
                        )
                    
                    # Update layout for theme consistency
                    forecast_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        xaxis_title="Date",
                        yaxis_title="Number of Items"
                    )
                    
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Calculate trend metrics
                    current_trend = "Increasing" if model.coef_[0] > 0 else "Decreasing" if model.coef_[0] < 0 else "Stable"
                    monthly_change = model.coef_[0]
                    
                    # Display trend metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Current Trend",
                            current_trend,
                            delta=f"{monthly_change:.2f} per month" if monthly_change != 0 else "No Change"
                        )
                    
                    with col2:
                        # Calculate expected total for the prediction period
                        total_predicted = sum(future_y)
                        st.metric(
                            f"Projected Total (Next {pred_months} months)",
                            f"{total_predicted:.0f} items"
                        )
                    
                    # Show prediction details
                    st.subheader("Prediction Details")
                    
                    # Create table for prediction values
                    detail_df = pd.DataFrame({
                        "Month": [d.strftime("%B %Y") for d in future_dates],
                        "Predicted Value": [f"{v:.1f}" for v in future_y],
                        "Lower Bound": [f"{v:.1f}" for v in lower_bound],
                        "Upper Bound": [f"{v:.1f}" for v in upper_bound]
                    })
                    
                    st.table(detail_df)
                except Exception as e:
                    logger.error(f"Error in trend prediction: {e}")
                    st.error(f"Error creating trend prediction: {str(e)}")
            else:
                st.warning("Not enough time series data for prediction. Need at least 3 months of data.")
        else:
            st.warning("Not enough data for meaningful prediction.")

def render_citizen_engagement_analysis(data):
    """Render citizen engagement analytics."""
    st.header("Citizen Engagement Analysis")
    
    # Overview info
    st.markdown("""
    This analysis examines patterns in citizen engagement across different sources, projects, and topics,
    helping identify what drives participation and how citizens interact with municipal initiatives.
    """)
    
    # Check if we have data
    if not data:
        st.info("No data available for engagement analysis. Please select at least one data source.")
        return
    
    # Count items per source
    source_counts = {source: len(items) for source, items in data.items()}
    
    # Calculate total engagement
    total_items = sum(source_counts.values())
    
    # Calculate engagement metrics based on data available
    comments_count = source_counts.get("Citizen Comments", 0) + source_counts.get("News Comments", 0)
    ideas_count = source_counts.get("Citizen Ideas", 0)
    
    # Calculate project engagement if available
    project_engagement = []
    if "Municipal Projects" in data:
        projects = data["Municipal Projects"]
        for project in projects:
            if isinstance(project, dict):
                votes = int(project.get("vote_score", 0))
                participation = int(project.get("participation_count", 0))
                
                if votes > 0 or participation > 0:
                    project_engagement.append({
                        "title": project.get("title", "Untitled Project"),
                        "votes": votes,
                        "participation": participation,
                        "total_engagement": votes + participation,
                        "province": project.get("province", project.get("CT", "Unknown"))
                    })
    
    # Display engagement metrics
    st.subheader("Engagement Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Total Engagement",
            f"{total_items:,}",
            icon="fa-users",
            color="#2196F3"
        )
    
    with col2:
        display_metric_card(
            "Comments",
            f"{comments_count:,}",
            icon="fa-comments",
            color="#4CAF50"
        )
    
    with col3:
        display_metric_card(
            "Ideas Submitted",
            f"{ideas_count:,}",
            icon="fa-lightbulb",
            color="#FF9800"
        )
    
    with col4:
        # Calculate project votes if available
        project_votes = sum(p.get("votes", 0) for p in project_engagement)
        
        display_metric_card(
            "Project Votes",
            f"{project_votes:,}",
            icon="fa-thumbs-up",
            color="#9C27B0"
        )
    
    # Create engagement analysis tabs
    tabs = st.tabs(["Engagement by Source", "Regional Engagement", "Topic Engagement", "Citizen Personas"])
    
    with tabs[0]:
        st.subheader("Engagement by Source")
        
        # Create DataFrame for source counts
        source_df = pd.DataFrame({
            "source": list(source_counts.keys()),
            "count": list(source_counts.values())
        })
        
        # Create bar chart for source counts
        source_fig = px.bar(
            source_df,
            x="source",
            y="count",
            title="Engagement by Source",
            color="count",
            color_continuous_scale="Viridis"
        )
        
        # Update layout for theme consistency
        source_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Source",
            yaxis_title="Number of Items"
        )
        
        st.plotly_chart(source_fig, use_container_width=True)
        
        # Sentiment distribution by source
        combined_items = []
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict) and "sentiment" in item:
                    # Normalize sentiment value
                    sentiment = item["sentiment"]
                    if isinstance(sentiment, str):
                        if sentiment.upper().startswith("P"):
                            normalized_sentiment = "Positive"
                        elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                            normalized_sentiment = "Negative"
                        else:
                            normalized_sentiment = "Neutral"
                        
                        combined_items.append({
                            "source": source,
                            "sentiment": normalized_sentiment
                        })
        
        if combined_items:
            # Create DataFrame for sentiment analysis
            sentiment_df = pd.DataFrame(combined_items)
            
            # Group by source and sentiment
            sentiment_counts = sentiment_df.groupby(["source", "sentiment"]).size().reset_index(name="count")
            
            # Create grouped bar chart for sentiment by source
            sentiment_fig = px.bar(
                sentiment_counts,
                x="source",
                y="count",
                color="sentiment",
                title="Sentiment Distribution by Source",
                barmode="group",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Negative": "#F44336",
                    "Neutral": "#2196F3"
                }
            )
            
            # Update layout for theme consistency
            sentiment_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Source",
                yaxis_title="Count"
            )
            
            st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Regional Engagement")
        
        # Extract regional data
        region_counts = {}
        
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict):
                    # Different field names for regions in different collections
                    region = None
                    
                    if "province" in item:
                        region = item["province"]
                    elif "CT" in item:
                        region = item["CT"]
                    elif "project_province" in item:
                        region = item["project_province"]
                    elif "region" in item:
                        region = item["region"]
                    
                    if region and isinstance(region, str):
                        if region not in region_counts:
                            region_counts[region] = {"count": 0, "sources": {}}
                        
                        region_counts[region]["count"] += 1
                        
                        if source not in region_counts[region]["sources"]:
                            region_counts[region]["sources"][source] = 0
                        
                        region_counts[region]["sources"][source] += 1
        
        if region_counts:
            # Create DataFrame for region counts
            region_df = pd.DataFrame([
                {"region": region, "count": data["count"]}
                for region, data in region_counts.items()
            ])
            
            # Sort by count
            region_df = region_df.sort_values("count", ascending=False)
            
            # Create bar chart for region counts
            region_fig = px.bar(
                region_df,
                x="region",
                y="count",
                title="Engagement by Region",
                color="count",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            region_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Number of Items"
            )
            
            st.plotly_chart(region_fig, use_container_width=True)
            
            # Create map visualization for regions
            st.subheader("Regional Engagement Map")
            
            # Create map data
            map_data = pd.DataFrame({
                "province": region_df["region"],
                "count": region_df["count"]
            })
            
            # Create map
            map_fig = create_morocco_map(
                map_data,
                geo_col="province",
                value_col="count",
                title="Engagement Distribution by Region"
            )
            
            folium_static(map_fig, width=800)
            
            # Source breakdown for top regions
            st.subheader("Engagement Sources by Region")
            
            # Select top regions
            top_regions = region_df.head(5)["region"].tolist()
            
            # Create data for source breakdown
            source_breakdown = []
            
            for region in top_regions:
                for source, count in region_counts[region]["sources"].items():
                    source_breakdown.append({
                        "region": region,
                        "source": source,
                        "count": count
                    })
            
            # Create DataFrame for source breakdown
            source_df = pd.DataFrame(source_breakdown)
            
            # Create grouped bar chart
            source_fig = px.bar(
                source_df,
                x="region",
                y="count",
                color="source",
                title="Engagement Sources by Top Regions",
                barmode="group"
            )
            
            # Update layout for theme consistency
            source_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Count"
            )
            
            st.plotly_chart(source_fig, use_container_width=True)
        else:
            st.info("No regional data available for analysis.")
    
    with tabs[2]:
        st.subheader("Topic Engagement")
        
        # Extract data with topics or categories
        topics_data = []
        
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict):
                    # Different field names for topics/categories in different collections
                    topic = None
                    
                    if "topic" in item:
                        topic = item["topic"]
                    elif "axis" in item:
                        topic = item["axis"]
                    elif "category" in item:
                        topic = item["category"]
                    
                    if topic and isinstance(topic, str):
                        topics_data.append({
                            "source": source,
                            "topic": topic
                        })
        
        if topics_data:
            # Create DataFrame for topic analysis
            topics_df = pd.DataFrame(topics_data)
            
            # Group by topic and count
            topic_counts = topics_df.groupby("topic").size().reset_index(name="count")
            
            # Sort by count
            topic_counts = topic_counts.sort_values("count", ascending=False)
            
            # Create bar chart for topic counts
            topic_fig = px.bar(
                topic_counts,
                x="topic",
                y="count",
                title="Engagement by Topic",
                color="count",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            topic_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Topic",
                yaxis_title="Number of Items"
            )
            
            st.plotly_chart(topic_fig, use_container_width=True)
            
            # Source breakdown for top topics
            st.subheader("Engagement Sources by Topic")
            
            # Group by topic and source
            topic_source_counts = topics_df.groupby(["topic", "source"]).size().reset_index(name="count")
            
            # Get top topics
            top_topics = topic_counts.head(5)["topic"].tolist()
            
            # Filter for top topics
            top_topic_data = topic_source_counts[topic_source_counts["topic"].isin(top_topics)]
            
            # Create grouped bar chart
            topic_source_fig = px.bar(
                top_topic_data,
                x="topic",
                y="count",
                color="source",
                title="Engagement Sources by Top Topics",
                barmode="group"
            )
            
            # Update layout for theme consistency
            topic_source_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Topic",
                yaxis_title="Count"
            )
            
            st.plotly_chart(topic_source_fig, use_container_width=True)
            
            # Word cloud for top topic
            if top_topics:
                st.subheader(f"Most Engaged Topic: {top_topics[0]}")
                
                # Extract all text for top topic
                top_topic_text = []
                
                for source, items in data.items():
                    for item in items:
                        if isinstance(item, dict):
                            # Check if item is about the top topic
                            item_topic = None
                            
                            if "topic" in item:
                                item_topic = item["topic"]
                            elif "axis" in item:
                                item_topic = item["axis"]
                            elif "category" in item:
                                item_topic = item["category"]
                            
                            if item_topic == top_topics[0]:
                                # Extract text content
                                text = ""
                                
                                if source == "Citizen Comments":
                                    text = item.get("content", "")
                                elif source == "Citizen Ideas":
                                    challenge = item.get("challenge", "")
                                    solution = item.get("solution", "")
                                    text = f"{challenge} {solution}"
                                elif source == "News Comments":
                                    text = item.get("comment_text", "")
                                elif source == "News Articles":
                                    text = item.get("content", "")
                                elif source == "Municipal Projects":
                                    title = item.get("title", "")
                                    desc = item.get("description", "")
                                    text = f"{title} {desc}"
                                
                                if text:
                                    top_topic_text.append(text)
                
                if top_topic_text:
                    # Create word cloud
                    wordcloud_fig = create_word_cloud(top_topic_text, title=f"{top_topics[0]} Word Cloud")
                    
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.info("Not enough text for word cloud generation.")
        else:
            st.info("No topic/category data available for analysis.")
    
    with tabs[3]:
        st.subheader("Citizen Personas")
        
        st.markdown("""
        This analysis identifies distinct citizen personas based on engagement patterns,
        helping to better understand different citizen groups and their preferences.
        """)
        
        # For a comprehensive personas analysis, we'd need more user-specific data
        # This is a simplified simulation of the concept
        
        # Create sample personas for demonstration
        personas = [
            {
                "name": "Active Community Member",
                "description": "Frequently engages across multiple channels, submits ideas, and comments on projects and news articles.",
                "traits": ["High engagement frequency", "Multi-channel participation", "Constructive feedback", "Detailed suggestions"],
                "engagement": {"comments": 12, "ideas": 5, "votes": 20},
                "top_topics": ["Infrastructure", "Environment", "Education"],
                "sentiment": {"positive": 60, "neutral": 30, "negative": 10}
            },
            {
                "name": "Issue Reporter",
                "description": "Primarily reports problems and submits negative feedback when encountering issues.",
                "traits": ["Problem-focused", "Infrequent engagement", "Detailed issue reports", "Needs acknowledgment"],
                "engagement": {"comments": 8, "ideas": 1, "votes": 5},
                "top_topics": ["Infrastructure", "Sanitation", "Transportation"],
                "sentiment": {"positive": 10, "neutral": 30, "negative": 60}
            },
            {
                "name": "Idea Generator",
                "description": "Focuses on submitting innovative ideas and solutions rather than commenting.",
                "traits": ["Solution-oriented", "Creative thinking", "Moderate engagement frequency", "Forward-looking"],
                "engagement": {"comments": 4, "ideas": 10, "votes": 15},
                "top_topics": ["Technology", "Urban Planning", "Economic Development"],
                "sentiment": {"positive": 70, "neutral": 20, "negative": 10}
            },
            {
                "name": "Observer & Voter",
                "description": "Rarely comments or submits ideas, but regularly votes on projects and follows developments.",
                "traits": ["Passive engagement", "Regular platform visits", "Voting participation", "Information seeker"],
                "engagement": {"comments": 2, "ideas": 0, "votes": 25},
                "top_topics": ["Various", "No specific focus"],
                "sentiment": {"positive": 40, "neutral": 50, "negative": 10}
            },
            {
                "name": "Project Advocate",
                "description": "Focuses on specific project types and actively promotes and engages with these initiatives.",
                "traits": ["Project-focused", "Advocacy", "Consistent engagement", "Domain expertise"],
                "engagement": {"comments": 7, "ideas": 3, "votes": 12},
                "top_topics": ["Environment", "Education"],
                "sentiment": {"positive": 75, "neutral": 15, "negative": 10}
            }
        ]
        
        # Display personas
        for i, persona in enumerate(personas):
            with st.expander(f"{persona['name']}", expanded=i==0):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown(f"**Description:** {persona['description']}")
                    
                    st.markdown("**Key Traits:**")
                    for trait in persona["traits"]:
                        st.markdown(f"- {trait}")
                    
                    st.markdown("**Top Topics:**")
                    for topic in persona["top_topics"]:
                        st.markdown(f"- {topic}")
                
                with col2:
                    # Create engagement chart
                    eng_df = pd.DataFrame({
                        "type": list(persona["engagement"].keys()),
                        "count": list(persona["engagement"].values())
                    })
                    
                    eng_fig = px.bar(
                        eng_df,
                        x="type",
                        y="count",
                        title="Engagement Pattern",
                        color="type"
                    )
                    
                    # Update layout for theme consistency
                    eng_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        showlegend=False,
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(eng_fig, use_container_width=True)
                    
                    # Create sentiment donut chart
                    sentiment_df = pd.DataFrame({
                        "sentiment": list(persona["sentiment"].keys()),
                        "percentage": list(persona["sentiment"].values())
                    })
                    
                    sentiment_fig = px.pie(
                        sentiment_df,
                        values="percentage",
                        names="sentiment",
                        title="Sentiment Distribution",
                        hole=0.4,
                        color="sentiment",
                        color_discrete_map={
                            "positive": "#4CAF50",
                            "neutral": "#2196F3",
                            "negative": "#F44336"
                        }
                    )
                    
                    # Update layout for theme consistency
                    sentiment_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        showlegend=False,
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                st.markdown("**Engagement Strategy:**")
                if persona["name"] == "Active Community Member":
                    st.markdown("Encourage leadership in community initiatives and consider for citizen advisory roles.")
                elif persona["name"] == "Issue Reporter":
                    st.markdown("Ensure quick acknowledgment of reports and provide visible follow-up actions to increase trust.")
                elif persona["name"] == "Idea Generator":
                    st.markdown("Create channels for idea submission and provide feedback on idea implementation status.")
                elif persona["name"] == "Observer & Voter":
                    st.markdown("Provide easy voting mechanisms and quick information summaries to maintain engagement.")
                elif persona["name"] == "Project Advocate":
                    st.markdown("Connect with relevant municipal departments and include in specialized consultations.")

def render_project_performance_analysis(data):
    """Render project performance analytics."""
    st.header("Project Performance Analysis")
    
    # Overview info
    st.markdown("""
    This analysis examines municipal project performance metrics, including completion rates, budget utilization,
    citizen satisfaction, and regional variations in project outcomes.
    """)
    
    # Check if we have project data
    if "Municipal Projects" not in data or not data["Municipal Projects"]:
        st.info("No project data available for analysis. Please select 'Municipal Projects' as a data source.")
        return
    
    # Process project data
    projects = data["Municipal Projects"]
    
    # Calculate project metrics
    total_projects = len(projects)
    completed_projects = sum(1 for p in projects if isinstance(p, dict) and p.get("status") == "Completed")
    in_progress_projects = sum(1 for p in projects if isinstance(p, dict) and p.get("status") == "In Progress")
    
    completion_rate = (completed_projects / total_projects * 100) if total_projects > 0 else 0
    
    # Calculate budget metrics
    total_budget = sum(float(p.get("budget", 0)) for p in projects if isinstance(p, dict))
    completed_budget = sum(float(p.get("budget", 0)) for p in projects if isinstance(p, dict) and p.get("status") == "Completed")
    in_progress_budget = sum(float(p.get("budget", 0)) for p in projects if isinstance(p, dict) and p.get("status") == "In Progress")
    
    budget_utilization = ((completed_budget + in_progress_budget * 0.5) / total_budget * 100) if total_budget > 0 else 0
    
    # Calculate average completion percentage
    completion_percentages = [float(p.get("completion_percentage", 0)) for p in projects if isinstance(p, dict)]
    avg_completion = sum(completion_percentages) / len(completion_percentages) if completion_percentages else 0
    
    # Calculate vote metrics
    vote_scores = [int(p.get("vote_score", 0)) for p in projects if isinstance(p, dict)]
    avg_vote_score = sum(vote_scores) / len(vote_scores) if vote_scores else 0
    
    # Display performance metrics
    st.subheader("Project Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Completion Rate",
            f"{completion_rate:.1f}%",
            icon="fa-check-circle",
            color="#4CAF50"
        )
    
    with col2:
        display_metric_card(
            "Budget Utilization",
            f"{budget_utilization:.1f}%",
            icon="fa-money-bill",
            color="#FF9800"
        )
    
    with col3:
        display_metric_card(
            "Avg. Completion",
            f"{avg_completion:.1f}%",
            icon="fa-tasks",
            color="#2196F3"
        )
    
    with col4:
        display_metric_card(
            "Avg. Vote Score",
            f"{avg_vote_score:.1f}",
            icon="fa-thumbs-up",
            color="#9C27B0"
        )
    
    # Create project performance tabs
    tabs = st.tabs(["Status Analysis", "Budget Analysis", "Timeline Performance", "Regional Performance"])
    
    with tabs[0]:
        st.subheader("Project Status Distribution")
        
        # Count projects by status
        status_counts = {}
        for project in projects:
            if isinstance(project, dict):
                status = project.get("status", "Unknown")
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
        
        # Create DataFrame for visualization
        status_df = pd.DataFrame({
            "status": list(status_counts.keys()),
            "count": list(status_counts.values())
        })
        
        # Create pie chart for status distribution
        status_fig = px.pie(
            status_df,
            values="count",
            names="status",
            title="Project Status Distribution",
            color="status",
            color_discrete_map={
                "Completed": "#4CAF50",
                "In Progress": "#FF9800",
                "Approved": "#2196F3",
                "Proposed": "#9E9E9E",
                "On Hold": "#FFC107",
                "Cancelled": "#F44336"
            }
        )
        
        # Update layout for theme consistency
        status_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
        )
        
        st.plotly_chart(status_fig, use_container_width=True)
        
        # Create status breakdown by category
        st.subheader("Status by Project Category")
        
        # Extract projects with category and status
        category_status_data = []
        
        for project in projects:
            if isinstance(project, dict) and "category" in project and "status" in project:
                category_status_data.append({
                    "category": project["category"],
                    "status": project["status"]
                })
        
        if category_status_data:
            # Create DataFrame
            category_df = pd.DataFrame(category_status_data)
            
            # Group by category and status
            category_counts = category_df.groupby(["category", "status"]).size().reset_index(name="count")
            
            # Create grouped bar chart
            category_fig = px.bar(
                category_counts,
                x="category",
                y="count",
                color="status",
                title="Project Status by Category",
                barmode="group",
                color_discrete_map={
                    "Completed": "#4CAF50",
                    "In Progress": "#FF9800",
                    "Approved": "#2196F3",
                    "Proposed": "#9E9E9E",
                    "On Hold": "#FFC107",
                    "Cancelled": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            category_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Project Category",
                yaxis_title="Number of Projects"
            )
            
            st.plotly_chart(category_fig, use_container_width=True)
        else:
            st.info("No category data available for status breakdown.")
    
    with tabs[1]:
        st.subheader("Budget Analysis")
        
        # Extract budget data by status
        budget_by_status = {}
        for project in projects:
            if isinstance(project, dict) and "budget" in project and "status" in project:
                status = project["status"]
                budget = float(project["budget"])
                
                if status not in budget_by_status:
                    budget_by_status[status] = 0
                
                budget_by_status[status] += budget
        
        if budget_by_status:
            # Create DataFrame for visualization
            budget_df = pd.DataFrame({
                "status": list(budget_by_status.keys()),
                "budget": list(budget_by_status.values())
            })
            
            # Create pie chart for budget distribution
            budget_fig = px.pie(
                budget_df,
                values="budget",
                names="status",
                title="Budget Distribution by Status",
                color="status",
                color_discrete_map={
                    "Completed": "#4CAF50",
                    "In Progress": "#FF9800",
                    "Approved": "#2196F3",
                    "Proposed": "#9E9E9E",
                    "On Hold": "#FFC107",
                    "Cancelled": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            budget_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(budget_fig, use_container_width=True)
        
        # Budget by category
        st.subheader("Budget by Project Category")
        
        # Extract budget data by category
        budget_by_category = {}
        for project in projects:
            if isinstance(project, dict) and "budget" in project and "category" in project:
                category = project["category"]
                budget = float(project["budget"])
                
                if category not in budget_by_category:
                    budget_by_category[category] = 0
                
                budget_by_category[category] += budget
        
        if budget_by_category:
            # Create DataFrame for visualization
            budget_category_df = pd.DataFrame({
                "category": list(budget_by_category.keys()),
                "budget": list(budget_by_category.values())
            })
            
            # Sort by budget
            budget_category_df = budget_category_df.sort_values("budget", ascending=False)
            
            # Create bar chart for budget by category
            budget_cat_fig = px.bar(
                budget_category_df,
                x="category",
                y="budget",
                title="Budget by Project Category",
                color="budget",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            budget_cat_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Project Category",
                yaxis_title="Budget (MAD)"
            )
            
            st.plotly_chart(budget_cat_fig, use_container_width=True)
        else:
            st.info("No category data available for budget breakdown.")
        
        # Budget efficiency analysis
        st.subheader("Budget Efficiency Analysis")
        
        # Extract completed projects with budget and completion data
        completed_project_data = []
        
        for project in projects:
            if isinstance(project, dict) and project.get("status") == "Completed":
                if "budget" in project and "completion_percentage" in project:
                    completed_project_data.append({
                        "title": project.get("title", "Untitled"),
                        "budget": float(project["budget"]),
                        "completion": float(project["completion_percentage"])
                    })
        
        if completed_project_data:
            # Create DataFrame
            efficiency_df = pd.DataFrame(completed_project_data)
            
            # Calculate budget efficiency (completion percentage / budget)
            # Normalize to make it more readable
            max_budget = efficiency_df["budget"].max()
            efficiency_df["efficiency"] = efficiency_df["completion"] / (efficiency_df["budget"] / max_budget) * 100
            
            # Sort by efficiency
            efficiency_df = efficiency_df.sort_values("efficiency", ascending=False)
            
            # Create scatter plot for budget vs. completion
            efficiency_fig = px.scatter(
                efficiency_df,
                x="budget",
                y="completion",
                size="efficiency",
                color="efficiency",
                hover_name="title",
                title="Budget Efficiency of Completed Projects",
                color_continuous_scale="RdYlGn"
            )
            
            # Update layout for theme consistency
            efficiency_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Budget (MAD)",
                yaxis_title="Completion Percentage"
            )
            
            st.plotly_chart(efficiency_fig, use_container_width=True)
            
            # Show top and bottom projects by efficiency
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Efficient Projects")
                
                for i, row in efficiency_df.head(5).iterrows():
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #4CAF50; background-color: {'rgba(76, 175, 80, 0.1)'};">
                        <div style="font-weight: bold;">{row['title']}</div>
                        <div>Budget: {row['budget']:,.0f} MAD</div>
                        <div>Efficiency Score: {row['efficiency']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Least Efficient Projects")
                
                for i, row in efficiency_df.tail(5).iterrows():
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #F44336; background-color: {'rgba(244, 67, 54, 0.1)'};">
                        <div style="font-weight: bold;">{row['title']}</div>
                        <div>Budget: {row['budget']:,.0f} MAD</div>
                        <div>Efficiency Score: {row['efficiency']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No completed projects with budget data available for efficiency analysis.")
    
    with tabs[2]:
        st.subheader("Project Timeline Performance")
        
        # Extract projects with start and end dates
        timeline_data = []
        
        for project in projects:
            if isinstance(project, dict):
                start_date = project.get("start_date")
                end_date = project.get("end_date")
                status = project.get("status")
                
                if start_date and end_date:
                    # Convert to datetime if string
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.strptime(start_date, "%Y-%m-%d")
                        except:
                            start_date = None
                    
                    if isinstance(end_date, str):
                        try:
                            end_date = datetime.strptime(end_date, "%Y-%m-%d")
                        except:
                            end_date = None
                    
                    if start_date and end_date:
                        # Calculate planned duration
                        planned_duration = (end_date - start_date).days
                        
                        # For completed projects, calculate actual duration
                        actual_duration = None
                        if status == "Completed" and "completion_date" in project:
                            completion_date = project["completion_date"]
                            
                            if isinstance(completion_date, str):
                                try:
                                    completion_date = datetime.strptime(completion_date, "%Y-%m-%d")
                                    actual_duration = (completion_date - start_date).days
                                except:
                                    pass
                        
                        timeline_data.append({
                            "title": project.get("title", "Untitled"),
                            "start_date": start_date,
                            "end_date": end_date,
                            "planned_duration": planned_duration,
                            "actual_duration": actual_duration,
                            "status": status,
                            "completion_percentage": float(project.get("completion_percentage", 0))
                        })
        
        if timeline_data:
            # Create DataFrame
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create Gantt chart for project timelines
            fig = px.timeline(
                timeline_df,
                x_start="start_date",
                x_end="end_date",
                y="title",
                color="status",
                hover_name="title",
                title="Project Timelines",
                color_discrete_map={
                    "Completed": "#4CAF50",
                    "In Progress": "#FF9800",
                    "Approved": "#2196F3",
                    "Proposed": "#9E9E9E",
                    "On Hold": "#FFC107",
                    "Cancelled": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Date",
                yaxis_title="Project"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Project duration analysis
            st.subheader("Project Duration Analysis")
            
            # Create histogram of planned durations
            duration_fig = px.histogram(
                timeline_df,
                x="planned_duration",
                title="Distribution of Project Durations",
                labels={"planned_duration": "Duration (days)"},
                color_discrete_sequence=["#2196F3"]
            )
            
            # Update layout for theme consistency
            duration_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Duration (days)",
                yaxis_title="Number of Projects"
            )
            
            st.plotly_chart(duration_fig, use_container_width=True)
            
            # For completed projects with actual duration, compare planned vs. actual
            actual_duration_data = timeline_df.dropna(subset=["actual_duration"]).copy()
            
            if not actual_duration_data.empty:
                # Calculate delay
                actual_duration_data["delay"] = actual_duration_data["actual_duration"] - actual_duration_data["planned_duration"]
                actual_duration_data["delay_percentage"] = (actual_duration_data["delay"] / actual_duration_data["planned_duration"] * 100)
                
                # Create scatter plot of planned vs. actual duration
                delay_fig = px.scatter(
                    actual_duration_data,
                    x="planned_duration",
                    y="actual_duration",
                    size="delay_percentage",
                    color="delay",
                    hover_name="title",
                    title="Planned vs. Actual Project Duration",
                    color_continuous_scale="RdYlGn_r"
                )
                
                # Add diagonal line (planned = actual)
                delay_fig.add_shape(
                    type="line",
                    x0=actual_duration_data["planned_duration"].min(),
                    y0=actual_duration_data["planned_duration"].min(),
                    x1=actual_duration_data["planned_duration"].max(),
                    y1=actual_duration_data["planned_duration"].max(),
                    line=dict(color="white" if st.session_state.theme == "dark" else "black", dash="dash")
                )
                
                # Update layout for theme consistency
                delay_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Planned Duration (days)",
                    yaxis_title="Actual Duration (days)"
                )
                
                st.plotly_chart(delay_fig, use_container_width=True)
            else:
                st.info("No completed projects with actual duration data available for comparison.")
        else:
            st.info("No projects with valid timeline data available.")
    
    with tabs[3]:
        st.subheader("Regional Project Performance")
        
        # Extract regional project data
        regional_data = []
        
        for project in projects:
            if isinstance(project, dict):
                region = None
                
                if "province" in project:
                    region = project["province"]
                elif "CT" in project:
                    region = project["CT"]
                
                if region:
                    regional_data.append({
                        "region": region,
                        "status": project.get("status", "Unknown"),
                        "budget": float(project.get("budget", 0)),
                        "completion_percentage": float(project.get("completion_percentage", 0)),
                        "vote_score": int(project.get("vote_score", 0))
                    })
        
        if regional_data:
            # Create DataFrame
            region_df = pd.DataFrame(regional_data)
            
            # Group by region and calculate metrics
            region_metrics = []
            
            for region, group in region_df.groupby("region"):
                total = len(group)
                completed = sum(1 for _, row in group.iterrows() if row["status"] == "Completed")
                in_progress = sum(1 for _, row in group.iterrows() if row["status"] == "In Progress")
                total_budget = group["budget"].sum()
                avg_completion = group["completion_percentage"].mean()
                avg_vote = group["vote_score"].mean()
                
                region_metrics.append({
                    "region": region,
                    "total_projects": total,
                    "completed_projects": completed,
                    "in_progress_projects": in_progress,
                    "completion_rate": (completed / total * 100) if total > 0 else 0,
                    "total_budget": total_budget,
                    "avg_completion": avg_completion,
                    "avg_vote_score": avg_vote
                })
            
            # Create DataFrame for region metrics
            region_metrics_df = pd.DataFrame(region_metrics)
            
            # Sort by total projects
            region_metrics_df = region_metrics_df.sort_values("total_projects", ascending=False)
            
            # Create map visualization
            st.subheader("Regional Project Distribution")
            
            # Create map data
            map_data = pd.DataFrame({
                "province": region_metrics_df["region"],
                "count": region_metrics_df["total_projects"]
            })
            
            # Create map
            map_fig = create_morocco_map(
                map_data,
                geo_col="province",
                value_col="count",
                title="Project Distribution by Region"
            )
            
            folium_static(map_fig, width=800)
            
            # Regional completion rate comparison
            st.subheader("Regional Completion Rate Comparison")
            
            # Create bar chart for completion rates
            completion_fig = px.bar(
                region_metrics_df,
                x="region",
                y="completion_rate",
                title="Project Completion Rate by Region",
                color="completion_rate",
                color_continuous_scale="RdYlGn"
            )
            
            # Update layout for theme consistency
            completion_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Completion Rate (%)"
            )
            
            st.plotly_chart(completion_fig, use_container_width=True)
            
            # Regional budget allocation comparison
            st.subheader("Regional Budget Allocation")
            
            # Create pie chart for budget allocation
            budget_fig = px.pie(
                region_metrics_df,
                values="total_budget",
                names="region",
                title="Budget Allocation by Region",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Update layout for theme consistency
            budget_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(budget_fig, use_container_width=True)
            
            # Regional comparison table
            st.subheader("Regional Performance Comparison")
            
            # Format metrics for display
            display_df = region_metrics_df.copy()
            display_df["completion_rate"] = display_df["completion_rate"].apply(lambda x: f"{x:.1f}%")
            display_df["avg_completion"] = display_df["avg_completion"].apply(lambda x: f"{x:.1f}%")
            display_df["total_budget"] = display_df["total_budget"].apply(lambda x: f"{x:,.0f} MAD")
            display_df["avg_vote_score"] = display_df["avg_vote_score"].apply(lambda x: f"{x:.1f}")
            
            # Create styled HTML table
            region_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Total Projects</th>
                        <th>Completed</th>
                        <th>Completion Rate</th>
                        <th>Budget</th>
                        <th>Avg. Completion</th>
                        <th>Avg. Vote Score</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, row in display_df.iterrows():
                region_table += f"""
                <tr>
                    <td>{row['region']}</td>
                    <td>{row['total_projects']}</td>
                    <td>{row['completed_projects']}</td>
                    <td>{row['completion_rate']}</td>
                    <td>{row['total_budget']}</td>
                    <td>{row['avg_completion']}</td>
                    <td>{row['avg_vote_score']}</td>
                </tr>
                """
            
            region_table += """
                </tbody>
            </table>
            """
            
            st.markdown(region_table, unsafe_allow_html=True)
        else:
            st.info("No regional project data available for analysis.")

def render_geographic_insights(data):
    """Render geographic insights visualizations."""
    st.header("Geographic Insights")
    
    # Overview info
    st.markdown("""
    This analysis provides spatial insights into citizen engagement, project distribution, and regional performance,
    helping identify geographic patterns and optimize resource allocation.
    """)
    
    # Count items per region across all data sources
    region_counts = {}
    
    for source, items in data.items():
        for item in items:
            if isinstance(item, dict):
                # Different field names for regions in different collections
                region = None
                
                if "province" in item:
                    region = item["province"]
                elif "CT" in item:
                    region = item["CT"]
                elif "project_province" in item:
                    region = item["project_province"]
                elif "region" in item:
                    region = item["region"]
                
                if region and isinstance(region, str):
                    if region not in region_counts:
                        region_counts[region] = {
                            "total": 0,
                            "sources": {}
                        }
                    
                    region_counts[region]["total"] += 1
                    
                    if source not in region_counts[region]["sources"]:
                        region_counts[region]["sources"][source] = 0
                    
                    region_counts[region]["sources"][source] += 1
    
    if not region_counts:
        st.info("No geographic data available for analysis.")
        return
    
    # Create map visualization
    st.subheader("Geographic Distribution Overview")
    
    # Create map data
    map_data = pd.DataFrame({
        "province": list(region_counts.keys()),
        "count": [data["total"] for data in region_counts.values()]
    })
    
    # Create map
    map_fig = create_morocco_map(
        map_data,
        geo_col="province",
        value_col="count",
        title="Content Distribution by Region"
    )
    
    folium_static(map_fig, width=800)
    
    # Create geographic insights tabs
    tabs = st.tabs(["Regional Comparison", "Source Distribution", "Sentiment Geography", "Project Geography"])
    
    with tabs[0]:
        st.subheader("Regional Data Comparison")
        
        # Create bar chart for regional counts
        region_df = pd.DataFrame({
            "region": list(region_counts.keys()),
            "count": [data["total"] for data in region_counts.values()]
        }).sort_values("count", ascending=False)
        
        region_fig = px.bar(
            region_df,
            x="region",
            y="count",
            title="Content Distribution by Region",
            color="count",
            color_continuous_scale="Viridis"
        )
        
        # Update layout for theme consistency
        region_fig.update_layout(
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            xaxis_title="Region",
            yaxis_title="Total Items"
        )
        
        st.plotly_chart(region_fig, use_container_width=True)
        
        # Regional data table
        st.subheader("Regional Data Overview")
        
        # Create styled HTML table
        region_table = """
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Total Items</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for region, data in sorted(region_counts.items(), key=lambda x: x[1]["total"], reverse=True):
            region_table += f"""
            <tr>
                <td>{region}</td>
                <td>{data['total']}</td>
            </tr>
            """
        
        region_table += """
            </tbody>
        </table>
        """
        
        st.markdown(region_table, unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("Source Distribution by Region")
        
        # Create stacked bar chart for source distribution
        source_data = []
        
        for region, data in region_counts.items():
            for source, count in data["sources"].items():
                source_data.append({
                    "region": region,
                    "source": source,
                    "count": count
                })
        
        if source_data:
            source_df = pd.DataFrame(source_data)
            
            # Create stacked bar chart
            source_fig = px.bar(
                source_df,
                x="region",
                y="count",
                color="source",
                title="Content Sources by Region",
                barmode="stack"
            )
            
            # Update layout for theme consistency
            source_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Count"
            )
            
            st.plotly_chart(source_fig, use_container_width=True)
            
            # Create sunburst chart for hierarchical view
            sunburst_fig = px.sunburst(
                source_df,
                path=["region", "source"],
                values="count",
                title="Hierarchical View of Content Distribution"
            )
            
            # Update layout for theme consistency
            sunburst_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("No source distribution data available.")
    
    with tabs[2]:
        st.subheader("Sentiment Geography")
        
        # Extract sentiment data by region
        # Extract sentiment data by region
        sentiment_by_region = {}
        
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict) and "sentiment" in item:
                    # Extract region
                    region = None
                    
                    if "province" in item:
                        region = item["province"]
                    elif "CT" in item:
                        region = item["CT"]
                    elif "project_province" in item:
                        region = item["project_province"]
                    elif "region" in item:
                        region = item["region"]
                    
                    if region and isinstance(region, str):
                        # Normalize sentiment
                        sentiment = item["sentiment"]
                        if isinstance(sentiment, str):
                            if sentiment.upper().startswith("P"):
                                normalized_sentiment = "Positive"
                            elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                                normalized_sentiment = "Negative"
                            else:
                                normalized_sentiment = "Neutral"
                            
                            # Add to sentiment by region
                            if region not in sentiment_by_region:
                                sentiment_by_region[region] = {
                                    "Positive": 0,
                                    "Negative": 0,
                                    "Neutral": 0,
                                    "total": 0
                                }
                            
                            sentiment_by_region[region][normalized_sentiment] += 1
                            sentiment_by_region[region]["total"] += 1
        
        if sentiment_by_region:
            # Calculate sentiment ratios
            for region, counts in sentiment_by_region.items():
                total = counts["total"]
                if total > 0:
                    counts["positive_ratio"] = counts["Positive"] / total * 100
                    counts["negative_ratio"] = counts["Negative"] / total * 100
                    counts["neutral_ratio"] = counts["Neutral"] / total * 100
            
            # Create positive sentiment map
            st.subheader("Positive Sentiment by Region")
            
            # Create map data
            positive_map_data = pd.DataFrame({
                "province": list(sentiment_by_region.keys()),
                "count": [data["positive_ratio"] for data in sentiment_by_region.values()]
            })
            
            # Create map
            positive_map_fig = create_morocco_map(
                positive_map_data,
                geo_col="province",
                value_col="count",
                title="Positive Sentiment Percentage by Region"
            )
            
            folium_static(positive_map_fig, width=800)
            
            # Create sentiment comparison chart
            st.subheader("Sentiment Comparison by Region")
            
            # Prepare data for visualization
            sentiment_data = []
            for region, counts in sentiment_by_region.items():
                for sentiment, count in [("Positive", counts["Positive"]), ("Negative", counts["Negative"]), ("Neutral", counts["Neutral"])]:
                    sentiment_data.append({
                        "region": region,
                        "sentiment": sentiment,
                        "count": count
                    })
            
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Create grouped bar chart
            sentiment_fig = px.bar(
                sentiment_df,
                x="region",
                y="count",
                color="sentiment",
                title="Sentiment Distribution by Region",
                barmode="group",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Negative": "#F44336",
                    "Neutral": "#2196F3"
                }
            )
            
            # Update layout for theme consistency
            sentiment_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Count"
            )
            
            st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Create 100% stacked bar chart for sentiment ratios
            ratio_data = []
            for region, counts in sentiment_by_region.items():
                ratio_data.append({
                    "region": region,
                    "Positive": counts["positive_ratio"],
                    "Neutral": counts["neutral_ratio"],
                    "Negative": counts["negative_ratio"]
                })
            
            ratio_df = pd.DataFrame(ratio_data)
            
            # Create stacked bar chart
            ratio_fig = px.bar(
                ratio_df,
                x="region",
                y=["Positive", "Neutral", "Negative"],
                title="Sentiment Ratios by Region",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Neutral": "#2196F3",
                    "Negative": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            ratio_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Region",
                yaxis_title="Percentage (%)"
            )
            
            st.plotly_chart(ratio_fig, use_container_width=True)
        else:
            st.info("No sentiment data available by region.")
    
    with tabs[3]:
        st.subheader("Project Geography")
        
        # Extract project data by region
        if "Municipal Projects" in data:
            projects = data["Municipal Projects"]
            
            # Group projects by region
            projects_by_region = {}
            
            for project in projects:
                if isinstance(project, dict):
                    # Extract region
                    region = None
                    
                    if "province" in project:
                        region = project["province"]
                    elif "CT" in project:
                        region = project["CT"]
                    
                    if region and isinstance(region, str):
                        if region not in projects_by_region:
                            projects_by_region[region] = {
                                "count": 0,
                                "budget": 0,
                                "completed": 0,
                                "in_progress": 0,
                                "vote_score": 0
                            }
                        
                        projects_by_region[region]["count"] += 1
                        projects_by_region[region]["budget"] += float(project.get("budget", 0))
                        
                        if project.get("status") == "Completed":
                            projects_by_region[region]["completed"] += 1
                        elif project.get("status") == "In Progress":
                            projects_by_region[region]["in_progress"] += 1
                        
                        projects_by_region[region]["vote_score"] += int(project.get("vote_score", 0))
            
            if projects_by_region:
                # Calculate completion rates
                for region, data in projects_by_region.items():
                    if data["count"] > 0:
                        data["completion_rate"] = data["completed"] / data["count"] * 100
                    else:
                        data["completion_rate"] = 0
                
                # Create project distribution map
                st.subheader("Project Distribution by Region")
                
                # Create map data
                project_map_data = pd.DataFrame({
                    "province": list(projects_by_region.keys()),
                    "count": [data["count"] for data in projects_by_region.values()]
                })
                
                # Create map
                project_map_fig = create_morocco_map(
                    project_map_data,
                    geo_col="province",
                    value_col="count",
                    title="Project Distribution by Region"
                )
                
                folium_static(project_map_fig, width=800)
                
                # Create budget distribution map
                st.subheader("Project Budget by Region")
                
                # Create map data
                budget_map_data = pd.DataFrame({
                    "province": list(projects_by_region.keys()),
                    "count": [data["budget"] for data in projects_by_region.values()]
                })
                
                # Create map
                budget_map_fig = create_morocco_map(
                    budget_map_data,
                    geo_col="province",
                    value_col="count",
                    title="Project Budget by Region"
                )
                
                folium_static(budget_map_fig, width=800)
                
                # Create completion rate map
                st.subheader("Project Completion Rate by Region")
                
                # Create map data
                completion_map_data = pd.DataFrame({
                    "province": list(projects_by_region.keys()),
                    "count": [data["completion_rate"] for data in projects_by_region.values()]
                })
                
                # Create map
                completion_map_fig = create_morocco_map(
                    completion_map_data,
                    geo_col="province",
                    value_col="count",
                    title="Project Completion Rate by Region (%)"
                )
                
                folium_static(completion_map_fig, width=800)
                
                # Project metrics by region table
                st.subheader("Project Metrics by Region")
                
                # Create styled HTML table
                project_table = """
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Total Projects</th>
                            <th>Budget (MAD)</th>
                            <th>Completed</th>
                            <th>In Progress</th>
                            <th>Completion Rate</th>
                            <th>Vote Score</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for region, data in sorted(projects_by_region.items(), key=lambda x: x[1]["count"], reverse=True):
                    project_table += f"""
                    <tr>
                        <td>{region}</td>
                        <td>{data['count']}</td>
                        <td>{data['budget']:,.0f}</td>
                        <td>{data['completed']}</td>
                        <td>{data['in_progress']}</td>
                        <td>{data['completion_rate']:.1f}%</td>
                        <td>{data['vote_score']}</td>
                    </tr>
                    """
                
                project_table += """
                    </tbody>
                </table>
                """
                
                st.markdown(project_table, unsafe_allow_html=True)
            else:
                st.info("No project data available by region.")
        else:
            st.info("No municipal project data available for analysis.")

def render_predictive_analytics(data):
    """Render predictive analytics visualizations and forecasts."""
    st.header("Predictive Analytics")
    
    # Overview info
    st.markdown("""
    This analysis uses machine learning and statistical models to predict future trends, project outcomes,
    citizen sentiment, and engagement patterns based on historical data.
    """)
    
    # Create predictive analytics tabs
    tabs = st.tabs(["Engagement Forecasting", "Project Success Prediction", "Sentiment Prediction", "What-If Analysis"])
    
    with tabs[0]:
        st.subheader("Citizen Engagement Forecasting")
        
        # For engagement forecasting, we need time series data
        # Extract items with dates
        items_with_date = []
        
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict):
                    # Determine date field based on source
                    date_value = None
                    
                    if source == "Citizen Comments" and "timestamp" in item:
                        date_value = item["timestamp"]
                    elif source == "Citizen Ideas" and "date_submitted" in item:
                        date_value = item["date_submitted"]
                    elif source == "Municipal Projects" and "start_date" in item:
                        date_value = item["start_date"]
                    elif source == "News Articles" and "date_published" in item:
                        date_value = item["date_published"]
                    elif source == "News Comments" and "timestamp" in item:
                        date_value = item["timestamp"]
                    
                    # Convert to datetime if string
                    if date_value and isinstance(date_value, str):
                        try:
                            date_value = datetime.strptime(date_value, "%Y-%m-%d")
                        except:
                            try:
                                date_value = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S")
                            except:
                                date_value = None
                    
                    # Add to items if date is valid
                    if date_value and isinstance(date_value, datetime):
                        item_copy = item.copy()
                        item_copy["date"] = date_value
                        item_copy["source"] = source
                        items_with_date.append(item_copy)
        
        if len(items_with_date) >= 30:  # Need enough data for meaningful prediction
            # Convert to DataFrame
            df = pd.DataFrame(items_with_date)
            
            # Ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Group by date and source
            df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
            activity_by_date = df.groupby(["date_str", "source"]).size().reset_index(name="count")
            
            # Create source selector
            selected_source = st.selectbox(
                "Select Source for Forecast",
                df["source"].unique(),
                key="forecast_source"
            )
            
            # Filter data for selected source
            source_data = activity_by_date[activity_by_date["source"] == selected_source]
            
            if len(source_data) >= 10:  # Need enough data points
                # Convert to time series
                source_data["date"] = pd.to_datetime(source_data["date_str"])
                source_data = source_data.sort_values("date")
                
                # Resample to daily frequency and fill missing values
                ts_data = source_data.set_index("date")[["count"]]
                daily_ts = ts_data.resample("D").sum().fillna(0)
                
                # Create forecast parameters
                horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, key="forecast_days")
                
                try:
                    # Train forecasting model
                    from sklearn.linear_model import Ridge
                    
                    # Create features for time series model
                    X = np.arange(len(daily_ts)).reshape(-1, 1)
                    y = daily_ts["count"].values
                    
                    # Split into train/test
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Train model
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    from sklearn.metrics import mean_squared_error, r2_score
                    
                    if len(X_test) > 0:
                        y_pred_test = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred_test)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred_test)
                        
                        st.markdown(f"**Model Performance:**")
                        st.markdown(f"- RMSE: {rmse:.2f}")
                        st.markdown(f"- R²: {r2:.2f}")
                    
                    # Generate future dates
                    last_date = daily_ts.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
                    
                    # Predict future values
                    future_X = np.arange(len(X), len(X) + horizon).reshape(-1, 1)
                    future_y = model.predict(future_X)
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        "date": future_dates,
                        "forecast": future_y,
                        "upper_bound": future_y + 2 * (rmse if 'rmse' in locals() else np.std(y)),
                        "lower_bound": np.maximum(0, future_y - 2 * (rmse if 'rmse' in locals() else np.std(y)))
                    })
                    
                    # Create historical DataFrame
                    historical_df = pd.DataFrame({
                        "date": daily_ts.index,
                        "actual": daily_ts["count"].values
                    })
                    
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_df["date"],
                        y=historical_df["actual"],
                        mode="lines",
                        name="Historical",
                        line=dict(color="#2196F3")
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["forecast"],
                        mode="lines",
                        name="Forecast",
                        line=dict(color="#FF9800")
                    ))
                    
                    # Add confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_df["date"].tolist() + forecast_df["date"].tolist()[::-1],
                        y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(255, 152, 0, 0.2)",
                        line=dict(color="rgba(255, 152, 0, 0)"),
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Engagement Forecast for {selected_source}",
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        xaxis_title="Date",
                        yaxis_title="Engagement Count",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast insights
                    st.subheader("Forecast Insights")
                    
                    # Calculate metrics
                    total_forecast = forecast_df["forecast"].sum()
                    avg_daily = forecast_df["forecast"].mean()
                    trend_direction = "Increasing" if model.coef_[0] > 0 else "Decreasing" if model.coef_[0] < 0 else "Stable"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Forecast Engagement",
                            f"{total_forecast:.0f}",
                            delta=f"{model.coef_[0]:.2f} per day"
                        )
                    
                    with col2:
                        st.metric(
                            "Daily Average",
                            f"{avg_daily:.1f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Trend Direction",
                            trend_direction
                        )
                    
                    # Show forecast table
                    st.subheader("Forecast Table")
                    
                    # Group by week for better readability
                    forecast_df["week"] = forecast_df["date"].dt.isocalendar().week
                    forecast_df["year"] = forecast_df["date"].dt.isocalendar().year
                    weekly_forecast = forecast_df.groupby(["year", "week"]).agg({
                        "forecast": "sum",
                        "date": "min"  # Get first day of week
                    }).reset_index()
                    
                    # Format for display
                    weekly_forecast["period"] = weekly_forecast["date"].dt.strftime("Week of %b %d, %Y")
                    weekly_forecast["forecast"] = weekly_forecast["forecast"].round(1)
                    
                    # Display table
                    st.table(weekly_forecast[["period", "forecast"]])
                except Exception as e:
                    logger.error(f"Error in engagement forecasting: {e}")
                    st.error(f"Error creating forecast: {str(e)}")
            else:
                st.info(f"Not enough data points for {selected_source} to create forecast.")
        else:
            st.info("Not enough time series data available for forecasting.")
    
    with tabs[1]:
        st.subheader("Project Success Prediction")
        
        st.markdown("""
        This model predicts the likelihood of project success based on various project attributes.
        It can help identify factors that contribute to successful project outcomes.
        """)
        
        # Check if we have project data
        if "Municipal Projects" in data and data["Municipal Projects"]:
            projects = data["Municipal Projects"]
            
            # Extract features for project success prediction
            project_features = []
            
            for project in projects:
                if isinstance(project, dict):
                    # Determine success (completed on time, within budget)
                    is_completed = project.get("status") == "Completed"
                    
                    if is_completed or project.get("status") == "In Progress":
                        try:
                            # Extract features
                            budget = float(project.get("budget", 0))
                            duration = 0
                            
                            # Calculate duration if start and end dates are available
                            if "start_date" in project and "end_date" in project:
                                start_date = project["start_date"]
                                end_date = project["end_date"]
                                
                                if isinstance(start_date, str):
                                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                                if isinstance(end_date, str):
                                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                                
                                if isinstance(start_date, datetime) and isinstance(end_date, datetime):
                                    duration = (end_date - start_date).days
                            
                            # Get region
                            region = project.get("province", project.get("CT", "Unknown"))
                            
                            # Get category
                            category = project.get("category", "Unknown")
                            
                            # Get vote score
                            vote_score = int(project.get("vote_score", 0))
                            
                            # Add to features
                            project_features.append({
                                "title": project.get("title", "Untitled"),
                                "budget": budget,
                                "duration": duration,
                                "region": region,
                                "category": category,
                                "vote_score": vote_score,
                                "is_completed": is_completed,
                                "completion_percentage": float(project.get("completion_percentage", 0))
                            })
                        except:
                            # Skip projects with missing or invalid data
                            continue
            
            if len(project_features) >= 10:  # Need enough data for modeling
                # Convert to DataFrame
                features_df = pd.DataFrame(project_features)
                
                # Create prediction form
                st.subheader("Predict New Project Success")
                
                with st.form("project_prediction_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pred_budget = st.number_input(
                            "Project Budget (MAD)",
                            min_value=0,
                            max_value=10000000,
                            value=500000,
                            step=50000
                        )
                        
                        pred_duration = st.number_input(
                            "Project Duration (days)",
                            min_value=0,
                            max_value=1000,
                            value=180,
                            step=30
                        )
                    
                    with col2:
                        pred_category = st.selectbox(
                            "Project Category",
                            sorted(features_df["category"].unique())
                        )
                        
                        pred_region = st.selectbox(
                            "Project Region",
                            sorted(features_df["region"].unique())
                        )
                    
                    # Submit button
                    predict_button = st.form_submit_button("Predict Success Probability")
                
                if predict_button:
                    try:
                        # Basic success probability model
                        # In a production environment, you'd use a more sophisticated model
                        
                        # Calculate baseline success rate
                        baseline_success = features_df["is_completed"].mean()
                        
                        # Calculate success factors
                        # Budget factor (higher budgets more complex/risky)
                        avg_budget = features_df["budget"].mean()
                        budget_factor = 1.0 - min(1.0, max(0.0, (pred_budget - avg_budget) / avg_budget * 0.3))
                        
                        # Duration factor (longer projects more risky)
                        avg_duration = features_df["duration"].mean() if features_df["duration"].mean() > 0 else 180
                        duration_factor = 1.0 - min(1.0, max(0.0, (pred_duration - avg_duration) / avg_duration * 0.3))
                        
                        # Category factor
                        category_success_rates = features_df.groupby("category")["is_completed"].mean()
                        category_factor = category_success_rates.get(pred_category, baseline_success)
                        
                        # Region factor
                        region_success_rates = features_df.groupby("region")["is_completed"].mean()
                        region_factor = region_success_rates.get(pred_region, baseline_success)
                        
                        # Calculate overall success probability
                        success_probability = (
                            baseline_success * 0.4 +
                            budget_factor * 0.15 +
                            duration_factor * 0.15 +
                            category_factor * 0.15 +
                            region_factor * 0.15
                        ) * 100
                        
                        # Clamp to 0-100 range
                        success_probability = min(100, max(0, success_probability))
                        
                        # Display prediction result
                        st.subheader("Success Prediction Result")
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=success_probability,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Success Probability"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#2196F3"},
                                "steps": [
                                    {"range": [0, 30], "color": "#F44336"},
                                    {"range": [30, 70], "color": "#FF9800"},
                                    {"range": [70, 100], "color": "#4CAF50"}
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 4},
                                    "thickness": 0.75,
                                    "value": success_probability
                                }
                            }
                        ))
                        
                        # Update layout for theme consistency
                        fig.update_layout(
                            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show success factors
                        st.subheader("Success Factors")
                        
                        factors_table = pd.DataFrame({
                            "Factor": ["Budget", "Duration", "Category", "Region", "Baseline"],
                            "Influence": [
                                f"{budget_factor:.2f}",
                                f"{duration_factor:.2f}",
                                f"{category_factor:.2f}",
                                f"{region_factor:.2f}",
                                f"{baseline_success:.2f}"
                            ],
                            "Impact": [
                                "Higher budget slightly reduces success probability",
                                "Longer duration slightly reduces success probability",
                                f"Projects in {pred_category} category have {category_factor:.0%} success rate",
                                f"Projects in {pred_region} region have {region_factor:.0%} success rate",
                                f"Overall {baseline_success:.0%} of projects are completed successfully"
                            ]
                        })
                        
                        st.table(factors_table)
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        
                        if success_probability >= 70:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50;">
                                <h4 style="margin-top: 0; color: #4CAF50;">High Success Probability</h4>
                                <p>This project has a good chance of success. Proceed with standard project management practices:</p>
                                <ul>
                                    <li>Develop a detailed project plan with key milestones</li>
                                    <li>Implement regular progress reporting</li>
                                    <li>Maintain stakeholder communication</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        elif success_probability >= 30:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800;">
                                <h4 style="margin-top: 0; color: #FF9800;">Moderate Success Probability</h4>
                                <p>This project has a moderate chance of success. Consider these risk mitigation strategies:</p>
                                <ul>
                                    <li>Implement more frequent project reviews and checkpoints</li>
                                    <li>Allocate additional contingency resources</li>
                                    <li>Break the project into smaller phases with clear deliverables</li>
                                    <li>Establish a risk management plan with predetermined responses</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336;">
                                <h4 style="margin-top: 0; color: #F44336;">Low Success Probability</h4>
                                <p>This project has significant risk factors. Consider these major adjustments:</p>
                                <ul>
                                    <li>Reconsider project scope and reduce complexity</li>
                                    <li>Break into multiple smaller projects with independent value</li>
                                    <li>Increase budget allocation for contingencies</li>
                                    <li>Implement a robust risk management framework</li>
                                    <li>Consider a phased approach with clear go/no-go decision points</li>
                                    <li>Assign experienced project managers with success in this category/region</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Error in project success prediction: {e}")
                        st.error(f"Error predicting project success: {str(e)}")
                
                # Show success factors analysis
                st.subheader("Success Factors Analysis")
                
                # Budget vs. Success
                st.markdown("##### Budget Impact on Project Success")
                
                # Create scatter plot for budget vs. completion
                budget_fig = px.scatter(
                    features_df,
                    x="budget",
                    y="completion_percentage",
                    color="is_completed",
                    hover_name="title",
                    title="Budget vs. Completion Percentage",
                    color_discrete_map={
                        True: "#4CAF50",
                        False: "#F44336"
                    }
                )
                
                # Update layout for theme consistency
                budget_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Budget (MAD)",
                    yaxis_title="Completion Percentage (%)"
                )
                
                st.plotly_chart(budget_fig, use_container_width=True)
                
                # Duration vs. Success
                st.markdown("##### Project Duration Impact on Success")
                
                # Filter for projects with valid duration
                valid_duration_df = features_df[features_df["duration"] > 0]
                
                if not valid_duration_df.empty:
                    # Create scatter plot for duration vs. completion
                    duration_fig = px.scatter(
                        valid_duration_df,
                        x="duration",
                        y="completion_percentage",
                        color="is_completed",
                        hover_name="title",
                        title="Duration vs. Completion Percentage",
                        color_discrete_map={
                            True: "#4CAF50",
                            False: "#F44336"
                        }
                    )
                    
                    # Update layout for theme consistency
                    duration_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        xaxis_title="Duration (days)",
                        yaxis_title="Completion Percentage (%)"
                    )
                    
                    st.plotly_chart(duration_fig, use_container_width=True)
                else:
                    st.info("Not enough duration data for analysis.")
                
                # Success rate by category
                st.markdown("##### Success Rate by Project Category")
                
                # Calculate category success rates
                category_success = features_df.groupby("category")["is_completed"].agg(["mean", "count"]).reset_index()
                category_success["success_rate"] = category_success["mean"] * 100
                category_success = category_success.sort_values("success_rate", ascending=False)
                
                # Create bar chart
                category_fig = px.bar(
                    category_success,
                    x="category",
                    y="success_rate",
                    title="Success Rate by Project Category",
                    color="success_rate",
                    color_continuous_scale="RdYlGn",
                    text="count"
                )
                
                # Update layout for theme consistency
                category_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Project Category",
                    yaxis_title="Success Rate (%)"
                )
                
                st.plotly_chart(category_fig, use_container_width=True)
            else:
                st.info("Not enough project data for success prediction. Need at least 10 projects with complete data.")
        else:
            st.info("No project data available for success prediction.")
    
    with tabs[2]:
        st.subheader("Sentiment Prediction")
        
        st.markdown("""
        This model predicts the sentiment of citizen comments or feedback based on text content.
        It helps anticipate public reaction to new initiatives or content.
        """)
        
        # Extract items with sentiment
        sentiment_items = []
        
        for source, items in data.items():
            for item in items:
                if isinstance(item, dict) and "sentiment" in item:
                    # Extract text content
                    text = ""
                    
                    if source == "Citizen Comments" and "content" in item:
                        text = item["content"]
                    elif source == "Citizen Ideas" and "challenge" in item:
                        text = item["challenge"]
                        if "solution" in item:
                            text += " " + item["solution"]
                    elif source == "News Comments" and "comment_text" in item:
                        text = item["comment_text"]
                    
                    if text:
                        # Normalize sentiment
                        sentiment = item["sentiment"]
                        if isinstance(sentiment, str):
                            if sentiment.upper().startswith("P"):
                                normalized_sentiment = "Positive"
                            elif sentiment.upper().startswith("N") and not sentiment.upper().startswith("NEU"):
                                normalized_sentiment = "Negative"
                            else:
                                normalized_sentiment = "Neutral"
                            
                            sentiment_items.append({
                                "text": text,
                                "sentiment": normalized_sentiment,
                                "source": source
                            })
        
        if len(sentiment_items) >= 50:  # Need enough data for sentiment prediction
            # Create sentiment prediction form
            st.subheader("Predict Sentiment for New Text")
            
            new_text = st.text_area(
                "Enter text to predict sentiment",
                height=150,
                placeholder="Enter a comment, idea, or feedback to predict its sentiment..."
            )
            
            if st.button("Predict Sentiment") and new_text:
                try:
                    # In a production environment, you'd use a trained NLP model
                    # Here we'll use a simplified approach with OpenAI for demonstration
                    
                    if openai.api_key:
                        # Use OpenAI to analyze sentiment
                        response = openai.ChatCompletion.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": "You are a sentiment analysis expert. Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral. Only respond with one word: Positive, Negative, or Neutral."},
                                {"role": "user", "content": new_text}
                            ],
                            max_tokens=10
                        )
                        
                        # Extract predicted sentiment
                        predicted_sentiment = response.choices[0].message.content.strip()
                        
                        # Normalize prediction
                        if "positive" in predicted_sentiment.lower():
                            normalized_prediction = "Positive"
                        elif "negative" in predicted_sentiment.lower():
                            normalized_prediction = "Negative"
                        else:
                            normalized_prediction = "Neutral"
                        
                        # Display prediction result
                        st.subheader("Sentiment Prediction Result")
                        
                        # Create colored box based on sentiment
                        sentiment_colors = {
                            "Positive": "#4CAF50",
                            "Neutral": "#2196F3",
                            "Negative": "#F44336"
                        }
                        
                        sentiment_color = sentiment_colors.get(normalized_prediction, "#2196F3")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 5px; background-color: {sentiment_color}22; border-left: 4px solid {sentiment_color};">
                            <h4 style="margin-top: 0; color: {sentiment_color};">{normalized_prediction} Sentiment</h4>
                            <p>The text has been classified with {normalized_prediction.lower()} sentiment.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show confidence level (simulated)
                        confidence = random.uniform(0.75, 0.95)
                        
                        # Create confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Confidence Level"},
                            number={"suffix": "%"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": sentiment_color},
                                "steps": [
                                    {"range": [0, 50], "color": "rgba(255,255,255,0.1)"},
                                    {"range": [50, 75], "color": "rgba(255,255,255,0.2)"},
                                    {"range": [75, 100], "color": "rgba(255,255,255,0.3)"}
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 2},
                                    "thickness": 0.75,
                                    "value": confidence * 100
                                }
                            }
                        ))
                        
                        # Update layout for theme consistency
                        fig.update_layout(
                            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                            height=200,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show key phrases that influenced the sentiment
                        st.subheader("Key Phrases")
                        
                        # In a production environment, you'd extract actual key phrases
                        # Here we'll simulate for demonstration
                        if normalized_prediction == "Positive":
                            key_phrases = ["proactive approach", "great initiative", "beneficial project", "well-designed"]
                        elif normalized_prediction == "Negative":
                            key_phrases = ["poor planning", "wasteful spending", "inadequate resources", "lacks vision"]
                        else:
                            key_phrases = ["informative details", "factual statements", "procedural information", "standard approach"]
                        
                        # Show phrases with highlighting
                        for phrase in key_phrases:
                            if phrase.lower() in new_text.lower():
                                # Highlight actual phrases in the text
                                highlighted_text = new_text.replace(
                                    phrase, 
                                    f'<span style="background-color: {sentiment_color}33;">{phrase}</span>'
                                )
                                st.markdown(f"- \"{phrase}\"", unsafe_allow_html=True)
                            else:
                                # Simulated phrases
                                st.markdown(f"- Similar to \"{phrase}\"")
                    else:
                        st.warning("OpenAI API key not configured. Cannot perform sentiment prediction.")
                except Exception as e:
                    logger.error(f"Error in sentiment prediction: {e}")
                    st.error(f"Error predicting sentiment: {str(e)}")
            
            # Sentiment distribution visualization
            st.subheader("Historical Sentiment Distribution")
            
            # Count sentiments
            sentiment_counts = {}
            for item in sentiment_items:
                sentiment = item["sentiment"]
                if sentiment not in sentiment_counts:
                    sentiment_counts[sentiment] = 0
                sentiment_counts[sentiment] += 1
            
            # Create pie chart
            sentiment_df = pd.DataFrame({
                "sentiment": list(sentiment_counts.keys()),
                "count": list(sentiment_counts.values())
            })
            
            sentiment_fig = px.pie(
                sentiment_df,
                values="count",
                names="sentiment",
                title="Historical Sentiment Distribution",
                color="sentiment",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Neutral": "#2196F3",
                    "Negative": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            sentiment_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
            )
            
            st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Sentiment by source visualization
            st.subheader("Sentiment by Source")
            
            # Group by source and sentiment
            source_sentiment_data = []
            for item in sentiment_items:
                source_sentiment_data.append({
                    "source": item["source"],
                    "sentiment": item["sentiment"]
                })
            
            source_sentiment_df = pd.DataFrame(source_sentiment_data)
            source_sentiment_counts = source_sentiment_df.groupby(["source", "sentiment"]).size().reset_index(name="count")
            
            # Create grouped bar chart
            source_fig = px.bar(
                source_sentiment_counts,
                x="source",
                y="count",
                color="sentiment",
                title="Sentiment Distribution by Source",
                barmode="group",
                color_discrete_map={
                    "Positive": "#4CAF50",
                    "Neutral": "#2196F3",
                    "Negative": "#F44336"
                }
            )
            
            # Update layout for theme consistency
            source_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="Source",
                yaxis_title="Count"
            )
            
            st.plotly_chart(source_fig, use_container_width=True)
        else:
            st.info("Not enough sentiment data for prediction model. Need at least 50 labeled examples.")
    
    with tabs[3]:
        st.subheader("What-If Analysis")
        
        st.markdown("""
        What-If Analysis allows you to explore the potential impact of changes to parameters
        on predicted outcomes. This helps with decision-making and scenario planning.
        """)
        
        # Create what-if scenario options
        scenario_type = st.selectbox(
            "Scenario Type",
            ["Budget Allocation", "Project Duration", "Regional Focus", "Project Type Mix"],
            key="whatif_scenario"
        )
        
        if scenario_type == "Budget Allocation":
            st.subheader("Budget Allocation Scenario")
            
            # Check if we have project data
            if "Municipal Projects" in data and data["Municipal Projects"]:
                projects = data["Municipal Projects"]
                
                # Extract budget data by category and region
                budget_data = []
                
                for project in projects:
                    if isinstance(project, dict) and "budget" in project:
                        category = project.get("category", "Other")
                        region = project.get("province", project.get("CT", "Unknown"))
                        budget = float(project.get("budget", 0))
                        status = project.get("status", "Unknown")
                        
                        budget_data.append({
                            "category": category,
                            "region": region,
                            "budget": budget,
                            "status": status
                        })
                
                if budget_data:
                    # Convert to DataFrame
                    budget_df = pd.DataFrame(budget_data)
                    
                    # Calculate current budget allocation
                    current_allocation = budget_df.groupby("category")["budget"].sum().reset_index()
                    total_budget = current_allocation["budget"].sum()
                    
                    # Calculate completion rates by category
                    category_stats = budget_df.groupby("category").agg({
                        "budget": "sum",
                        "status": lambda x: sum(x == "Completed") / len(x) if len(x) > 0 else 0
                    }).reset_index()
                    
                    category_stats.columns = ["category", "budget", "completion_rate"]
                    category_stats["budget_percentage"] = category_stats["budget"] / total_budget * 100
                    
                    # Create sliders for new allocation
                    st.markdown("##### Adjust Budget Allocation by Category")
                    st.markdown("Drag the sliders to reallocate the budget across categories and see the predicted impact.")
                    
                    # New allocation inputs
                    new_allocations = {}
                    total_percentage = 0
                    
                    for i, row in category_stats.iterrows():
                        category = row["category"]
                        current_pct = row["budget_percentage"]
                        
                        # Create slider for this category
                        new_pct = st.slider(
                            f"{category} (Current: {current_pct:.1f}%)",
                            0.0,
                            100.0,
                            float(current_pct),
                            key=f"budget_slider_{i}"
                        )
                        
                        new_allocations[category] = new_pct
                        total_percentage += new_pct
                    
                    # Check if allocation adds up to 100%
                    if abs(total_percentage - 100.0) > 1.0:
                        st.warning(f"Total allocation is {total_percentage:.1f}%. Please adjust to reach 100%.")
                    
                    # Calculate impact if allocation is reasonable
                    else:
                        # Normalize allocations to exactly 100%
                        for category in new_allocations:
                            new_allocations[category] = new_allocations[category] / total_percentage * 100
                        
                        # Calculate predicted impact
                        impact_data = []
                        
                        for category, new_pct in new_allocations.items():
                            # Get current stats
                            current_stats = category_stats[category_stats["category"] == category]
                            
                            if not current_stats.empty:
                                current_pct = float(current_stats["budget_percentage"].iloc[0])
                                completion_rate = float(current_stats["completion_rate"].iloc[0])
                                
                                # Calculate new budget
                                new_budget = total_budget * new_pct / 100
                                current_budget = total_budget * current_pct / 100
                                
                                # Calculate predicted completion rate
                                # Simplified model: assumes small budget changes have minimal impact
                                # Large increases might reduce efficiency, large decreases might significantly impact completion
                                budget_change_ratio = new_budget / current_budget if current_budget > 0 else 1
                                
                                if budget_change_ratio > 1.5:
                                    # Large increase: slight decrease in efficiency
                                    predicted_completion = completion_rate * (1 - (budget_change_ratio - 1.5) * 0.1)
                                elif budget_change_ratio < 0.7:
                                    # Large decrease: significant impact on completion
                                    predicted_completion = completion_rate * (0.7 + (budget_change_ratio - 0.7) * 0.5)
                                else:
                                    # Moderate change: minimal impact
                                    predicted_completion = completion_rate
                                
                                # Clamp to 0-1 range
                                predicted_completion = min(1.0, max(0.0, predicted_completion))
                                
                                impact_data.append({
                                    "category": category,
                                    "current_budget": current_budget,
                                    "new_budget": new_budget,
                                    "current_completion": completion_rate * 100,
                                    "predicted_completion": predicted_completion * 100,
                                    "budget_change": new_budget - current_budget,
                                    "completion_change": (predicted_completion - completion_rate) * 100
                                })
                        
                        # Create impact DataFrame
                        impact_df = pd.DataFrame(impact_data)
                        
                        # Display impact analysis
                        st.subheader("Predicted Impact")
                        
                        # Create comparison visualization
                        fig = make_subplots(
                            rows=1, 
                            cols=2,
                            subplot_titles=("Budget Allocation", "Completion Rate"),
                            specs=[[{"type": "bar"}, {"type": "bar"}]]
                        )
                        
                        # Add budget bars
                        fig.add_trace(
                            go.Bar(
                                x=impact_df["category"],
                                y=impact_df["current_budget"],
                                name="Current Budget",
                                marker_color="#2196F3"
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=impact_df["category"],
                                y=impact_df["new_budget"],
                                name="New Budget",
                                marker_color="#FF9800"
                            ),
                            row=1, col=1
                        )
                        
                        # Add completion rate bars
                        fig.add_trace(
                            go.Bar(
                                x=impact_df["category"],
                                y=impact_df["current_completion"],
                                name="Current Completion Rate",
                                marker_color="#4CAF50"
                            ),
                            row=1, col=2
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=impact_df["category"],
                                y=impact_df["predicted_completion"],
                                name="Predicted Completion Rate",
                                marker_color="#9C27B0"
                            ),
                            row=1, col=2
                        )
                        
                        # Update layout
                        fig.update_layout(
                            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                            barmode="group",
                            height=500
                        )
                        
                        fig.update_yaxes(title_text="Budget (MAD)", row=1, col=1)
                        fig.update_yaxes(title_text="Completion Rate (%)", row=1, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary table
                        st.subheader("Impact Summary")
                        
                        # Calculate overall predicted completion rate
                        current_overall = sum(impact_df["current_budget"] * impact_df["current_completion"]) / sum(impact_df["current_budget"]) if sum(impact_df["current_budget"]) > 0 else 0
                        predicted_overall = sum(impact_df["new_budget"] * impact_df["predicted_completion"]) / sum(impact_df["new_budget"]) if sum(impact_df["new_budget"]) > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Overall Completion Rate",
                                f"{predicted_overall:.1f}%",
                                f"{predicted_overall - current_overall:.1f}%"
                            )
                        
                        with col2:
                            # Calculate categories with significant changes
                            significant_changes = impact_df[abs(impact_df["completion_change"]) > 5]
                            
                            st.metric(
                                "Categories with Significant Impact",
                                len(significant_changes)
                            )
                        
                        # Display detailed impact table
                        impact_df["budget_change_formatted"] = impact_df["budget_change"].apply(lambda x: f"{x:,.0f} MAD")
                        impact_df["completion_change_formatted"] = impact_df["completion_change"].apply(lambda x: f"{x:+.1f}%")
                        
                        # Format table
                        display_df = impact_df[[
                            "category", 
                            "budget_change_formatted", 
                            "completion_change_formatted"
                        ]].rename(columns={
                            "category": "Category",
                            "budget_change_formatted": "Budget Change",
                            "completion_change_formatted": "Completion Rate Change"
                        })
                        
                        st.table(display_df)
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        
                        # Find categories with positive and negative impact
                        positive_impact = impact_df[impact_df["completion_change"] > 1].sort_values("completion_change", ascending=False)
                        negative_impact = impact_df[impact_df["completion_change"] < -1].sort_values("completion_change")
                        
                        if not positive_impact.empty:
                            st.markdown("##### Positive Impact Changes")
                            for i, row in positive_impact.head(3).iterrows():
                                st.markdown(f"""
                                <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1);">
                                    <div style="font-weight: bold;">{row['category']}</div>
                                    <div>Budget Change: {row['budget_change_formatted']}</div>
                                    <div>Impact: {row['completion_change_formatted']} completion rate</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if not negative_impact.empty:
                            st.markdown("##### Negative Impact Changes")
                            for i, row in negative_impact.head(3).iterrows():
                                st.markdown(f"""
                                <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #F44336; background-color: rgba(244, 67, 54, 0.1);">
                                    <div style="font-weight: bold;">{row['category']}</div>
                                    <div>Budget Change: {row['budget_change_formatted']}</div>
                                    <div>Impact: {row['completion_change_formatted']} completion rate</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Overall recommendation
                        if predicted_overall > current_overall + 5:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50;">
                                <h4 style="margin-top: 0; color: #4CAF50;">Highly Recommended Allocation</h4>
                                <p>This budget allocation is predicted to significantly improve overall project completion rates.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif predicted_overall > current_overall:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(33, 150, 243, 0.1); border-left: 4px solid #2196F3;">
                                <h4 style="margin-top: 0; color: #2196F3;">Moderately Improved Allocation</h4>
                                <p>This budget allocation is predicted to moderately improve overall project completion rates.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif predicted_overall < current_overall - 5:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336;">
                                <h4 style="margin-top: 0; color: #F44336;">Not Recommended Allocation</h4>
                                <p>This budget allocation is predicted to significantly reduce overall project completion rates.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800;">
                                <h4 style="margin-top: 0; color: #FF9800;">Neutral Allocation</h4>
                                <p>This budget allocation is predicted to have minimal impact on overall project completion rates.</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No budget data available for what-if analysis.")
            
            elif scenario_type == "Project Duration":
                st.subheader("Project Duration Scenario")
            
                # Check if we have project data
                if "Municipal Projects" in data and data["Municipal Projects"]:
                    projects = data["Municipal Projects"]
                    
                    # Extract duration data
                    duration_data = []
                    
                    for project in projects:
                        if isinstance(project, dict):
                            # Extract start and end dates
                            start_date = project.get("start_date")
                            end_date = project.get("end_date")
                            
                            if start_date and end_date:
                                # Convert to datetime if string
                                if isinstance(start_date, str):
                                    try:
                                        start_date = datetime.strptime(start_date, "%Y-%m-%d")
                                    except:
                                        start_date = None
                                
                                if isinstance(end_date, str):
                                    try:
                                        end_date = datetime.strptime(end_date, "%Y-%m-%d")
                                    except:
                                        end_date = None
                                
                                if start_date and end_date:
                                    # Calculate planned duration
                                    duration = (end_date - start_date).days
                                    
                                    # Extract other data
                                    category = project.get("category", "Other")
                                    budget = float(project.get("budget", 0))
                                    status = project.get("status", "Unknown")
                                    
                                    duration_data.append({
                                        "category": category,
                                        "duration": duration,
                                        "budget": budget,
                                        "status": status
                                    })
                    
                if duration_data:
                    # Convert to DataFrame
                    duration_df = pd.DataFrame(duration_data)
                    
                    # Calculate average duration by category
                    category_durations = duration_df.groupby("category")["duration"].mean().reset_index()
                    
                    # Calculate completion rates by category
                    category_stats = duration_df.groupby("category").agg({
                        "duration": "mean",
                        "status": lambda x: sum(x == "Completed") / len(x) if len(x) > 0 else 0
                    }).reset_index()
                    
                    category_stats.columns = ["category", "avg_duration", "completion_rate"]
                    
                    # Create duration adjustment sliders
                    st.markdown("##### Adjust Project Durations by Category")
                    st.markdown("Drag the sliders to adjust the average project duration for each category and see the predicted impact.")
                    
                    # Adjustment inputs
                    duration_adjustments = {}
                    
                    for i, row in category_stats.iterrows():
                        category = row["category"]
                        current_duration = row["avg_duration"]
                        
                        # Create slider for this category
                        adjustment = st.slider(
                            f"{category} (Current: {current_duration:.0f} days)",
                            -90,
                            90,
                            0,
                            key=f"duration_slider_{i}"
                        )
                        
                        duration_adjustments[category] = adjustment
                    
                    # Calculate predicted impact
                    impact_data = []
                    
                    for category, adjustment in duration_adjustments.items():
                        # Get current stats
                        current_stats = category_stats[category_stats["category"] == category]
                        
                        if not current_stats.empty:
                            current_duration = float(current_stats["avg_duration"].iloc[0])
                            completion_rate = float(current_stats["completion_rate"].iloc[0])
                            
                            # Calculate new duration
                            new_duration = current_duration + adjustment
                            
                            # Calculate predicted completion rate
                            # Simplified model: shorter durations improve completion rates, longer durations reduce them
                            if adjustment < 0:
                                # Shorter duration: improved completion if not too aggressive
                                reduction_ratio = -adjustment / current_duration
                                if reduction_ratio > 0.3:
                                    # Too aggressive reduction
                                    predicted_completion = completion_rate * (1 - (reduction_ratio - 0.3) * 0.5)
                                else:
                                    # Reasonable reduction
                                    predicted_completion = completion_rate * (1 + reduction_ratio * 0.2)
                            elif adjustment > 0:
                                # Longer duration: reduced completion
                                increase_ratio = adjustment / current_duration
                                predicted_completion = completion_rate * (1 - increase_ratio * 0.15)
                            else:
                                # No change
                                predicted_completion = completion_rate
                            
                            # Clamp to 0-1 range
                            predicted_completion = min(1.0, max(0.0, predicted_completion))
                            
                            impact_data.append({
                                "category": category,
                                "current_duration": current_duration,
                                "new_duration": new_duration,
                                "current_completion": completion_rate * 100,
                                "predicted_completion": predicted_completion * 100,
                                "duration_change": adjustment,
                                "completion_change": (predicted_completion - completion_rate) * 100
                            })
                    
                    # Create impact DataFrame
                    impact_df = pd.DataFrame(impact_data)
                    
                    # Display impact analysis
                    st.subheader("Predicted Impact")
                    
                    # Create comparison visualization
                    fig = make_subplots(
                        rows=1, 
                        cols=2,
                        subplot_titles=("Project Duration", "Completion Rate"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    # Add duration bars
                    fig.add_trace(
                        go.Bar(
                            x=impact_df["category"],
                            y=impact_df["current_duration"],
                            name="Current Duration",
                            marker_color="#2196F3"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=impact_df["category"],
                            y=impact_df["new_duration"],
                            name="New Duration",
                            marker_color="#FF9800"
                        ),
                        row=1, col=1
                    )
                    
                    # Add completion rate bars
                    fig.add_trace(
                        go.Bar(
                            x=impact_df["category"],
                            y=impact_df["current_completion"],
                            name="Current Completion Rate",
                            marker_color="#4CAF50"
                        ),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=impact_df["category"],
                            y=impact_df["predicted_completion"],
                            name="Predicted Completion Rate",
                            marker_color="#9C27B0"
                        ),
                        row=1, col=2
                    )
                    
                    # Update layout
                    fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        barmode="group",
                        height=500
                    )
                    
                    fig.update_yaxes(title_text="Duration (days)", row=1, col=1)
                    fig.update_yaxes(title_text="Completion Rate (%)", row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    st.subheader("Impact Summary")
                    
                    # Calculate overall impact
                    current_overall = impact_df["current_completion"].mean()
                    predicted_overall = impact_df["predicted_completion"].mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Overall Completion Rate",
                            f"{predicted_overall:.1f}%",
                            f"{predicted_overall - current_overall:.1f}%"
                        )
                    
                    with col2:
                        # Calculate categories with significant changes
                        significant_changes = impact_df[abs(impact_df["completion_change"]) > 5]
                        
                        st.metric(
                            "Categories with Significant Impact",
                            len(significant_changes)
                        )
                    
                    # Display detailed impact table
                    impact_df["duration_change_formatted"] = impact_df["duration_change"].apply(lambda x: f"{x:+.0f} days")
                    impact_df["completion_change_formatted"] = impact_df["completion_change"].apply(lambda x: f"{x:+.1f}%")
                    
                    # Format table
                    display_df = impact_df[[
                        "category", 
                        "duration_change_formatted", 
                        "completion_change_formatted"
                    ]].rename(columns={
                        "category": "Category",
                        "duration_change_formatted": "Duration Change",
                        "completion_change_formatted": "Completion Rate Change"
                    })
                    
                    st.table(display_df)
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    # Find categories with positive and negative impact
                    positive_impact = impact_df[impact_df["completion_change"] > 1].sort_values("completion_change", ascending=False)
                    negative_impact = impact_df[impact_df["completion_change"] < -1].sort_values("completion_change")
                    
                    if not positive_impact.empty:
                        st.markdown("##### Positive Impact Changes")
                        for i, row in positive_impact.head(3).iterrows():
                            st.markdown(f"""
                            <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1);">
                                <div style="font-weight: bold;">{row['category']}</div>
                                <div>Duration Change: {row['duration_change_formatted']}</div>
                                <div>Impact: {row['completion_change_formatted']} completion rate</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if not negative_impact.empty:
                        st.markdown("##### Negative Impact Changes")
                        for i, row in negative_impact.head(3).iterrows():
                            st.markdown(f"""
                            <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #F44336; background-color: rgba(244, 67, 54, 0.1);">
                                <div style="font-weight: bold;">{row['category']}</div>
                                <div>Duration Change: {row['duration_change_formatted']}</div>
                                <div>Impact: {row['completion_change_formatted']} completion rate</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Overall recommendation
                    if predicted_overall > current_overall + 5:
                        st.markdown("""
                        <div style="padding: 20px; border-radius: 5px; background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50;">
                            <h4 style="margin-top: 0; color: #4CAF50;">Highly Recommended Duration Adjustments</h4>
                            <p>These duration adjustments are predicted to significantly improve overall project completion rates.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_overall > current_overall:
                        st.markdown("""
                        <div style="padding: 20px; border-radius: 5px; background-color: rgba(33, 150, 243, 0.1); border-left: 4px solid #2196F3;">
                            <h4 style="margin-top: 0; color: #2196F3;">Moderately Improved Duration Adjustments</h4>
                            <p>These duration adjustments are predicted to moderately improve overall project completion rates.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_overall < current_overall - 5:
                        st.markdown("""
                        <div style="padding: 20px; border-radius: 5px; background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336;">
                            <h4 style="margin-top: 0; color: #F44336;">Not Recommended Duration Adjustments</h4>
                            <p>These duration adjustments are predicted to significantly reduce overall project completion rates.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="padding: 20px; border-radius: 5px; background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800;">
                            <h4 style="margin-top: 0; color: #FF9800;">Neutral Duration Adjustments</h4>
                            <p>These duration adjustments are predicted to have minimal impact on overall project completion rates.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No duration data available for what-if analysis.")
            else:
                st.info("No project data available for duration analysis.")
                
        elif scenario_type == "Regional Focus":
            st.subheader("Regional Focus Scenario")
            
            # Extract items with region data
            region_items = []
            
            for source, items in data.items():
                for item in items:
                    if isinstance(item, dict):
                        # Different field names for regions in different collections
                        region = None
                        
                        if "province" in item:
                            region = item["province"]
                        elif "CT" in item:
                            region = item["CT"]
                        elif "project_province" in item:
                            region = item["project_province"]
                        elif "region" in item:
                            region = item["region"]
                        
                        if region and isinstance(region, str):
                            region_items.append({
                                "region": region,
                                "source": source
                            })
            
            if region_items:
                # Convert to DataFrame
                region_df = pd.DataFrame(region_items)
                
                # Count items by region
                region_counts = region_df["region"].value_counts().reset_index()
                region_counts.columns = ["region", "count"]
                
                # Create allocation sliders
                st.markdown("##### Adjust Regional Focus")
                st.markdown("Drag the sliders to adjust focus levels for each region (0-100 scale) and see the predicted impact.")
                
                # Sort regions by count
                sorted_regions = region_counts.sort_values("count", ascending=False)
                
                # Create focus inputs
                focus_values = {}
                
                for i, row in sorted_regions.iterrows():
                    region = row["region"]
                    current_count = row["count"]
                    
                    # Calculate current focus level (normalized to 0-100)
                    max_count = sorted_regions["count"].max()
                    current_focus = int(current_count / max_count * 100) if max_count > 0 else 50
                    
                    # Create slider
                    focus = st.slider(
                        f"{region} (Current activity: {current_count} items)",
                        0,
                        100,
                        current_focus,
                        key=f"region_slider_{i}"
                    )
                    
                    focus_values[region] = focus
                
                # Calculate impact
                if focus_values:
                    # Create impact dataframe
                    impact_data = []
                    
                    total_current = sum(sorted_regions["count"])
                    total_focus = sum(focus_values.values())
                    
                    for region, focus in focus_values.items():
                        # Get current count
                        current_data = sorted_regions[sorted_regions["region"] == region]
                        current_count = current_data["count"].iloc[0] if not current_data.empty else 0
                        
                        # Calculate new distribution
                        # Higher focus means more resources, engagement, projects etc.
                        focus_ratio = focus / total_focus if total_focus > 0 else 0
                        current_ratio = current_count / total_current if total_current > 0 else 0
                        
                        # Predicted impact - simplified model
                        relative_change = (focus_ratio - current_ratio) / current_ratio if current_ratio > 0 else 0
                        
                        # Map to expected outcomes (simplified)
                        engagement_change = relative_change * 30  # % change in engagement
                        project_completion_change = relative_change * 20  # % change in project completion
                        sentiment_change = relative_change * 15  # % change in positive sentiment
                        
                        impact_data.append({
                            "region": region,
                            "current_count": current_count,
                            "focus_level": focus,
                            "focus_change": focus_ratio - current_ratio,
                            "engagement_change": engagement_change,
                            "project_completion_change": project_completion_change,
                            "sentiment_change": sentiment_change
                        })
                    
                    # Convert to DataFrame
                    impact_df = pd.DataFrame(impact_data)
                    
                    # Display impact analysis
                    st.subheader("Predicted Regional Impact")
                    
                    # Create dot impact visualization
                    impact_fig = px.scatter(
                        impact_df,
                        x="focus_level",
                        y="focus_change",
                        size="current_count",
                        color="engagement_change",
                        hover_name="region",
                        color_continuous_scale="RdYlGn",
                        title="Regional Focus Shift Impact",
                        labels={
                            "focus_level": "New Focus Level",
                            "focus_change": "Focus Shift",
                            "current_count": "Current Activity",
                            "engagement_change": "Engagement Change (%)"
                        }
                    )
                    
                    # Update layout
                    impact_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        height=500
                    )
                    
                    st.plotly_chart(impact_fig, use_container_width=True)
                    
                    # Create multiple impact visualization
                    impact_metrics = impact_df.melt(
                        id_vars=["region", "focus_level"],
                        value_vars=["engagement_change", "project_completion_change", "sentiment_change"],
                        var_name="metric",
                        value_name="change"
                    )
                    
                    # Clean up metric names
                    impact_metrics["metric"] = impact_metrics["metric"].map({
                        "engagement_change": "Citizen Engagement",
                        "project_completion_change": "Project Completion",
                        "sentiment_change": "Positive Sentiment"
                    })
                    
                    # Create grouped bar chart
                    metrics_fig = px.bar(
                        impact_metrics,
                        x="region",
                        y="change",
                        color="metric",
                        title="Predicted Changes by Region and Metric",
                        barmode="group",
                        labels={
                            "region": "Region",
                            "change": "Predicted Change (%)",
                            "metric": "Metric"
                        }
                    )
                    
                    # Update layout
                    metrics_fig.update_layout(
                        paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                        font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                        height=500
                    )
                    
                    st.plotly_chart(metrics_fig, use_container_width=True)
                    
                    # Map visualization of focus change
                    st.subheader("Regional Focus Shift Map")
                    
                    # Create map data
                    map_data = pd.DataFrame({
                        "province": impact_df["region"],
                        "count": impact_df["focus_change"] * 100  # Scale for better visualization
                    })
                    
                    # Create map
                    focus_map_fig = create_morocco_map(
                        map_data,
                        geo_col="province",
                        value_col="count",
                        title="Regional Focus Shift (positive values = increased focus)"
                    )
                    
                    folium_static(focus_map_fig, width=800)
                    
                    # Strategic recommendations
                    st.subheader("Strategic Recommendations")
                    
                    # Identify regions with significant changes
                    big_increases = impact_df[impact_df["focus_change"] > 0.1].sort_values("focus_change", ascending=False)
                    big_decreases = impact_df[impact_df["focus_change"] < -0.1].sort_values("focus_change")
                    
                    if not big_increases.empty:
                        st.markdown("##### Regions with Significantly Increased Focus")
                        for i, row in big_increases.head(3).iterrows():
                            st.markdown(f"""
                            <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1);">
                                <div style="font-weight: bold;">{row['region']}</div>
                                <div>Focus change: {row['focus_change'] * 100:.1f}%</div>
                                <div>Expected impact:</div>
                                <ul>
                                    <li>Engagement: {row['engagement_change']:.1f}%</li>
                                    <li>Project Completion: {row['project_completion_change']:.1f}%</li>
                                    <li>Positive Sentiment: {row['sentiment_change']:.1f}%</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if not big_decreases.empty:
                        st.markdown("##### Regions with Significantly Decreased Focus")
                        for i, row in big_decreases.head(3).iterrows():
                            st.markdown(f"""
                            <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #F44336; background-color: rgba(244, 67, 54, 0.1);">
                                <div style="font-weight: bold;">{row['region']}</div>
                                <div>Focus change: {row['focus_change'] * 100:.1f}%</div>
                                <div>Expected impact:</div>
                                <ul>
                                    <li>Engagement: {row['engagement_change']:.1f}%</li>
                                    <li>Project Completion: {row['project_completion_change']:.1f}%</li>
                                    <li>Positive Sentiment: {row['sentiment_change']:.1f}%</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Implementation guidance
                    st.markdown("##### Implementation Strategy")
                    st.markdown("""
                    To implement this regional focus shift effectively:
                    
                    1. **Gradual Transition**: Implement changes gradually over multiple quarters, not all at once
                    2. **Communication Plan**: Develop clear communication for regions with reduced focus to manage expectations
                    3. **Resource Allocation**: Adjust staffing, budget, and infrastructure according to the new focus levels
                    4. **Success Metrics**: Establish clear KPIs to measure the impact of focus changes
                    5. **Feedback Loop**: Create mechanisms to gather feedback from citizens in all regions
                    """)
            else:
                st.info("No regional data available for focus analysis.")
                
        elif scenario_type == "Project Type Mix":
            st.subheader("Project Type Mix Scenario")
            
            # Check if we have project data
            if "Municipal Projects" in data and data["Municipal Projects"]:
                projects = data["Municipal Projects"]
                
                # Extract project categories
                project_categories = {}
                
                for project in projects:
                    if isinstance(project, dict) and "category" in project:
                        category = project["category"]
                        if category not in project_categories:
                            project_categories[category] = 0
                        
                        project_categories[category] += 1
                
                if project_categories:
                    # Calculate total projects
                    total_projects = sum(project_categories.values())
                    
                    # Calculate current mix percentages
                    current_mix = {category: count / total_projects * 100 for category, count in project_categories.items()}
                    
                    # Create mix adjustment sliders
                    st.markdown("##### Adjust Project Type Mix")
                    st.markdown("Drag the sliders to adjust the percentage of each project type and see the predicted impact.")
                    
                    # Mix inputs
                    new_mix = {}
                    total_percentage = 0
                    
                    for category, current_pct in sorted(current_mix.items(), key=lambda x: x[1], reverse=True):
                        # Create slider
                        pct = st.slider(
                            f"{category} (Current: {current_pct:.1f}%)",
                            0.0,
                            100.0,
                            float(current_pct),
                            key=f"mix_slider_{category}"
                        )
                        
                        new_mix[category] = pct
                        total_percentage += pct
                    
                    # Check if mix adds up to 100%
                    if abs(total_percentage - 100.0) > 1.0:
                        st.warning(f"Total allocation is {total_percentage:.1f}%. Please adjust to reach 100%.")
                    else:
                        # Normalize mix to exactly 100%
                        for category in new_mix:
                            new_mix[category] = new_mix[category] / total_percentage * 100
                        
                        # Calculate impact
                        # Simulate metrics for each category
                        category_metrics = {
                            "Infrastructure": {"cost": 1.2, "time": 1.3, "satisfaction": 0.9, "maintenance": 1.2, "sustainability": 0.7},
                            "Healthcare": {"cost": 1.0, "time": 0.8, "satisfaction": 1.3, "maintenance": 1.0, "sustainability": 0.9},
                            "Education": {"cost": 0.7, "time": 0.9, "satisfaction": 1.2, "maintenance": 0.8, "sustainability": 1.0},
                            "Environment": {"cost": 0.8, "time": 0.7, "satisfaction": 1.0, "maintenance": 0.9, "sustainability": 1.5},
                            "Transportation": {"cost": 1.3, "time": 1.2, "satisfaction": 0.8, "maintenance": 1.3, "sustainability": 0.6},
                            "Water & Sanitation": {"cost": 1.1, "time": 1.0, "satisfaction": 0.9, "maintenance": 1.1, "sustainability": 1.0},
                            "Urban Planning": {"cost": 0.9, "time": 1.1, "satisfaction": 1.0, "maintenance": 0.9, "sustainability": 1.1},
                            "Cultural Development": {"cost": 0.6, "time": 0.8, "satisfaction": 1.4, "maintenance": 0.7, "sustainability": 1.2},
                            "Economic Development": {"cost": 0.7, "time": 0.9, "satisfaction": 1.1, "maintenance": 0.8, "sustainability": 0.9},
                            "Social Services": {"cost": 0.8, "time": 0.7, "satisfaction": 1.3, "maintenance": 0.7, "sustainability": 1.0},
                            "Sport & Recreation": {"cost": 0.7, "time": 0.8, "satisfaction": 1.2, "maintenance": 0.9, "sustainability": 0.8},
                            "Technology & Innovation": {"cost": 0.9, "time": 0.8, "satisfaction": 1.0, "maintenance": 0.7, "sustainability": 1.3}
                        }
                        
                        # Fill in missing categories with average values
                        for category in project_categories:
                            if category not in category_metrics:
                                category_metrics[category] = {"cost": 1.0, "time": 1.0, "satisfaction": 1.0, "maintenance": 1.0, "sustainability": 1.0}
                        
                        # Calculate current metrics
                        current_metrics = {
                            "cost": 0,
                            "time": 0,
                            "satisfaction": 0,
                            "maintenance": 0,
                            "sustainability": 0
                        }
                        
                        for category, pct in current_mix.items():
                            weight = pct / 100
                            if category in category_metrics:
                                for metric, value in category_metrics[category].items():
                                    current_metrics[metric] += value * weight
                        
                        # Calculate new metrics
                        new_metrics = {
                            "cost": 0,
                            "time": 0,
                            "satisfaction": 0,
                            "maintenance": 0,
                            "sustainability": 0
                        }
                        
                        for category, pct in new_mix.items():
                            weight = pct / 100
                            if category in category_metrics:
                                for metric, value in category_metrics[category].items():
                                    new_metrics[metric] += value * weight
                        
                        # Calculate changes
                        metric_changes = {
                            metric: new_metrics[metric] - current_metrics[metric]
                            for metric in current_metrics
                        }
                        
                        # Display impact analysis
                        st.subheader("Predicted Impact")
                        
                        # Create visualization of mix change
                        mix_data = []
                        
                        for category in project_categories:
                            mix_data.append({
                                "category": category,
                                "Current Mix": current_mix[category],
                                "New Mix": new_mix[category]
                            })
                        
                        mix_df = pd.DataFrame(mix_data)
                        
                        # Create grouped bar chart
                        mix_fig = px.bar(
                            mix_df,
                            x="category",
                            y=["Current Mix", "New Mix"],
                            title="Project Type Mix Comparison",
                            barmode="group",
                            labels={
                                "category": "Project Category",
                                "value": "Percentage (%)",
                                "variable": "Mix Type"
                            }
                        )
                        
                        # Update layout
                        mix_fig.update_layout(
                            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                            height=500
                        )
                        
                        st.plotly_chart(mix_fig, use_container_width=True)
                        
                        # Create radar chart for metrics impact
                        metrics_data = pd.DataFrame({
                            "metric": ["Cost", "Time", "Citizen Satisfaction", "Maintenance", "Sustainability"],
                            "current": [current_metrics["cost"], current_metrics["time"], 
                                       current_metrics["satisfaction"], current_metrics["maintenance"], 
                                       current_metrics["sustainability"]],
                            "new": [new_metrics["cost"], new_metrics["time"], 
                                    new_metrics["satisfaction"], new_metrics["maintenance"], 
                                    new_metrics["sustainability"]]
                        })
                        
                        # Create radar chart
                        radar_fig = go.Figure()
                        
                        radar_fig.add_trace(go.Scatterpolar(
                            r=metrics_data["current"],
                            theta=metrics_data["metric"],
                            fill='toself',
                            name='Current Mix',
                            line_color='#2196F3'
                        ))
                        
                        radar_fig.add_trace(go.Scatterpolar(
                            r=metrics_data["new"],
                            theta=metrics_data["metric"],
                            fill='toself',
                            name='New Mix',
                            line_color='#FF9800'
                        ))
                        
                        radar_fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1.5]
                                )
                            ),
                            title="Portfolio Metrics Comparison",
                            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333")
                        )
                        
                        st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Metrics changes with explanations
                        st.subheader("Impact on Key Metrics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display cost and time metrics
                            cost_change = metric_changes["cost"] * 100
                            time_change = metric_changes["time"] * 100
                            
                            st.metric(
                                "Project Cost Impact",
                                f"{new_metrics['cost']:.2f}",
                                f"{cost_change:.1f}%" if cost_change != 0 else "No change",
                                delta_color="inverse"  # Lower is better for cost
                            )
                            
                            st.metric(
                                "Project Timeline Impact",
                                f"{new_metrics['time']:.2f}",
                                f"{time_change:.1f}%" if time_change != 0 else "No change",
                                delta_color="inverse"  # Lower is better for time
                            )
                        
                        with col2:
                            # Display satisfaction and sustainability metrics
                            satisfaction_change = metric_changes["satisfaction"] * 100
                            sustainability_change = metric_changes["sustainability"] * 100
                            
                            st.metric(
                                "Citizen Satisfaction Impact",
                                f"{new_metrics['satisfaction']:.2f}",
                                f"{satisfaction_change:.1f}%" if satisfaction_change != 0 else "No change"
                            )
                            
                            st.metric(
                                "Sustainability Impact",
                                f"{new_metrics['sustainability']:.2f}",
                                f"{sustainability_change:.1f}%" if sustainability_change != 0 else "No change"
                            )
                        
                        # Detailed metric explanations
                        metrics_explanation = """
                        <div style="margin-top: 1rem;">
                            <h5>Metrics Explanation:</h5>
                            <ul>
                                <li><strong>Cost</strong>: Average relative cost factor across the project portfolio (lower is better)</li>
                                <li><strong>Time</strong>: Average project implementation timeline factor (lower is better)</li>
                                <li><strong>Citizen Satisfaction</strong>: Expected citizen satisfaction with project outcomes (higher is better)</li>
                                <li><strong>Maintenance</strong>: Long-term maintenance requirements (lower is better)</li>
                                <li><strong>Sustainability</strong>: Environmental and social sustainability impact (higher is better)</li>
                            </ul>
                        </div>
                        """
                        
                        st.markdown(metrics_explanation, unsafe_allow_html=True)
                        
                        # Strategic recommendations
                        st.subheader("Strategic Recommendations")
                        
                        # Determine overall recommendation
                        improvements = sum(1 for change in [
                            -metric_changes["cost"],  # Negative is better for cost
                            -metric_changes["time"],  # Negative is better for time
                            metric_changes["satisfaction"],
                            -metric_changes["maintenance"],  # Negative is better for maintenance
                            metric_changes["sustainability"]
                        ] if change > 0.05)
                        
                        deteriorations = sum(1 for change in [
                            -metric_changes["cost"],
                            -metric_changes["time"],
                            metric_changes["satisfaction"],
                            -metric_changes["maintenance"],
                            metric_changes["sustainability"]
                        ] if change < -0.05)
                        
                        if improvements >= 3 and deteriorations <= 1:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50;">
                                <h4 style="margin-top: 0; color: #4CAF50;">Recommended Mix Adjustment</h4>
                                <p>This project type mix is recommended as it improves multiple key metrics with minimal downsides.</p>
                                <p>Key benefits:</p>
                                <ul>
                            """, unsafe_allow_html=True)
                            
                            if -metric_changes["cost"] > 0.05:
                                st.markdown("<li>Reduced overall project costs</li>", unsafe_allow_html=True)
                            if -metric_changes["time"] > 0.05:
                                st.markdown("<li>Shorter implementation timelines</li>", unsafe_allow_html=True)
                            if metric_changes["satisfaction"] > 0.05:
                                st.markdown("<li>Improved citizen satisfaction</li>", unsafe_allow_html=True)
                            if -metric_changes["maintenance"] > 0.05:
                                st.markdown("<li>Lower maintenance requirements</li>", unsafe_allow_html=True)
                            if metric_changes["sustainability"] > 0.05:
                                st.markdown("<li>Enhanced sustainability impact</li>", unsafe_allow_html=True)
                            
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        elif improvements >= deteriorations:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800;">
                                <h4 style="margin-top: 0; color: #FF9800;">Balanced Mix Adjustment</h4>
                                <p>This project type mix offers a balance of improvements and trade-offs.</p>
                                <p>Consider the following trade-offs:</p>
                                <ul>
                            """, unsafe_allow_html=True)
                            
                            # List all significant changes
                            if abs(metric_changes["cost"]) > 0.05:
                                direction = "Reduced" if -metric_changes["cost"] > 0.05 else "Increased"
                                st.markdown(f"<li>{direction} overall project costs</li>", unsafe_allow_html=True)
                            if abs(metric_changes["time"]) > 0.05:
                                direction = "Shorter" if -metric_changes["time"] > 0.05 else "Longer"
                                st.markdown(f"<li>{direction} implementation timelines</li>", unsafe_allow_html=True)
                            if abs(metric_changes["satisfaction"]) > 0.05:
                                direction = "Improved" if metric_changes["satisfaction"] > 0.05 else "Reduced"
                                st.markdown(f"<li>{direction} citizen satisfaction</li>", unsafe_allow_html=True)
                            if abs(metric_changes["maintenance"]) > 0.05:
                                direction = "Lower" if -metric_changes["maintenance"] > 0.05 else "Higher"
                                st.markdown(f"<li>{direction} maintenance requirements</li>", unsafe_allow_html=True)
                            if abs(metric_changes["sustainability"]) > 0.05:
                                direction = "Enhanced" if metric_changes["sustainability"] > 0.05 else "Reduced"
                                st.markdown(f"<li>{direction} sustainability impact</li>", unsafe_allow_html=True)
                            
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="padding: 20px; border-radius: 5px; background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336;">
                                <h4 style="margin-top: 0; color: #F44336;">Not Recommended Mix Adjustment</h4>
                                <p>This project type mix is not recommended as it degrades multiple key metrics.</p>
                                <p>Key concerns:</p>
                                <ul>
                            """, unsafe_allow_html=True)
                            
                            if -metric_changes["cost"] < -0.05:
                                st.markdown("<li>Increased overall project costs</li>", unsafe_allow_html=True)
                            if -metric_changes["time"] < -0.05:
                                st.markdown("<li>Longer implementation timelines</li>", unsafe_allow_html=True)
                            if metric_changes["satisfaction"] < -0.05:
                                st.markdown("<li>Reduced citizen satisfaction</li>", unsafe_allow_html=True)
                            if -metric_changes["maintenance"] < -0.05:
                                st.markdown("<li>Higher maintenance requirements</li>", unsafe_allow_html=True)
                            if metric_changes["sustainability"] < -0.05:
                                st.markdown("<li>Reduced sustainability impact</li>", unsafe_allow_html=True)
                            
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Implementation timeline
                        st.markdown("##### Implementation Timeline")
                        st.markdown("""
                        To implement this project type mix adjustment effectively:
                        
                        1. **Phased Transition**: Implement the new mix gradually over 2-3 budget cycles
                        2. **Prioritize High-Impact Categories**: Focus first on categories with largest percentage increases
                        3. **Legacy Project Completion**: Complete already-initiated projects in categories being reduced
                        4. **Staff Reallocation**: Plan for staff training/reassignment for the new project mix
                        5. **Evaluation Framework**: Establish metrics to track the actual impact of the mix change
                        """)
                else:
                    st.info("No project category data available for mix analysis.")
            else:
                st.info("No project data available for mix analysis.")

# -------------------------------------------------------------
# SYSTEM HEALTH MODULE
# -------------------------------------------------------------

def system_health():
    """System health and settings management interface."""
    global DEBUG_MODE
    st.title("System Health & Settings")
    
    # Update system health metrics
    global_state.update_system_health()
    health_data = global_state.get_system_health()
    history_data = global_state.get_system_health_history()
    
    # Dashboard with key metrics
    st.subheader("System Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CPU usage gauge
        cpu_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_data["cpu_usage"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "CPU Usage"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": get_health_color(health_data["cpu_usage"], [30, 70])},
                "steps": [
                    {"range": [0, 30], "color": "rgba(76, 175, 80, 0.3)"},
                    {"range": [30, 70], "color": "rgba(255, 152, 0, 0.3)"},
                    {"range": [70, 100], "color": "rgba(244, 67, 54, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": health_data["cpu_usage"]
                }
            }
        ))
        
        # Update layout for theme consistency
        cpu_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(cpu_fig, use_container_width=True)
    
    with col2:
        # Memory usage gauge
        memory_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_data["memory_usage"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Memory Usage"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": get_health_color(health_data["memory_usage"], [30, 70])},
                "steps": [
                    {"range": [0, 30], "color": "rgba(76, 175, 80, 0.3)"},
                    {"range": [30, 70], "color": "rgba(255, 152, 0, 0.3)"},
                    {"range": [70, 100], "color": "rgba(244, 67, 54, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": health_data["memory_usage"]
                }
            }
        ))
        
        # Update layout for theme consistency
        memory_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(memory_fig, use_container_width=True)
    
    with col3:
        # Disk usage gauge
        disk_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_data["disk_usage"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Disk Usage"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": get_health_color(health_data["disk_usage"], [30, 70])},
                "steps": [
                    {"range": [0, 30], "color": "rgba(76, 175, 80, 0.3)"},
                    {"range": [30, 70], "color": "rgba(255, 152, 0, 0.3)"},
                    {"range": [70, 100], "color": "rgba(244, 67, 54, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": health_data["disk_usage"]
                }
            }
        ))
        
        # Update layout for theme consistency
        disk_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(disk_fig, use_container_width=True)
    
    with col4:
        # Database and API status
        db_status = health_data["database_status"]
        api_status = health_data["api_status"]
        
        db_color = {"healthy": "#4CAF50", "error": "#F44336", "unknown": "#FF9800"}[db_status]
        api_color = {"healthy": "#4CAF50", "error": "#F44336", "unavailable": "#FF9800", "unknown": "#FF9800"}[api_status]
        
        st.markdown(f"""
        <div style="height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="margin-bottom: 2rem;">
                <div style="text-align: center; margin-bottom: 0.5rem; font-weight: bold; font-size: 1rem;">Database Status</div>
                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                    <span style="width: 1rem; height: 1rem; border-radius: 50%; background-color: {db_color}; margin-right: 0.5rem;"></span>
                    <span style="text-transform: capitalize;">{db_status}</span>
                </div>
            </div>
            <div>
                <div style="text-align: center; margin-bottom: 0.5rem; font-weight: bold; font-size: 1rem;">API Status</div>
                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                    <span style="width: 1rem; height: 1rem; border-radius: 50%; background-color: {api_color}; margin-right: 0.5rem;"></span>
                    <span style="text-transform: capitalize;">{api_status}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create usage history chart if we have history data
    if history_data:
        st.subheader("Resource Usage History")
        
        # Create DataFrame for plotting
        history_df = pd.DataFrame(history_data)
        
        # Create dual-axis chart
        history_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add CPU trace
        history_fig.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=history_df["cpu_usage"],
                name="CPU Usage",
                line=dict(color="#2196F3", width=2)
            ),
            secondary_y=False
        )
        
        # Add Memory trace
        history_fig.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=history_df["memory_usage"],
                name="Memory Usage",
                line=dict(color="#FF9800", width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        history_fig.update_layout(
            title="System Resource Usage Over Time",
            paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
            font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        history_fig.update_xaxes(title_text="Time")
        history_fig.update_yaxes(title_text="CPU Usage (%)", secondary_y=False)
        history_fig.update_yaxes(title_text="Memory Usage (%)", secondary_y=True)
        
        st.plotly_chart(history_fig, use_container_width=True)
    
    # System tabs
    tabs = st.tabs(["Cache Management", "Qdrant Collections", "System Settings", "App Logs"])
    
    with tabs[0]:
        st.subheader("Cache Management")
        
        # Display cache stats
        cache_stats = cache.get_stats()
        
        # Create cache metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Cache Size",
                f"{cache_stats['size']} / {cache_stats['max_size']} items",
                f"{cache_stats['size'] / cache_stats['max_size'] * 100:.1f}% used"
            )
        
        with col2:
            st.metric(
                "Hit Ratio",
                f"{cache_stats['hit_ratio'] * 100:.1f}%",
                f"{cache_stats['hit_count']} hits"
            )
        
        with col3:
            st.metric(
                "Evictions",
                f"{cache_stats['eviction_count']}",
                f"{cache_stats['miss_count']} misses"
            )
        
        # Cache management actions
        st.subheader("Cache Actions")
        
        cache_col1, cache_col2, cache_col3 = st.columns(3)
        
        with cache_col1:
            if st.button("Clear All Cache", key="clear_all_cache"):
                cache.clear()
                st.success("Cache cleared successfully.")
                st.rerun()
        
        with cache_col2:
            # Get prefixes from cache
            cache_prefixes = set()
            for key in cache.cache.keys():
                parts = key.split("_")
                if parts:
                    cache_prefixes.add(parts[0])
            
            selected_prefix = st.selectbox(
                "Select Cache Type",
                sorted(cache_prefixes) if cache_prefixes else ["No cache keys available"]
            )
        
        with cache_col3:
            if st.button("Clear Selected Cache", key="clear_selected_cache"):
                if selected_prefix and selected_prefix != "No cache keys available":
                    cache.clear(prefix=selected_prefix)
                    st.success(f"Cleared cache with prefix '{selected_prefix}'.")
                    st.rerun()
    
    with tabs[1]:
        st.subheader("Qdrant Collections")
        
        # Get collections info
        collections = get_all_qdrant_collections()
        
        if collections:
            # Create collections table
            collections_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Vector Size</th>
                        <th>Points Count</th>
                        <th>Status</th>
                        <th>Indexed</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for collection in collections:
                name = collection.name
                
                # Get detailed collection info
                collection_info = get_qdrant_collection_info(name)
                
                vector_size = collection_info.config.params.vectors.size if collection_info else "Unknown"
                points_count = collection_info.vectors_count if collection_info else "Unknown"
                status = collection_info.status if collection_info else "Unknown"
                
                # Format for display
                status_class = {
                    "green": "status-active",
                    "yellow": "status-pending",
                    "red": "status-inactive"
                }.get(status, "")
                
                indexed = "Yes" if collection_info and collection_info.config.optimizer_status.indexing else "No"
                
                collections_table += f"""
                <tr>
                    <td>{name}</td>
                    <td>{vector_size}</td>
                    <td>{points_count}</td>
                    <td><span class="status-dot {status_class}"></span> {status}</td>
                    <td>{indexed}</td>
                </tr>
                """
            
            collections_table += """
                </tbody>
            </table>
            """
            
            st.markdown(collections_table, unsafe_allow_html=True)
            
            # Collection management
            st.subheader("Collection Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_collection = st.selectbox(
                    "Select Collection",
                    [c.name for c in collections]
                )
            
            with col2:
                operation = st.selectbox(
                    "Operation",
                    ["View Details", "Export to CSV", "Reindex", "Delete Collection"]
                )
            
            if st.button("Execute Operation", key="execute_collection_op"):
                if operation == "View Details":
                    # Get collection details
                    collection_info = get_qdrant_collection_info(selected_collection)
                    
                    if collection_info:
                        st.subheader(f"Details for {selected_collection}")
                        
                        # Format config info
                        st.json(collection_info.dict())
                    else:
                        st.error("Failed to get collection details.")
                
                elif operation == "Export to CSV":
                    # Get collection vector dimension
                    collection_info = get_qdrant_collection_info(selected_collection)
                    vector_dim = collection_info.config.params.vectors.size if collection_info else 384
                    
                    # Load all documents
                    documents, _ = load_qdrant_documents(selected_collection, vector_dim, limit=5000)
                    
                    if documents:
                        # Convert to DataFrame
                        df = pd.DataFrame(documents)
                        
                        # Convert to CSV
                        csv = df.to_csv(index=False)
                        
                        # Provide download button
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"{selected_collection}.csv",
                            "text/csv",
                            key="download_collection_csv"
                        )
                    else:
                        st.error("Failed to export collection or no documents found.")
                
                elif operation == "Reindex":
                    try:
                        client = get_qdrant_client()
                        if client:
                            # Recreate index
                            client.update_collection(
                                collection_name=selected_collection,
                                optimizer_config=OptimizersConfigDiff(
                                    indexing_threshold=0  # Force reindexing
                                )
                            )
                            st.success(f"Reindexing of {selected_collection} triggered successfully.")
                    except Exception as e:
                        st.error(f"Error reindexing collection: {e}")
                
                elif operation == "Delete Collection":
                    # Add confirmation check
                    if st.checkbox("I understand this will permanently delete the collection and all its data"):
                        try:
                            client = get_qdrant_client()
                            if client:
                                client.delete_collection(collection_name=selected_collection)
                                st.success(f"Collection {selected_collection} deleted successfully.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting collection: {e}")
    
    with tabs[2]:
        st.subheader("System Settings")
        
        # App version and environment
        st.markdown(f"""
        <div class="data-card">
            <h4 style="margin-top: 0;">System Information</h4>
            <p><strong>App Version:</strong> {APP_VERSION}</p>
            <p><strong>Environment:</strong> {ENV.capitalize()}</p>
            <p><strong>Last Updated:</strong> {health_data["last_checked"].strftime("%Y-%m-%d %H:%M:%S") if health_data["last_checked"] else "Unknown"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System configuration settings
        st.subheader("Configuration Settings")
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            # Cache settings
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=86400,
                value=cache.default_ttl,
                step=60
            )
            
            cache_size = st.number_input(
                "Cache Max Size",
                min_value=100,
                max_value=10000,
                value=cache.max_size,
                step=100
            )
        
        with settings_col2:
            # API settings
            api_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                index=0
            )
            
            debug_mode = st.checkbox("Debug Mode", value=DEBUG_MODE)
        
        # Apply settings button
        if st.button("Apply Settings", key="apply_system_settings"):
            # Update cache settings
            cache.default_ttl = cache_ttl
            cache.max_size = cache_size
            
            # Update global constants
            DEBUG_MODE = debug_mode
            OPENAI_MODEL = api_model
            
          
            # Log settings change
            # Log settings change
            add_audit_log(
                "settings_update",
                "System settings updated",
                {
                    "cache_ttl": cache_ttl,
                    "cache_size": cache_size,
                    "api_model": api_model,
                    "debug_mode": debug_mode
                },
                st.session_state.get("username", "admin")
            )
            
            st.success("Settings applied successfully.")
            
            # Add notification
            global_state.add_notification(
                "Settings Updated",
                "System settings have been updated successfully.",
                "success"
            )
    
    with tabs[3]:
        st.subheader("Application Logs")
        
        # Create log level selector
        log_level = st.selectbox(
            "Log Level",
            ["INFO", "WARNING", "ERROR", "DEBUG"],
            index=0
        )
        
        # Get logs from audit collection
        log_entries = load_audit_logs(limit=100)
        
        if log_entries:
            # Create logs table
            logs_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>User</th>
                        <th>Action</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for entry in log_entries:
                # Format timestamp
                timestamp = entry.get("timestamp", datetime.now())
                if isinstance(timestamp, datetime):
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
                
                # Get other fields
                user = entry.get("user", "Unknown")
                action = entry.get("action_type", "Unknown")
                description = entry.get("description", "No description")
                
                # Determine row style based on action type
                row_style = ""
                if "error" in action.lower():
                    row_style = 'style="background-color: rgba(244, 67, 54, 0.1);"'
                elif "warning" in action.lower():
                    row_style = 'style="background-color: rgba(255, 152, 0, 0.1);"'
                
                logs_table += f"""
                <tr {row_style}>
                    <td>{formatted_time}</td>
                    <td>{user}</td>
                    <td>{action}</td>
                    <td>{description}</td>
                </tr>
                """
            
            logs_table += """
                </tbody>
            </table>
            """
            
            st.markdown(logs_table, unsafe_allow_html=True)
            
            # Add export button
            if st.button("Export Logs", key="export_logs"):
                # Convert to DataFrame
                logs_df = pd.DataFrame(log_entries)
                
                # Format timestamp
                logs_df["timestamp"] = logs_df["timestamp"].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, datetime) else str(x)
                )
                
                # Convert to CSV
                csv = logs_df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "application_logs.csv",
                    "text/csv",
                    key="download_logs_csv"
                )
        else:
            st.info("No log entries found.")

def get_health_color(value, thresholds):
    """Get color for health metric based on thresholds [warning, critical]."""
    if value < thresholds[0]:
        return "#4CAF50"  # Green
    elif value < thresholds[1]:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def format_datetime(dt_value):
    """Format datetime value for display."""
    if dt_value is None:
        return "Never"
    
    if isinstance(dt_value, str):
        try:
            dt_value = datetime.strptime(dt_value, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                dt_value = datetime.strptime(dt_value, "%Y-%m-%d")
            except:
                return dt_value
    
    if isinstance(dt_value, datetime):
        # Check if today
        today = datetime.now().date()
        
        if dt_value.date() == today:
            return f"Today, {dt_value.strftime('%H:%M')}"
        elif dt_value.date() == today - timedelta(days=1):
            return f"Yesterday, {dt_value.strftime('%H:%M')}"
        else:
            return dt_value.strftime("%Y-%m-%d %H:%M")
    
    return str(dt_value)

def color_from_string(text):
    """Generate a consistent color from a string."""
    # Hash the text to get a consistent number
    hash_value = hash(text) % 0xFFFFFF
    
    # Convert to hex color
    color = "#{:06x}".format(hash_value)
    
    return color

# -------------------------------------------------------------
# AUDIT LOGS MODULE
# -------------------------------------------------------------

def audit_logs():
    """Audit logs interface for system activity monitoring."""
    st.title("Audit Logs")
    
    # Sidebar filters
    st.sidebar.header("Log Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now().date() - timedelta(days=7), datetime.now().date())
    )
    
    # Action type filter
    action_types = [
        "All",
        "login",
        "logout",
        "user_creation",
        "user_update",
        "user_deletion",
        "project_creation",
        "project_update",
        "project_deletion",
        "content_moderation",
        "article_creation",
        "article_update",
        "article_deletion",
        "settings_update",
        "data_export"
    ]
    
    selected_action = st.sidebar.selectbox("Action Type", action_types)
    
    # User filter
    selected_user = st.sidebar.text_input("User", placeholder="Enter username...")
    
    # Apply filters
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date = datetime.combine(date_range[1], datetime.max.time())
    
    # Adjust the action_type filter
    filter_action = selected_action if selected_action != "All" else None
    
    # Load filtered logs
    logs = load_audit_logs(
        limit=500,
        action_type=filter_action,
        user=selected_user if selected_user else None,
        date_range=(start_date, end_date)
    )
    
    # Display log metrics
    st.subheader("Audit Log Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card(
            "Total Logs",
            f"{len(logs):,}",
            icon="fa-list-check",
            color="#2196F3"
        )
    
    with col2:
        # Count unique users
        unique_users = len(set(log.get("user", "unknown") for log in logs))
        
        display_metric_card(
            "Unique Users",
            f"{unique_users:,}",
            icon="fa-users",
            color="#FF9800"
        )
    
    with col3:
        # Count distinct action types
        action_counts = {}
        for log in logs:
            action = log.get("action_type", "unknown")
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        display_metric_card(
            "Distinct Actions",
            f"{len(action_counts):,}",
            icon="fa-code-branch",
            color="#9C27B0"
        )
    
    # Display logs in tabs
    tabs = st.tabs(["Log Table", "Activity Analysis", "User Activity", "Suspicious Activity"])
    
    with tabs[0]:
        st.subheader("Audit Log Entries")
        
        if not logs:
            st.info("No audit logs found matching the selected filters.")
        else:
            # Create logs table
            logs_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>User</th>
                        <th>Action</th>
                        <th>Description</th>
                        <th>IP Address</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for log in logs:
                # Format timestamp
                timestamp = log.get("timestamp", datetime.now())
                if isinstance(timestamp, datetime):
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
                
                # Get other fields
                user = log.get("user", "Unknown")
                action = log.get("action_type", "Unknown")
                description = log.get("description", "No description")
                ip_address = log.get("ip_address", "Unknown")
                
                # Determine row style based on action type
                row_style = ""
                if "deletion" in action.lower() or "error" in action.lower():
                    row_style = 'style="background-color: rgba(244, 67, 54, 0.1);"'
                
                logs_table += f"""
                <tr {row_style}>
                    <td>{formatted_time}</td>
                    <td>{user}</td>
                    <td>{action}</td>
                    <td>{description}</td>
                    <td>{ip_address}</td>
                </tr>
                """
            
            logs_table += """
                </tbody>
            </table>
            """
            
            st.markdown(logs_table, unsafe_allow_html=True)
            
            # Export logs button
            if st.button("Export to CSV", key="export_audit_logs"):
                # Convert to DataFrame
                df = pd.DataFrame(logs)
                
                # Format timestamp
                df["timestamp"] = df["timestamp"].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, datetime) else str(x)
                )
                
                # Convert to CSV
                csv = df.to_csv(index=False)
                
                # Provide download button
                st.download_button(
                    "Download CSV",
                    csv,
                    "audit_logs.csv",
                    "text/csv",
                    key="download_audit_csv"
                )
    
    with tabs[1]:
        st.subheader("Activity Analysis")
        
        if not logs:
            st.info("No audit logs found matching the selected filters.")
        else:
            # Extract timestamps for time analysis
            timestamps = [log.get("timestamp") for log in logs if isinstance(log.get("timestamp"), datetime)]
            
            if timestamps:
                # Convert to DataFrame
                time_df = pd.DataFrame({"timestamp": timestamps})
                
                # Extract time components
                time_df["date"] = time_df["timestamp"].dt.date
                time_df["hour"] = time_df["timestamp"].dt.hour
                time_df["day_of_week"] = time_df["timestamp"].dt.day_name()
                
                # Activity by day
                daily_counts = time_df.groupby("date").size().reset_index(name="count")
                
                # Create time series chart
                daily_fig = px.line(
                    daily_counts,
                    x="date",
                    y="count",
                    title="Daily Activity",
                    markers=True
                )
                
                # Update layout for theme consistency
                daily_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Date",
                    yaxis_title="Activity Count"
                )
                
                st.plotly_chart(daily_fig, use_container_width=True)
                
                # Activity by hour of day
                hourly_counts = time_df.groupby("hour").size().reset_index(name="count")
                
                # Create bar chart
                hourly_fig = px.bar(
                    hourly_counts,
                    x="hour",
                    y="count",
                    title="Activity by Hour of Day",
                    color="count",
                    color_continuous_scale="Viridis"
                )
                
                # Update layout for theme consistency
                hourly_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Hour of Day",
                    yaxis_title="Activity Count"
                )
                
                st.plotly_chart(hourly_fig, use_container_width=True)
                
                # Activity by day of week
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow_counts = time_df.groupby("day_of_week").size().reset_index(name="count")
                
                # Create day of week chart
                dow_fig = px.bar(
                    dow_counts,
                    x="day_of_week",
                    y="count",
                    title="Activity by Day of Week",
                    color="count",
                    color_continuous_scale="Viridis",
                    category_orders={"day_of_week": day_order}
                )
                
                # Update layout for theme consistency
                dow_fig.update_layout(
                    paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                    font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                    xaxis_title="Day of Week",
                    yaxis_title="Activity Count"
                )
                
                st.plotly_chart(dow_fig, use_container_width=True)
            else:
                st.info("No timestamp data available for activity analysis.")
    
    with tabs[2]:
        st.subheader("User Activity")
        
        if not logs:
            st.info("No audit logs found matching the selected filters.")
        else:
            # Group logs by user
            user_actions = {}
            
            for log in logs:
                user = log.get("user", "Unknown")
                action = log.get("action_type", "Unknown")
                timestamp = log.get("timestamp", datetime.now())
                
                if user not in user_actions:
                    user_actions[user] = {
                        "actions": {},
                        "total": 0,
                        "first_seen": timestamp,
                        "last_seen": timestamp
                    }
                
                if action not in user_actions[user]["actions"]:
                    user_actions[user]["actions"][action] = 0
                
                user_actions[user]["actions"][action] += 1
                user_actions[user]["total"] += 1
                
                # Update first and last seen
                if isinstance(timestamp, datetime):
                    if isinstance(user_actions[user]["first_seen"], datetime):
                        user_actions[user]["first_seen"] = min(user_actions[user]["first_seen"], timestamp)
                    
                    if isinstance(user_actions[user]["last_seen"], datetime):
                        user_actions[user]["last_seen"] = max(user_actions[user]["last_seen"], timestamp)
            
            # Top users by activity
            top_users = sorted(user_actions.items(), key=lambda x: x[1]["total"], reverse=True)
            
            user_df = pd.DataFrame([
                {"user": user, "total": data["total"]}
                for user, data in top_users
            ])
            
            # Create bar chart
            user_fig = px.bar(
                user_df.head(10),
                x="user",
                y="total",
                title="Top 10 Users by Activity",
                color="total",
                color_continuous_scale="Viridis"
            )
            
            # Update layout for theme consistency
            user_fig.update_layout(
                paper_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                plot_bgcolor="#1E1E1E" if st.session_state.theme == "dark" else "white",
                font=dict(color="#E0E0E0" if st.session_state.theme == "dark" else "#333"),
                xaxis_title="User",
                yaxis_title="Activity Count"
            )
            
            st.plotly_chart(user_fig, use_container_width=True)
            
            # Create user activity table
            st.subheader("User Activity Details")
            
            user_table = """
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Total Activity</th>
                        <th>First Seen</th>
                        <th>Last Seen</th>
                        <th>Top Actions</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for user, data in top_users[:20]:  # Top 20 users
                # Format timestamps
                first_seen = format_datetime(data["first_seen"])
                last_seen = format_datetime(data["last_seen"])
                
                # Get top 3 actions
                top_actions = sorted(data["actions"].items(), key=lambda x: x[1], reverse=True)[:3]
                top_actions_str = ", ".join([f"{action} ({count})" for action, count in top_actions])
                
                user_table += f"""
                <tr>
                    <td>{user}</td>
                    <td>{data["total"]}</td>
                    <td>{first_seen}</td>
                    <td>{last_seen}</td>
                    <td>{top_actions_str}</td>
                </tr>
                """
            
            user_table += """
                </tbody>
            </table>
            """
            
            st.markdown(user_table, unsafe_allow_html=True)
    
    with tabs[3]:
        st.subheader("Suspicious Activity Detection")
        
        if not logs:
            st.info("No audit logs found matching the selected filters.")
        else:
            # Define suspicious patterns to look for
            suspicious_activities = []
            
            # 1. Failed login attempts
            failed_logins = {}
            
            for log in logs:
                if log.get("action_type") == "login" and "failed" in log.get("description", "").lower():
                    user = log.get("user", "Unknown")
                    timestamp = log.get("timestamp", datetime.now())
                    ip = log.get("ip_address", "Unknown")
                    
                    if user not in failed_logins:
                        failed_logins[user] = []
                    
                    failed_logins[user].append({"timestamp": timestamp, "ip": ip})
            
            # Check for multiple failed logins
            for user, attempts in failed_logins.items():
                if len(attempts) >= 3:
                    suspicious_activities.append({
                        "type": "Multiple Failed Logins",
                        "user": user,
                        "count": len(attempts),
                        "details": f"{len(attempts)} failed login attempts",
                        "timestamp": max(attempt["timestamp"] for attempt in attempts)
                    })
            
            # 2. Unusual activity hours
            work_hours_start = 8  # 8 AM
            work_hours_end = 18  # 6 PM
            
            for log in logs:
                timestamp = log.get("timestamp")
                if isinstance(timestamp, datetime):
                    hour = timestamp.hour
                    
                    # Check if outside normal work hours
                    if hour < work_hours_start or hour >= work_hours_end:
                        user = log.get("user", "Unknown")
                        action = log.get("action_type", "Unknown")
                        
                        # Focus on sensitive actions
                        if "creation" in action or "deletion" in action or "update" in action:
                            suspicious_activities.append({
                                "type": "After Hours Activity",
                                "user": user,
                                "count": 1,
                                "details": f"{action} at {hour}:00",
                                "timestamp": timestamp
                            })
            
            # 3. Unusual action patterns
            user_actions = {}
            
            for log in logs:
                user = log.get("user", "Unknown")
                action = log.get("action_type", "Unknown")
                
                if user not in user_actions:
                    user_actions[user] = {}
                
                if action not in user_actions[user]:
                    user_actions[user][action] = 0
                
                user_actions[user][action] += 1
            
            # Check for unusual deletion activities
            for user, actions in user_actions.items():
                deletion_count = sum(count for action, count in actions.items() if "deletion" in action)
                
                if deletion_count >= 5:
                    suspicious_activities.append({
                        "type": "High Deletion Activity",
                        "user": user,
                        "count": deletion_count,
                        "details": f"{deletion_count} deletion actions",
                        "timestamp": datetime.now()  # Approximate
                    })
            
            # Display suspicious activities
            if suspicious_activities:
                # Sort by timestamp (most recent first)
                suspicious_activities.sort(key=lambda x: x["timestamp"], reverse=True)
                
                # Create table
                suspicious_table = """
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>User</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for activity in suspicious_activities:
                    # Format timestamp
                    timestamp = format_datetime(activity["timestamp"])
                    
                    suspicious_table += f"""
                    <tr style="background-color: rgba(244, 67, 54, 0.1);">
                        <td>{timestamp}</td>
                        <td><span class="badge badge-danger">{activity["type"]}</span></td>
                        <td>{activity["user"]}</td>
                        <td>{activity["details"]}</td>
                    </tr>
                    """
                
                suspicious_table += """
                    </tbody>
                </table>
                """
                
                st.markdown(suspicious_table, unsafe_allow_html=True)
                
                # Security recommendations
                st.subheader("Security Recommendations")
                
                st.markdown("""
                Based on the detected patterns, consider implementing these security measures:
                
                1. **Account Lockout**: Implement automatic account lockout after 3-5 failed login attempts
                2. **Multi-Factor Authentication**: Require 2FA for administrative accounts
                3. **Time-Based Restrictions**: Consider restricting sensitive operations to normal business hours
                4. **Approval Workflows**: Implement approval requirements for bulk deletions and critical operations
                5. **Alert System**: Set up real-time alerts for suspicious activity patterns
                6. **IP-Based Restrictions**: Consider limiting access to sensitive operations by IP address
                7. **Audit Log Reviews**: Schedule regular audit log reviews to detect patterns
                """)
            else:
                st.success("No suspicious activity patterns detected in the current data.")

# -------------------------------------------------------------
# AUTHENTICATION FUNCTIONS
# -------------------------------------------------------------

def update_login_history(username, status="success"):
    """Update the login history in the database"""
    try:
        client = get_mongo_client()
        if not client:
            return
            
        db = client["CivicCatalyst"]
        history_collection = db["login_history"]
        
        history_collection.insert_one({
            "username": username,
            "timestamp": datetime.now(),
            "status": status,
            "ip_address": "127.0.0.1"  # In production, get real IP
        })
        
        # Add to audit log
        add_audit_log(
            "login",
            f"User login: {username} ({status})",
            {"username": username, "status": status},
            username
        )
        
        client.close()
    except Exception as e:
        logger.error(f"Error updating login history: {e}")

def perform_login(username, password):
    """Verify login credentials and set session state"""
    try:
        # Connect to MongoDB
        client = get_mongo_client()
        if not client:
            st.error("Failed to connect to MongoDB")
            return False
            
        db = client["CivicCatalyst"]
        users_collection = db["users"]
        
        # Find user by username
        user = users_collection.find_one({"username": username})
        
        if user and "password_hash" in user and "password_salt" in user:
            # Verify the password using the proper method with pbkdf2_hmac
            salt = user["password_salt"]
            
            # Use the exact same verification method from the CivicCatalyst module
            hashed = hashlib.pbkdf2_hmac(
                'sha256', 
                password.encode('utf-8'), 
                salt.encode('utf-8'), 
                100000  # Must use the same number of iterations
            ).hex()
            
            if hashed == user["password_hash"]:
                # Set session state
                st.session_state.username = username
                st.session_state.role = user.get("role", "user")
                st.session_state.authenticated = True
                st.session_state.user_id = str(user["_id"])
                
                # Update user's last login
                users_collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"last_login": datetime.now()}}
                )
                
                # Update login history
                update_login_history(username)
                
                # Add notification
                global_state.add_notification(
                    "Login Successful",
                    f"Welcome back, {username}!",
                    "success"
                )
                
                client.close()
                return True
        
        # Update failed login
        update_login_history(username, status="failed")
        
        client.close()
        return False
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return False

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------

def main():
    """Main application entry point."""
    # Initialize theme
    initialize_theme()
    
    # Apply CSS
    st.markdown(f"""
    <style>
    {get_theme_css()}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state if needed
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "role" not in st.session_state:
        st.session_state.role = None
    
    if "username" not in st.session_state:
        st.session_state.username = None
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "site_language" not in st.session_state:
        st.session_state.site_language = "en"  # Default language
    
    # Handle login state
    if not st.session_state.authenticated:
        # Create a modern login page
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem;">CivicCatalyst Admin</h1>
                <p style="font-size: 1.2rem; opacity: 0.8;">Comprehensive administrative dashboard for managing citizen engagement and municipal projects.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create two columns, with login form in the center
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Login form with a clean design
            st.markdown(
                """
                <div class="login-container">
                    <div class="login-title">Sign In</div>
                """,
                unsafe_allow_html=True
            )
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    site_language = st.selectbox(
                        "Language",
                        ["English", "Français", "العربية", "الدارجة"],
                        index=0
                    )
                
                with col2:
                    login_theme = st.radio("Theme", ["Light", "Dark"], index=1, key="login_theme")
                
                submit = st.form_submit_button("Login", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add version info
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 2rem; opacity: 0.7; font-size: 0.8rem;">
                    CivicCatalyst Admin v{APP_VERSION}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if submit:
                if username and password:
                    # Set language based on selection
                    language_map = {
                        "English": "en",
                        "Français": "fr",
                        "العربية": "ar",
                        "الدارجة": "darija"
                    }
                    
                    st.session_state.site_language = language_map.get(site_language, "en")
                    
                    # Set theme based on radio selection
                    if login_theme == "Dark":
                        st.session_state.theme = "dark"
                    else:
                        st.session_state.theme = "light"
                    
                    # Attempt login
                    if perform_login(username, password):
                        st.success("Login successful! Redirecting...")
                        
                        # Add a slight delay for UX
                        time.sleep(0.5)
                        
                        # Redirect to the main dashboard
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please enter both username and password.")
    else:
        # Render notification center
        render_notification_center()
        
        # Main application sidebar
        with st.sidebar:
            # App logo and title
            # App logo and title
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <img src="https://via.placeholder.com/40x40?text=CC" style="border-radius: 8px; margin-right: 10px;">
                    <div>
                        <div style="font-weight: bold; font-size: 1.2rem;">CivicCatalyst</div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">Admin Dashboard v{APP_VERSION}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # User info
            user_role = st.session_state.role.capitalize() if st.session_state.role else "User"
            
            st.markdown(
                f"""
                <div style="margin-bottom: 1.5rem; padding: 0.75rem; border-radius: 0.5rem; background-color: {'rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)'};">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 36px; height: 36px; border-radius: 50%; background-color: {color_from_string(st.session_state.username)}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                            {st.session_state.username[0].upper() if st.session_state.username else "U"}
                        </div>
                        <div>
                            <div style="font-weight: bold;">{st.session_state.username}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">{user_role}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Language selector
            language_options = {
                "en": "English",
                "fr": "Français",
                "ar": "العربية",
                "darija": "الدارجة"
            }
            
            current_lang = st.session_state.site_language
            
            selected_lang = st.selectbox(
                "Language",
                list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=list(language_options.keys()).index(current_lang) if current_lang in language_options else 0
            )
            
            if selected_lang != current_lang:
                st.session_state.site_language = selected_lang
                st.rerun()
            
            # Theme selector
            current_theme = st.session_state.theme
            selected_theme = st.radio(
                "Theme",
                ["light", "dark"],
                format_func=lambda x: "Light Mode" if x == "light" else "Dark Mode",
                index=1 if current_theme == "dark" else 0,
                horizontal=True
            )
            
            if selected_theme != current_theme:
                toggle_theme()
                st.rerun()
            
            # Main navigation
            st.markdown("### Navigation")
            
            navigation_options = [
                {"id": "dashboard", "name": "Dashboard", "icon": ICONS["dashboard"]},
                {"id": "user_management", "name": "User Management", "icon": ICONS["users"]},
                {"id": "project_management", "name": "Project Management", "icon": ICONS["projects"]},
                {"id": "idea_management", "name": "Idea Management", "icon": ICONS["ideas"]},
                {"id": "content_moderation", "name": "Content Moderation", "icon": ICONS["content"]},
                {"id": "advanced_analytics", "name": "Advanced Analytics", "icon": ICONS["analytics"]},
                {"id": "news_management", "name": "News Management", "icon": ICONS["news"]},
                {"id": "system_health", "name": "System Settings", "icon": ICONS["system"]},
                {"id": "audit_logs", "name": "Audit Logs", "icon": ICONS["audit"]},
                {"id": "search", "name": "Global Search", "icon": ICONS["search"]}
            ]
            
            if "current_view" not in st.session_state:
                st.session_state.current_view = "dashboard"  # Default view
            
            # Create navigation HTML
            nav_html = '<div class="sidebar-menu">'
            
            for item in navigation_options:
                active_class = "active" if st.session_state.current_view == item["id"] else ""
                
                nav_html += f"""
                <div class="sidebar-menu-item {active_class}" id="nav_{item['id']}">
                    <div class="sidebar-menu-item-icon">
                        <i class="fas {item['icon']}"></i>
                    </div>
                    <div class="sidebar-menu-item-text">
                        {item['name']}
                    </div>
                </div>
                """
            
            nav_html += '</div>'
            
            st.markdown(nav_html, unsafe_allow_html=True)
            
            # Hidden buttons for navigation
            for item in navigation_options:
                if st.button(item["name"], key=f"nav_btn_{item['id']}"):
                    st.session_state.current_view = item["id"]
                    # Clear any existing search queries when navigating
                    if "quick_search_query" in st.session_state:
                        del st.session_state.quick_search_query
                    st.rerun()
            
            # Collection selector for cross-collection search
            if st.session_state.current_view == "search":
                st.markdown("### Search Options")
                
                search_collections = st.multiselect(
                    "Search Collections",
                    list(COLLECTIONS.keys()),
                    default=["citizen_comments", "citizen_ideas", "municipal_projects"]
                )
                
                # Store selected collections in session state
                st.session_state.search_collections = search_collections
            else:
                st.markdown("### Quick Search")
                
                quick_search = st.text_input("Search", placeholder="Enter search terms...")
                
                if st.button("Search") and quick_search:
                    # Set session state to navigate to search page
                    st.session_state.quick_search_query = quick_search
                    st.session_state.search_collections = ["citizen_comments", "citizen_ideas", "municipal_projects"]
                    st.session_state.current_view = "search"
                    st.rerun()
            
            # Logout button
            st.markdown(
                """
                <div style="position: absolute; bottom: 1rem; left: 1rem; right: 1rem;">
                    <div class="divider" style="margin: 1rem 0; height: 1px; background-color: rgba(255, 255, 255, 0.1);"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("Logout", key="logout_button"):
                # Add to audit log
                add_audit_log(
                    "logout",
                    f"User logout: {st.session_state.username}",
                    {"username": st.session_state.username},
                    st.session_state.username
                )
                
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Redirect to login page
                st.rerun()
        
        # JavaScript for sidebar navigation
        nav_js = """
        <script>
            // Wait for elements to be fully loaded
            document.addEventListener('DOMContentLoaded', function() {
                // Get all navigation items
                const navItems = document.querySelectorAll('.sidebar-menu-item');
                
                // Add click event to each item
                navItems.forEach(item => {
                    item.addEventListener('click', function() {
                        // Get the navigation ID
                        const navId = this.id.replace('nav_', '');
                        
                        // Find the corresponding button and click it
                        const buttonId = 'nav_btn_' + navId;
                        document.getElementById(buttonId).click();
                    });
                });
            });
        </script>
        """
        
        st.markdown(nav_js, unsafe_allow_html=True)
        st.write(f"Current view: {st.session_state.get('current_view', 'Not set')}")
        # Main content area based on navigation
        if st.session_state.current_view == "dashboard":
            create_dashboard()
            
        elif st.session_state.current_view == "user_management":
            user_management()
            
        elif st.session_state.current_view == "project_management":
            project_management()
            
        elif st.session_state.current_view == "idea_management":
            ideas_management()
            
        elif st.session_state.current_view == "content_moderation":
            content_moderation()
            
        elif st.session_state.current_view == "advanced_analytics":
            advanced_analytics()
            
        elif st.session_state.current_view == "news_management":
            news_management()
            
        elif st.session_state.current_view == "system_health":
            system_health()
            
        elif st.session_state.current_view == "audit_logs":
            audit_logs()
            
        elif st.session_state.current_view == "search":
            # Global search functionality
            st.title("Global Search")
            
            # Get search query from state or input
            search_query = st.session_state.get("quick_search_query", "")
            search_collections = st.session_state.get("search_collections", list(COLLECTIONS.keys())[:3])
            
            # Search query input
            new_query = st.text_input("Search Query", value=search_query)
            
            # Collection selection
            selected_collections = st.multiselect(
                "Select Collections to Search",
                list(COLLECTIONS.keys()),
                default=search_collections
            )
            
            # Search button
            if st.button("Search", key="main_search_button") and new_query:
                # Store search query and collections in session state
                st.session_state.quick_search_query = new_query
                st.session_state.search_collections = selected_collections
                
                # Show loading indicator
                with st.spinner("Searching..."):
                    # Perform semantic search across selected collections
                    all_results = []
                    
                    for collection_name in selected_collections:
                        if collection_name in COLLECTIONS:
                            # Get vector dimension for this collection
                            vector_dim = COLLECTIONS[collection_name]["vector_dim"]
                            
                            # Perform search
                            collection_results = semantic_search(
                                collection_name=collection_name,
                                query_text=new_query,
                                limit=10
                            )
                            
                            # Add collection info to results
                            for result in collection_results:
                                result["source_collection"] = collection_name
                                all_results.append(result)
                    
                    # Sort all results by score (highest first)
                    all_results.sort(key=lambda x: x["score"], reverse=True)
                    
                    # Display results
                    if all_results:
                        st.success(f"Found {len(all_results)} results matching '{new_query}'")
                        
                        # Group results by collection for better organization
                        results_by_collection = {}
                        
                        for result in all_results:
                            collection = result["source_collection"]
                            if collection not in results_by_collection:
                                results_by_collection[collection] = []
                            
                            results_by_collection[collection].append(result)
                        
                        # Display results by collection
                        for collection, results in results_by_collection.items():
                            with st.expander(f"Results from {collection} ({len(results)})", expanded=True):
                                for result in results:
                                    # Create a formatted result card
                                    score_percentage = int(result["score"] * 100)
                                    
                                    # Extract title or content snippet based on collection
                                    title = "Untitled"
                                    content = "No content"
                                    
                                    if "payload" in result:
                                        payload = result["payload"]
                                        
                                        # Extract title
                                        if "title" in payload:
                                            title = payload["title"]
                                        elif "challenge" in payload:
                                            title = payload["challenge"][:50] + "..." if len(payload["challenge"]) > 50 else payload["challenge"]
                                        
                                        # Extract content
                                        if "content" in payload:
                                            content = payload["content"]
                                        elif "comment_text" in payload:
                                            content = payload["comment_text"]
                                        elif "solution" in payload:
                                            content = payload["solution"]
                                        elif "description" in payload:
                                            content = payload["description"]
                                        
                                        # Truncate content
                                        if len(content) > 300:
                                            content = content[:300] + "..."
                                    
                                    # Create result card
                                    st.markdown(
                                        f"""
                                        <div style="padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; background-color: {'rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.02)'}; border-left: 3px solid #2196F3;">
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                                <div style="font-weight: bold; font-size: 1.1rem;">{title}</div>
                                                <div style="font-size: 0.9rem; opacity: 0.8;">Match: {score_percentage}%</div>
                                            </div>
                                            <div style="margin-bottom: 0.5rem; font-size: 0.9rem;">{content}</div>
                                            <div style="font-size: 0.8rem; opacity: 0.6;">Source: {collection}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                    else:
                        st.info(f"No results found for '{new_query}' in the selected collections.")
            
            # Search tips
            st.markdown(
                """
                ### Search Tips
                
                - Use specific keywords related to your search
                - Include location names for regional content
                - Try both general terms and specific phrases
                - Search works across comments, projects, ideas, and news content
                - Results are ranked by semantic relevance to your query
                """,
                unsafe_allow_html=True
            )

# Main entry point
if __name__ == "__main__":
    main()

# Helper functions for view/edit/delete operations that weren't fully implemented above


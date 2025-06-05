import streamlit as st
import os
import hashlib
import time
import jwt
import re
import uuid
import json
import base64
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import random   
# External Dependencies
import pandas as pd
from gtts import gTTS
from pymongo import MongoClient, ASCENDING, DESCENDING
from io import BytesIO
from PIL import Image
import requests
import redis
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
# Load environment variables
SECRET_KEY = os.environ.get("SECRET_KEY", "civic_catalyst_secret_key_2024")
JWT_ALGORITHM = "HS256"
COOKIE_PASSWORD = os.environ.get("COOKIE_PASSWORD", "STRONG_PASSWORD_2024")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://154.44.186.241:27017")
REDIS_URL = os.environ.get("REDIS_URL", "redis://155.44.186.241:6379/0")

# Security settings
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME = 30  # minutes
PASSWORD_MIN_LENGTH = 10
PASSWORD_COMPLEXITY_SCORE = 3

# Session persistence settings
SESSION_EXPIRY = 21  # days
REFRESH_TOKEN_EXPIRY = 45  # days
AUTO_LOGOUT_WARNING = 5  # minutes

# UI/UX settings
DEFAULT_THEME = "cosmic"
AVAILABLE_THEMES = ["cosmic", "aurora", "sunset", "ocean", "forest", "neon", "royal", "minimal"]
NOTIFICATION_TYPES = ["success", "info", "warning", "error", "celebration"]

# Cache keys and TTL (seconds)
CACHE_TTL = {
    "user_profile": 300,       # 5 minutes
    "analytics": 600,          # 10 minutes
    "animations": 86400,       # 1 day
    "translations": 86400,     # 1 day
    "theme_data": 86400,       # 1 day
    "system_status": 60        # 1 minute
}

# Feature flags
FEATURE_FLAGS = {
    "advanced_analytics": True,
    "voice_synthesis": True,
    "redis_caching": True,
    "ai_assistance": True
}

# =============================================================================
# DATABASE & CACHING CONNECTIONS
# =============================================================================
# MongoDB connection pool singleton
class MongoConnection:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Use get_instance() to get the MongoDB connection")
        
        try:
            self.client = MongoClient(MONGO_URI, maxPoolSize=10, serverSelectionTimeoutMS=2000,connectTimeoutMS=2000, socketTimeoutMS=2000, connect=False)
            try:
                self.client.admin.command('ping')
            except Exception as e:
                print("Mongo ping failed:", e)

            self.db = self.client["CivicCatalyst"]
            
            # Verify connection
            self.client.server_info()
            
            # Initialize collections and indexes
            self._init_collections()
            
            # Clean up expired sessions
            self._cleanup_expired_data()
        except Exception as e:
            st.error(f"🔐 Database connection failed: {e}")
            self.client = None
            self.db = None
    
    def _init_collections(self):
        """Initialize collections with proper indexes"""
        if self.db is None:
            return
            
        collections_config = {
            "users": [
                ("username", {"unique": True}),
                ("email", {"unique": True}),
                ("created_at", {}),
                ("last_login", {})
            ],
            "sessions": [
                ("session_id", {"unique": True}),
                ("username", {}),
                ("expires", {"expireAfterSeconds": 0})
            ],
            "activity_log": [
                ("user", {}),
                ("timestamp", {}),
                ("action", {})
            ],
            "notifications": [
                ("user", {}),
                ("created_at", {}),
                ("read", {})
            ],
            "user_preferences": [
                ("username", {"unique": True}),
                ("updated_at", {})
            ]
        }
        
        for collection_name, indexes in collections_config.items():
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
            
            collection = self.db[collection_name]
            for field, options in indexes:
                try:
                    collection.create_index([(field, ASCENDING)], **options)
                except Exception:
                    # Index might already exist
                    pass
    
    def _cleanup_expired_data(self):
        """Clean up expired data to keep database optimized"""
        if self.db is None:
            return
            
        try:
            # Clean expired sessions
            self.db.sessions.delete_many({"expires": {"$lt": datetime.utcnow()}})
            
            # Keep only last 30 days of activity logs
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            self.db.activity_log.delete_many({"timestamp": {"$lt": thirty_days_ago}})
            
            # Clean old notifications
            ninety_days_ago = datetime.utcnow() - timedelta(days=90)
            self.db.notifications.delete_many({
                "created_at": {"$lt": ninety_days_ago},
                "read": True
            })
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
    
    def get_db(self):
        """Get database instance"""
        return self.db
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()

# Redis cache connection singleton
class RedisCache:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Use get_instance() to get the Redis cache")
        
        self.enabled = FEATURE_FLAGS.get("redis_caching", False)
        if not self.enabled:
            self.redis = None
            return
            
        try:
            self.redis = redis.from_url(
                REDIS_URL,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis.ping()
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}")
            self.redis = None
            self.enabled = False
    
    def get(self, key: str, default=None):
        """Get a value from cache"""
        if not self.enabled or not self.redis:
            return default
            
        try:
            value = self.redis.get(key)
            if value is None:
                return default
            return json.loads(value)
        except Exception:
            return default
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set a value in cache with TTL in seconds"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            serialized = json.dumps(value)
            return self.redis.setex(key, ttl, serialized)
        except Exception:
            return False
    
    def delete(self, key: str):
        """Delete a key from cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            return self.redis.delete(key)
        except Exception:
            return False
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return True
        except Exception:
            return False

# Get database and cache instances
def get_db():
    """Get MongoDB database instance"""
    return MongoConnection.get_instance().get_db()

def get_cache():
    """Get Redis cache instance"""
    return RedisCache.get_instance()

# =============================================================================
# ENHANCED SESSION MANAGEMENT
# =============================================================================
class SessionAuth:
    """Enhanced session-based authentication"""
    
    def __init__(self, prefix="civic_"):
        self.prefix = prefix
        # Initialize session state for auth data if not exists
        if f"{self.prefix}auth" not in st.session_state:
            st.session_state[f"{self.prefix}auth"] = {}
        
        # Initialize session tracking
        if f"{self.prefix}session_tracking" not in st.session_state:
            st.session_state[f"{self.prefix}session_tracking"] = {
                "login_attempts": 0,
                "last_activity": datetime.now(),
                "session_start": datetime.now(),
                "page_views": 0,
                "actions_performed": []
            }
    
    def ready(self):
        """Check if session auth is ready"""
        return True
    
    def get(self, key, default=None):
        """Get a value from session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return auth_data.get(key, default)
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key, value):
        """Set a value in session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        auth_data[key] = value
        st.session_state[f"{self.prefix}auth"] = auth_data
    
    def __delitem__(self, key):
        """Delete a value from session state"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        if key in auth_data:
            del auth_data[key]
            st.session_state[f"{self.prefix}auth"] = auth_data
    
    def __contains__(self, key):
        """Check if a key exists"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return key in auth_data
    
    def __len__(self):
        """Get number of keys"""
        auth_data = st.session_state.get(f"{self.prefix}auth", {})
        return len(auth_data)
    
    def save(self):
        """Update session tracking"""
        tracking = st.session_state.get(f"{self.prefix}session_tracking", {})
        tracking["last_activity"] = datetime.now()
        tracking["page_views"] = tracking.get("page_views", 0) + 1
        st.session_state[f"{self.prefix}session_tracking"] = tracking
    
    def track_action(self, action):
        """Track user actions for analytics"""
        tracking = st.session_state.get(f"{self.prefix}session_tracking", {})
        actions = tracking.get("actions_performed", [])
        actions.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "session_duration": (datetime.now() - tracking.get("session_start", datetime.now())).total_seconds()
        })
        tracking["actions_performed"] = actions[-50:]  # Keep last 50 actions
        st.session_state[f"{self.prefix}session_tracking"] = tracking

class SecureCookies:
    """Enhanced secure cookie alternative with encryption"""
    
    def __init__(self, prefix="civic_", password=COOKIE_PASSWORD):
        self.prefix = prefix
        self._is_ready = True
        
        # Create encryption key
        salt = b'streamlit_cookie_manager_secure'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet = Fernet(key)
        
        # Initialize cookies dict in session state
        if f"{self.prefix}cookies" not in st.session_state:
            st.session_state[f"{self.prefix}cookies"] = {}
            
        # Initialize cookie analytics
        if f"{self.prefix}cookie_analytics" not in st.session_state:
            st.session_state[f"{self.prefix}cookie_analytics"] = {
                "created": datetime.now(),
                "access_count": 0,
                "last_accessed": None,
                "modifications": []
            }
    
    def ready(self):
        """Check if cookie manager is ready"""
        return self._is_ready
    
    def get(self, key, default=None):
        """Get a cookie value with analytics tracking"""
        if not self.ready():
            return default
        
        # Update analytics
        analytics = st.session_state.get(f"{self.prefix}cookie_analytics", {})
        analytics["access_count"] = analytics.get("access_count", 0) + 1
        analytics["last_accessed"] = datetime.now()
        st.session_state[f"{self.prefix}cookie_analytics"] = analytics
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return cookies.get(key, default)
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key, value):
        """Set a cookie value with tracking"""
        if not self.ready():
            return
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        old_value = cookies.get(key)
        cookies[key] = value
        st.session_state[f"{self.prefix}cookies"] = cookies
        
        # Track modification
        analytics = st.session_state.get(f"{self.prefix}cookie_analytics", {})
        modifications = analytics.get("modifications", [])
        modifications.append({
            "key": key,
            "action": "set",
            "timestamp": datetime.now().isoformat(),
            "had_previous_value": old_value is not None
        })
        analytics["modifications"] = modifications[-100:]
        st.session_state[f"{self.prefix}cookie_analytics"] = analytics
    
    def __delitem__(self, key):
        """Delete a cookie with tracking"""
        if not self.ready():
            return
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        if key in cookies:
            del cookies[key]
            st.session_state[f"{self.prefix}cookies"] = cookies
            
            # Track deletion
            analytics = st.session_state.get(f"{self.prefix}cookie_analytics", {})
            modifications = analytics.get("modifications", [])
            modifications.append({
                "key": key,
                "action": "delete",
                "timestamp": datetime.now().isoformat()
            })
            analytics["modifications"] = modifications[-100:]
            st.session_state[f"{self.prefix}cookie_analytics"] = analytics
    
    def __contains__(self, key):
        """Check if a cookie exists"""
        if not self.ready():
            return False
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return key in cookies
    
    def __len__(self):
        """Get number of cookies"""
        if not self.ready():
            return 0
        
        cookies = st.session_state.get(f"{self.prefix}cookies", {})
        return len(cookies)
    
    def save(self):
        """Save cookies with tracking"""
        analytics = st.session_state.get(f"{self.prefix}cookie_analytics", {})
        analytics["last_saved"] = datetime.now()
        st.session_state[f"{self.prefix}cookie_analytics"] = analytics
    
    def load(self):
        """Load cookies with tracking"""
        analytics = st.session_state.get(f"{self.prefix}cookie_analytics", {})
        analytics["last_loaded"] = datetime.now()
        st.session_state[f"{self.prefix}cookie_analytics"] = analytics

# Helper function for cookie management
def get_cookies():
    """Get secure cookies instance"""
    return SecureCookies()

# =============================================================================
# TRANSLATIONS & LOCALIZATION
# =============================================================================
# Translation dictionary
translations = {
    "welcome_title": {
        "en": "🌟 Civic Catalyst",
        "fr": "🌟 Civic Catalyst",
        "ar": "🌟 المحفز المدني",
        "darija": "🌟 المحفز المدني"
    },
    "welcome_message": {
        "en": "Empowering Communities Through Intelligent Civic Engagement",
        "fr": "Autonomiser les communautés grâce à l'engagement civique intelligent",
        "ar": "تمكين المجتمعات من خلال المشاركة المدنية الذكية",
        "darija": "تقوية المجتمعات من خلال المشاركة المدنية الذكية"
    },
    "enhanced_welcome": {
        "en": "Welcome to the future of civic participation! Join thousands of citizens making a difference.",
        "fr": "Bienvenue dans l'avenir de la participation civique ! Rejoignez des milliers de citoyens qui font la différence.",
        "ar": "مرحباً بكم في مستقبل المشاركة المدنية! انضموا إلى آلاف المواطنين الذين يحدثون فرقاً.",
        "darija": "مرحباً بكم في مستقبل المشاركة المدنية! انضموا لآلاف المواطنين اللي كيدوزو فرق."
    },
    "login_title": {
        "en": "Login",
        "fr": "Connexion",
        "ar": "تسجيل الدخول",
        "darija": "دخول"
    },
    "register_header": {
        "en": "Register",
        "fr": "S'inscrire",
        "ar": "تسجيل",
        "darija": "تسجيل"
    },
    "forgot_password": {
        "en": "Forgot Password",
        "fr": "Mot de passe oublié",
        "ar": "نسيت كلمة المرور",
        "darija": "نسيت كلمة السر"
    },
    "welcome_back": {
        "en": "Welcome back",
        "fr": "Bon retour",
        "ar": "مرحباً بعودتك",
        "darija": "مرحباً بيك مرة أخرى"
    },
    "dashboard_welcome": {
        "en": "Welcome to your personalized dashboard",
        "fr": "Bienvenue sur votre tableau de bord personnalisé",
        "ar": "مرحباً بك في لوحة التحكم الشخصية",
        "darija": "مرحباً بيك ف لوحة التحكم ديالك"
    },
    "advanced_features": {
        "en": "Advanced Features",
        "fr": "Fonctionnalités avancées",
        "ar": "الميزات المتقدمة",
        "darija": "الميزات المتطورة"
    },
    "ai_assistance": {
        "en": "AI-Powered Assistance",
        "fr": "Assistance alimentée par l'IA",
        "ar": "المساعدة المدعومة بالذكاء الاصطناعي",
        "darija": "المساعدة بالذكاء الاصطناعي"
    },
    "voice_commands": {
        "en": "Voice Commands",
        "fr": "Commandes vocales",
        "ar": "الأوامر الصوتية",
        "darija": "الأوامر الصوتية"
    },
    "real_time_analytics": {
        "en": "Real-time Analytics",
        "fr": "Analyses en temps réel",
        "ar": "التحليلات في الوقت الفعلي",
        "darija": "التحليلات في الوقت الحقيقي"
    }
}

def get_translation(key: str, default: str = None) -> str:
    """Get translation with caching"""
    lang = st.session_state.get("site_language", "en")
    cache = get_cache()
    
    # Try to get from cache first
    cache_key = f"translation:{lang}:{key}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Get translation with fallback
    result = translations.get(key, {}).get(lang)
    if not result:
        result = translations.get(key, {}).get("en")
    if not result:
        result = default or key.replace("_", " ").title()
        
        # Log missing translation
        if "missing_translations" not in st.session_state:
            st.session_state["missing_translations"] = set()
        st.session_state["missing_translations"].add(f"{key}_{lang}")
    
    # Cache the result
    if result != key.replace("_", " ").title():  # Only cache valid translations
        cache.set(cache_key, result, CACHE_TTL["translations"])
    
    return result

# Translation function
def t(key: str) -> str:
    """Get translation for key"""
    return get_translation(key)

# =============================================================================
# VISUAL ASSETS & THEMING
# =============================================================================
# Animation resources
LOTTIE_ANIMATIONS = {
    "login": "https://assets8.lottiefiles.com/packages/lf20_hy4txm7l.json",
    "welcome": "https://assets5.lottiefiles.com/packages/lf20_g3dzz0po.json",
    "success": "https://assets3.lottiefiles.com/packages/lf20_rZQs19.json",
    "error": "https://assets5.lottiefiles.com/packages/lf20_qpwbiyxf.json",
    "loading": "https://assets3.lottiefiles.com/packages/lf20_Yey9E2.json",
    "profile": "https://assets3.lottiefiles.com/packages/lf20_qdexz4gx.json",
    "morocco": "https://assets10.lottiefiles.com/packages/lf20_UgZWvP.json",
    "celebration": "https://assets1.lottiefiles.com/packages/lf20_obhph3sh.json",
    "typing": "https://assets2.lottiefiles.com/packages/lf20_wwhbhx8v.json",
    "security": "https://assets1.lottiefiles.com/packages/lf20_pqwemv0k.json",
    "analytics": "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json",
    "settings": "https://assets4.lottiefiles.com/packages/lf20_1a8dx7zj.json",
    "rocket": "https://assets8.lottiefiles.com/packages/lf20_kkflmtur.json",
    "data": "https://assets2.lottiefiles.com/packages/lf20_dmw175hj.json",
    "notification": "https://assets1.lottiefiles.com/packages/lf20_txlkh1sz.json"
}

def load_lottie_animation(animation_key: str):
    """Load Lottie animation with caching"""
    if animation_key not in LOTTIE_ANIMATIONS:
        return None
        
    url = LOTTIE_ANIMATIONS[animation_key]
    cache = get_cache()
    
    # Try to get from cache first
    cache_key = f"animation:{animation_key}"
    cached_animation = cache.get(cache_key)
    if cached_animation:
        return cached_animation
    
    # Fetch animation
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
            
        animation_data = r.json()
        
        # Cache the animation
        cache.set(cache_key, animation_data, CACHE_TTL["animations"])
        
        return animation_data
    except Exception as e:
        print(f"Animation loading error: {e}")
        return None

def get_theme_colors(theme: str = None) -> Dict[str, str]:
    """Get theme colors with caching"""
    theme = theme or st.session_state.get("theme", DEFAULT_THEME)
    cache = get_cache()
    
    # Try to get from cache first
    cache_key = f"theme:{theme}"
    cached_colors = cache.get(cache_key)
    if cached_colors:
        return cached_colors
    
    # Theme color definitions
    themes = {
        "cosmic": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "accent": "#f093fb",
            "background": "linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.9)",
            "shadow": "0 20px 40px rgba(102, 126, 234, 0.3)"
        },
        "aurora": {
            "primary": "#48c6ef",
            "secondary": "#6f86d6",
            "accent": "#a8edea",
            "background": "linear-gradient(-45deg, #48c6ef, #6f86d6, #a8edea, #fed6e3)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.85)",
            "shadow": "0 20px 40px rgba(72, 198, 239, 0.3)"
        },
        "sunset": {
            "primary": "#ff9a56",
            "secondary": "#ff6b6b",
            "accent": "#feca57",
            "background": "linear-gradient(-45deg, #ff9a56, #ff6b6b, #feca57, #ff7675)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.9)",
            "shadow": "0 20px 40px rgba(255, 154, 86, 0.3)"
        },
        "ocean": {
            "primary": "#0080ff",
            "secondary": "#00c3ff",
            "accent": "#7ed321",
            "background": "linear-gradient(-45deg, #0080ff, #00c3ff, #7ed321, #50e3c2)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.9)",
            "shadow": "0 20px 40px rgba(0, 128, 255, 0.3)"
        },
        "forest": {
            "primary": "#56ab2f",
            "secondary": "#a8e6cf",
            "accent": "#88d8a3",
            "background": "linear-gradient(-45deg, #56ab2f, #a8e6cf, #88d8a3, #7fcdcd)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.9)",
            "shadow": "0 20px 40px rgba(86, 171, 47, 0.3)"
        },
        "neon": {
            "primary": "#ff006e",
            "secondary": "#8338ec",
            "accent": "#3a86ff",
            "background": "linear-gradient(-45deg, #ff006e, #8338ec, #3a86ff, #06ffa5)",
            "text": "#ffffff",
            "surface": "rgba(0, 0, 0, 0.8)",
            "shadow": "0 20px 40px rgba(255, 0, 110, 0.4)"
        },
        "royal": {
            "primary": "#8e44ad",
            "secondary": "#9b59b6",
            "accent": "#e74c3c",
            "background": "linear-gradient(-45deg, #8e44ad, #9b59b6, #e74c3c, #f39c12)",
            "text": "#ffffff",
            "surface": "rgba(255, 255, 255, 0.1)",
            "shadow": "0 20px 40px rgba(142, 68, 173, 0.4)"
        },
        "minimal": {
            "primary": "#2c3e50",
            "secondary": "#34495e",
            "accent": "#3498db",
            "background": "linear-gradient(-45deg, #ecf0f1, #bdc3c7, #95a5a6, #7f8c8d)",
            "text": "#2c3e50",
            "surface": "rgba(255, 255, 255, 0.95)",
            "shadow": "0 10px 30px rgba(44, 62, 80, 0.1)"
        }
    }
    
    colors = themes.get(theme, themes["cosmic"])
    
    # Cache the colors
    cache.set(cache_key, colors, CACHE_TTL["theme_data"])
    
    return colors

def apply_theme():
    """Apply theme with CSS injection"""
    theme = st.session_state.get("theme", DEFAULT_THEME)
    colors = get_theme_colors(theme)
    
    # Create particle effect
    particle_css = """
    <div class="particles">
    """
    
    # Generate random particles
    import random
    for i in range(20):
        left = random.randint(0, 100)
        delay = random.uniform(0, 6)
        duration = random.uniform(4, 8)
        
        particle_css += f"""
        <div class="particle" style="
            left: {left}%;
            animation-delay: {delay}s;
            animation-duration: {duration}s;
        "></div>
        """
    
    particle_css += "</div>"
    
    # Enhanced CSS with modern design principles
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {{
        --primary-color: {colors["primary"]};
        --secondary-color: {colors["secondary"]};
        --accent-color: {colors["accent"]};
        --background-gradient: {colors["background"]};
        --text-color: {colors["text"]};
        --surface-color: {colors["surface"]};
        --shadow: {colors["shadow"]};
        --border-radius: 16px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    * {{
        box-sizing: border-box;
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    
    .stApp {{
        background: var(--background-gradient);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }}
    
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        25% {{ background-position: 100% 50%; }}
        50% {{ background-position: 100% 100%; }}
        75% {{ background-position: 0% 100%; }}
    }}
    
    /* Particles */
    .particles {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }}
    
    .particle {{
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{
            transform: translateY(0) rotate(0deg);
            opacity: 1;
        }}
        50% {{
            transform: translateY(-100px) rotate(180deg);
            opacity: 0.5;
        }}
    }}
    
    /* Enhanced container styling */
    .enhanced-container {{
        background: var(--surface-color);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }}
    
    .enhanced-container::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        opacity: 0.8;
    }}
    
    .enhanced-container:hover {{
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }}
    
    /* Enhanced header styling */
    .enhanced-header {{
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
    }}
    
    .enhanced-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    
    .enhanced-subtitle {{
        font-size: 1.25rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.8;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    
    /* Enhanced button styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: var(--border-radius);
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        cursor: pointer;
        transition: var(--transition);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stButton > button::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* Enhanced form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        background: var(--surface-color);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        transition: var(--transition);
        backdrop-filter: blur(10px);
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {{
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: rgba(255, 255, 255, 0.9);
    }}
    
    /* Enhanced tabs */
    .stTabs {{
        background: var(--surface-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: var(--transition);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
    }}
    
    /* Enhanced metrics */
    .metric-card {{
        background: var(--surface-color);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: var(--shadow);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Enhanced alerts */
    .stAlert {{
        border-radius: var(--border-radius) !important;
        border: none !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
        animation: slideInAlert 0.3s ease-out forwards;
    }}
    
    @keyframes slideInAlert {{
        0% {{ 
            transform: translateY(-20px); 
            opacity: 0; 
        }}
        100% {{ 
            transform: translateY(0); 
            opacity: 1; 
        }}
    }}
    
    /* Enhanced sidebar */
    .css-1d391kg {{
        background: var(--surface-color) !important;
        backdrop-filter: blur(20px) !important;
    }}
    
    /* Enhanced loading animation */
    .loading-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }}
    
    .enhanced-loader {{
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 20px var(--primary-color);
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Enhanced progress bars */
    .progress-container {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }}
    
    .progress-bar {{
        height: 8px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 10px;
        transition: width 0.5s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .progress-bar::after {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    /* Enhanced floating elements */
    .floating-element {{
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-20px); }}
    }}
    
    /* Enhanced glassmorphism effect */
    .glassmorphism {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }}
    
    /* Enhanced typography */
    .enhanced-text {{
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: var(--text-color);
    }}
    
    .code-text {{
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0, 0, 0, 0.1);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .enhanced-title {{
            font-size: 2.5rem;
        }}
        
        .enhanced-container {{
            padding: 1.5rem;
            margin: 0.5rem 0;
        }}
        
        .stButton > button {{
            width: 100%;
            padding: 1rem;
        }}
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--primary-color);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--secondary-color);
    }}

    /* Notification system */
    .notification {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        max-width: 400px;
        animation: slideInNotification 0.3s ease-out forwards;
    }}
    
    @keyframes slideInNotification {{
        0% {{ transform: translateX(100%); opacity: 0; }}
        100% {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* Feature cards */
    .feature-card {{
        background: var(--surface-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: var(--transition);
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: var(--shadow);
    }}
    
    .feature-card h4 {{
        color: var(--primary-color);
        margin: 0.5rem 0;
    }}
    
    .feature-card p {{
        margin: 0;
        opacity: 0.8;
        line-height: 1.5;
    }}
    
    /* Quick access cards */
    .quick-access-card {{
        background: var(--surface-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: var(--transition);
        border: 2px solid transparent;
    }}
    
    .quick-access-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--shadow);
        border-color: var(--primary-color);
    }}
    
    /* Activity items */
    .activity-item {{
        background: var(--surface-color);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #666;
        transition: var(--transition);
    }}
    
    .activity-item:hover {{
        transform: translateX(5px);
    }}
    </style>
    {particle_css}
    """
    
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# UI COMPONENTS & NOTIFICATION SYSTEM
# =============================================================================
class NotificationSystem:
    """Enhanced notification system with animations"""
    
    @staticmethod
    def init():
        """Initialize notification system"""
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
    
    @staticmethod
    def add(message: str, notification_type: str = "info"):
        """Add a notification"""
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
        
        notification = {
            "message": message,
            "type": notification_type,
            "timestamp": time.time(),
            "id": str(uuid.uuid4())
        }
        
        st.session_state.notifications.append(notification)
    
    @staticmethod
    def render():
        """Render all notifications"""
        NotificationSystem.init()
        
        # Display notifications
        for i, notification in enumerate(st.session_state.notifications[:]):
            notification_type = notification.get("type", "info")
            message = notification.get("message", "")
            
            # Auto-remove notifications after 5 seconds
            if time.time() - notification.get("timestamp", 0) > 5:
                st.session_state.notifications.remove(notification)
                continue
            
            # Create notification HTML with animations
            notification_html = f"""
            <div class="notification notification-{notification_type}" 
                 style="top: {20 + (i * 70)}px;">
                <div style="background: var(--surface-color); backdrop-filter: blur(20px);
                            border-radius: 12px; padding: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            border-left: 4px solid var(--primary-color);">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">
                        {notification_type.title()} Notification
                    </div>
                    <div>{message}</div>
                </div>
            </div>
            """
            
            st.markdown(notification_html, unsafe_allow_html=True)

def create_language_selector():
    """Create language selector with flags"""
    lang_options = {
        "en": {"name": "English", "flag": "🇬🇧", "code": "en"},
        "fr": {"name": "Français", "flag": "🇫🇷", "code": "fr"},
        "ar": {"name": "العربية", "flag": "🇲🇦", "code": "ar"},
        "darija": {"name": "الدارجة", "flag": "🇲🇦", "code": "darija"}
    }
    
    current_lang = st.session_state.get("site_language", "en")
    
    # Create a container for the language selector
    st.markdown("#### 🌐 Select Language")
    
    # Use columns to create a horizontal layout
    cols = st.columns(len(lang_options))
    
    for idx, (code, info) in enumerate(lang_options.items()):
        with cols[idx]:
            # Create a button for each language
            is_selected = code == current_lang
            
            # Style the button based on selection
            if is_selected:
                st.markdown(f"""
                <div style="
                    background: var(--primary-color);
                    color: white;
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: var(--shadow);
                    transform: scale(1.05);
                ">
                    <div style="font-size: 2rem;">{info['flag']}</div>
                    <div style="font-weight: 600;">{info['name']}</div>
                    <div style="font-size: 0.8rem;">✓ Selected</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(
                    f"{info['flag']}\n{info['name']}", 
                    key=f"lang_{code}",
                    use_container_width=True
                ):
                    st.session_state["site_language"] = code
                    st.rerun
                    st.stop()

def create_progress_tracker():
    """Create a progress tracking system"""
    if "user_progress" not in st.session_state:
        st.session_state["user_progress"] = {
            "profile_completion": 45,
            "feature_exploration": 30,
            "community_engagement": 60,
            "achievement_points": 120
        }
    
    progress = st.session_state["user_progress"]
    
    progress_html = f"""
    <div class="progress-tracker" style="
        background: var(--surface-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <h3 style="text-align: center; margin-bottom: 2rem; color: var(--primary-color);">
            🎯 Your Progress Journey
        </h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
            <div class="progress-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>👤 Profile Completion</span>
                    <span style="font-weight: 600;">{progress['profile_completion']}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress['profile_completion']}%;"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>🔍 Feature Exploration</span>
                    <span style="font-weight: 600;">{progress['feature_exploration']}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress['feature_exploration']}%;"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>🤝 Community Engagement</span>
                    <span style="font-weight: 600;">{progress['community_engagement']}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress['community_engagement']}%;"></div>
                </div>
            </div>
            
            <div class="progress-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>🏆 Achievement Points</span>
                    <span style="font-weight: 600;">{progress['achievement_points']} pts</span>
                </div>
                <div style="background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); 
                            border-radius: 8px; padding: 0.5rem; text-align: center; color: white;">
                    Level {progress['achievement_points'] // 50 + 1}
                </div>
            </div>
        </div>
    </div>
    """
    
    return progress_html

def create_voice_system():
    """Create a voice feedback system"""
    if not FEATURE_FLAGS.get("voice_synthesis", False):
        return lambda text, lang, style: None
    
    def text_to_speech(text, lang="en", voice_style="normal"):
        """Text-to-speech with voice style options"""
        try:
            # Voice style mapping
            if voice_style == "slow":
                # Add pauses for slower speech
                text = text.replace(".", ". ... ").replace(",", ", .. ")
            elif voice_style == "excited":
                # Add excitement indicators
                text = text.replace(".", "!").replace("?", "?!")
            
            tts = gTTS(text=text, lang=lang, slow=(voice_style == "slow"))
            with BytesIO() as fp:
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_bytes = fp.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
            # Enhanced audio player with custom styling
            audio_html = f"""
            <div class="enhanced-audio-player" style="
                background: var(--surface-color);
                border-radius: 12px;
                padding: 1rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
            ">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem;">🔊</span>
                    <span style="font-weight: 500;">Voice Message ({lang.upper()})</span>
                </div>
                <audio controls style="width: 100%; height: 40px;">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            return audio_html
        except Exception as e:
            print(f"Voice synthesis error: {e}")
            return None
    
    return text_to_speech

# =============================================================================
# SECURITY & AUTHENTICATION FUNCTIONS
# =============================================================================
def hash_password(password: str) -> Tuple[bytes, bytes]:
    """Generate secure password hash"""
    import hashlib
    import os
    
    salt = os.urandom(32)  # 32 bytes of random salt
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt, 
        100000  # 100,000 iterations
    )
    return pwdhash, salt

def verify_password(stored_hash: bytes, stored_salt: bytes, provided_password: str) -> bool:
    """Verify password against stored hash"""
    import hashlib
    
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256', 
        provided_password.encode('utf-8'), 
        stored_salt, 
        100000  # Same number of iterations as used in hash_password
    )
    return pwdhash == stored_hash

def check_password_strength(password: str) -> Dict[str, Any]:
    """Enhanced password strength checking with detailed feedback"""
    score = 0
    feedback = []
    requirements_met = {
        'length': False,
        'uppercase': False,
        'lowercase': False,
        'digit': False,
        'special': False,
        'no_common': False,
        'no_sequential': False,
        'no_repeated': False
    }
    
    # Length check
    if len(password) >= PASSWORD_MIN_LENGTH:
        score += 2
        requirements_met['length'] = True
    else:
        feedback.append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
    
    # Character type checks
    if re.search(r'[A-Z]', password):
        score += 1
        requirements_met['uppercase'] = True
    else:
        feedback.append("Add uppercase letters")
    
    if re.search(r'[a-z]', password):
        score += 1
        requirements_met['lowercase'] = True
    else:
        feedback.append("Add lowercase letters")
    
    if re.search(r'\d', password):
        score += 1
        requirements_met['digit'] = True
    else:
        feedback.append("Add numbers")
    
    if re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        score += 1
        requirements_met['special'] = True
    else:
        feedback.append("Add special characters")
    
    # Advanced checks
    common_patterns = ['password', '123456', 'qwerty', 'admin', 'letmein', 'welcome']
    if not any(pattern in password.lower() for pattern in common_patterns):
        score += 1
        requirements_met['no_common'] = True
    else:
        feedback.append("Avoid common passwords")
    
    # Sequential character check
    if not re.search(r'(abc|bcd|cde|123|234|345|456|567|678|789)', password.lower()):
        score += 1
        requirements_met['no_sequential'] = True
    else:
        feedback.append("Avoid sequential characters")
    
    # Repeated character check
    if not re.search(r'(.)\1{2,}', password):
        score += 1
        requirements_met['no_repeated'] = True
    else:
        feedback.append("Avoid repeated characters")
    
    # Determine overall strength
    if score >= 7:
        strength = "excellent"
        strength_text = "Excellent"
        strength_color = "#27ae60"
    elif score >= 5:
        strength = "strong"
        strength_text = "Strong"
        strength_color = "#2ecc71"
    elif score >= 3:
        strength = "medium"
        strength_text = "Medium"
        strength_color = "#f39c12"
    else:
        strength = "weak"
        strength_text = "Weak"
        strength_color = "#e74c3c"
    
    is_valid = score >= PASSWORD_COMPLEXITY_SCORE
    
    return {
        'is_valid': is_valid,
        'score': score,
        'max_score': 8,
        'strength': strength,
        'strength_text': strength_text,
        'strength_color': strength_color,
        'feedback': feedback,
        'requirements_met': requirements_met
    }

def create_password_strength_visualizer(password: str) -> str:
    """Create a password strength visualizer"""
    if not password:
        return ""
    
    result = check_password_strength(password)
    
    # Create visual representation
    strength_html = f"""
    <div class="password-strength-container" style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">Password Strength</span>
            <span style="color: {result['strength_color']}; font-weight: 600;">
                {result['strength_text']} ({result['score']}/{result['max_score']})
            </span>
        </div>
        
        <div class="strength-bar-container" style="
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            <div class="strength-bar" style="
                width: {(result['score'] / result['max_score']) * 100}%;
                height: 100%;
                background: linear-gradient(90deg, {result['strength_color']}, {result['strength_color']}dd);
                border-radius: 10px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                    animation: shimmer 2s infinite;
                "></div>
            </div>
        </div>
        
        <div class="requirements-grid" style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.5rem;
            margin-bottom: 1rem;
        ">
    """
    
    requirement_labels = {
        'length': f'At least {PASSWORD_MIN_LENGTH} characters',
        'uppercase': 'Uppercase letters',
        'lowercase': 'Lowercase letters',
        'digit': 'Numbers',
        'special': 'Special characters',
        'no_common': 'No common patterns',
        'no_sequential': 'No sequential characters',
        'no_repeated': 'No repeated characters'
    }
    
    for req, label in requirement_labels.items():
        is_met = result['requirements_met'][req]
        icon = "✅" if is_met else "❌"
        color = "#27ae60" if is_met else "#e74c3c"
        
        strength_html += f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem;
            color: {color};
            font-size: 0.9rem;
        ">
            <span>{icon}</span>
            <span>{label}</span>
        </div>
        """
    
    strength_html += "</div>"
    
    if result['feedback']:
        strength_html += f"""
        <div class="feedback-section" style="
            background: rgba(231, 76, 60, 0.1);
            border-left: 4px solid #e74c3c;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        ">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Suggestions:</div>
            <ul style="margin: 0; padding-left: 1.5rem;">
        """
        
        for suggestion in result['feedback']:
            strength_html += f"<li>{suggestion}</li>"
        
        strength_html += "</ul></div>"
    
    strength_html += "</div>"
    
    return strength_html

def create_jwt_token(username: str, role: str, remember_me: bool = False) -> str:
    """Create a JWT token for authentication"""
    now = datetime.utcnow()
    expires = now + timedelta(days=SESSION_EXPIRY if remember_me else 1)
    
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": expires,
        "remember_me": remember_me,
        "jti": str(uuid.uuid4())  # Unique token ID
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    # Store token in cache for quick validation and potential revocation
    cache = get_cache()
    cache_key = f"auth:token:{username}:{payload['jti']}"
    cache.set(cache_key, {
        "username": username,
        "role": role, 
        "expires": expires.isoformat(),
        "created_at": now.isoformat(),
        "ip": None  # Could capture client IP if available
    }, ttl=SESSION_EXPIRY * 86400)
    
    return token

def verify_jwt_token(token: str) -> Tuple[bool, Dict[str, Any]]:
    """Verify JWT token and return payload if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check if token is in cache (for potential revocation)
        cache = get_cache()
        cache_key = f"auth:token:{payload['sub']}:{payload['jti']}"
        cached_token = cache.get(cache_key)
        
        if not cached_token:
            # Token not in cache, but it's valid by signature
            # This could happen if Redis was restarted or TTL expired
            # We'll still accept it if the JWT signature is valid
            pass
        
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, {"error": "Token expired"}
    except jwt.InvalidTokenError:
        return False, {"error": "Invalid token"}
    except Exception as e:
        return False, {"error": str(e)}

def verify_user(username: str, password: str) -> Tuple[bool, Optional[str], str]:
    """Verify user credentials against database"""
    db = get_db()
    
    if db is None:
        # Fallback for demo if DB is not available
        if username == "demo_user" and password == "demo_password":
            return True, "citizen", "Login successful"
        return False, None, "Invalid credentials"
    
    try:
        # Check for account lockout
        user = db.users.find_one({"username": username})
        if not user:
            # Don't reveal that username doesn't exist
            time.sleep(0.5)  # Add delay to prevent timing attacks
            return False, None, "Invalid credentials"
        
        # Check for account lockout
        security = user.get("security", {})
        locked_until = security.get("locked_until")
        if locked_until and datetime.fromisoformat(locked_until) > datetime.utcnow():
            return False, None, "Account is temporarily locked. Please try again later."
        
        # Verify password
        stored_hash = user.get("password_hash")
        stored_salt = user.get("password_salt")
        
        if not stored_hash or not stored_salt:
            return False, None, "Account error. Please contact support."
        
        is_valid = verify_password(stored_hash, stored_salt, password)
        
        if is_valid:
            # Reset failed attempts on successful login
            db.users.update_one(
                {"username": username},
                {
                    "$set": {
                        "security.failed_attempts": 0,
                        "security.locked_until": None,
                        "last_login": datetime.utcnow()
                    },
                    "$inc": {
                        "usage_stats.login_count": 1
                    }
                }
            )
            
            # Record successful login
            db.activity_log.insert_one({
                "user": username,
                "action": "login",
                "timestamp": datetime.utcnow(),
                "status": "success",
                "details": {
                    "method": "password"
                }
            })
            
            return True, user.get("role", "user"), "Login successful"
        else:
            # Increment failed attempts
            failed_attempts = security.get("failed_attempts", 0) + 1
            locked_until = None
            
            if failed_attempts >= MAX_LOGIN_ATTEMPTS:
                locked_until = (datetime.utcnow() + timedelta(minutes=LOCKOUT_TIME)).isoformat()
            
            db.users.update_one(
                {"username": username},
                {
                    "$set": {
                        "security.failed_attempts": failed_attempts,
                        "security.locked_until": locked_until
                    }
                }
            )
            
            # Record failed login
            db.activity_log.insert_one({
                "user": username,
                "action": "login",
                "timestamp": datetime.utcnow(),
                "status": "failed",
                "details": {
                    "method": "password",
                    "failed_attempts": failed_attempts,
                    "locked": locked_until is not None
                }
            })
            
            if locked_until:
                return False, None, f"Account locked. Try again after {LOCKOUT_TIME} minutes."
            
            return False, None, "Invalid credentials"
            
    except Exception as e:
        print(f"Authentication error: {e}")
        return False, None, "An error occurred during authentication"

def logout_user(cookies):
    """Log out user by clearing session and cookies"""
    username = st.session_state.get("username")
    
    if username:
        # Revoke JWT token if present
        token = st.session_state.get("jwt_token")
        if token:
            try:
                valid, payload = verify_jwt_token(token)
                if valid and "jti" in payload:
                    # Remove token from cache
                    cache = get_cache()
                    cache_key = f"auth:token:{username}:{payload['jti']}"
                    cache.delete(cache_key)
            except Exception:
                pass
        
        # Log the logout
        db = get_db()
        if db is not None:
            db.activity_log.insert_one({
                "user": username,
                "action": "logout",
                "timestamp": datetime.utcnow(),
                "status": "success"
            })
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Clear cookies
    if cookies:
        for key in list(cookies.keys()):
            del cookies[key]
        cookies.save()
    
    # Rerun to refresh the page
    st.rerun()
    st.stop()


def generate_password_reset_token(email: str) -> str:
    """Generate a secure token for password reset"""
    db = get_db()
    if db is None:
        # Fallback if DB not available
        return f"reset_token_{email}_{uuid.uuid4()}"
    
    try:
        # Find user by email
        user = db.users.find_one({"email": email})
        if not user:
            # Don't reveal that email doesn't exist, but don't generate a token
            return ""
        
        username = user.get("username")
        
        # Generate a secure token
        token_id = str(uuid.uuid4())
        expires = datetime.utcnow() + timedelta(hours=24)
        
        # Store token in database
        db.password_resets.insert_one({
            "token": token_id,
            "username": username,
            "email": email,
            "created_at": datetime.utcnow(),
            "expires": expires,
            "used": False
        })
        
        # Log the action
        db.activity_log.insert_one({
            "user": username,
            "action": "password_reset_request",
            "timestamp": datetime.utcnow(),
            "details": {
                "email": email,
                "expires": expires
            }
        })
        
        # Return token
        return token_id
    
    except Exception as e:
        print(f"Error generating reset token: {e}")
        return ""

# =============================================================================
# LOGIN AND AUTHENTICATION INTERFACES
# =============================================================================
def create_login_interface(cookies):
    """Create the main login interface"""
    # Apply theme
    apply_theme()
    
    # Initialize notification system
    NotificationSystem.init()
    NotificationSystem.render()
    
    # Main header with animation
    st.markdown(f"""
    <div class="enhanced-header">
        <h1 class="enhanced-title">{t('welcome_title')}</h1>
        <p class="enhanced-subtitle">{t('enhanced_welcome')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector
    create_language_selector()
    
    # Welcome animation
    welcome_col1, welcome_col2, welcome_col3 = st.columns([1, 2, 1])
    with welcome_col2:
        welcome_animation = load_lottie_animation("welcome")
        if welcome_animation:
            st_lottie(welcome_animation, height=200, key="main_welcome")
    
    # Main authentication interface
    auth_container = st.container()
    
    with auth_container:
        # Enhanced tabs with icons
        tab_labels = [
            "🔐 " + t("login_title"),
            "📝 " + t("register_header"),
            "🔄 " + t("forgot_password"),
            "ℹ️ About"
        ]
        
        auth_tabs = st.tabs(tab_labels)
        
        with auth_tabs[0]:
            # Login Tab
            create_login_form(cookies)
        
        with auth_tabs[1]:
            # Registration Tab
            create_registration_interface()
        
        with auth_tabs[2]:
            # Password Reset Tab
            create_password_reset_interface()
        
        with auth_tabs[3]:
            # About Tab
            create_about_interface()
    
    # Show progress tracker for returning users
    if st.session_state.get("show_progress", False):
        progress_tracker = create_progress_tracker()
        st.markdown(progress_tracker, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown(
        '''
        <div class="enhanced-container" style="text-align: center; margin-top: 3rem;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">🌟 Why Choose Civic Catalyst?</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
                    <h4>Lightning Fast</h4>
                    <p>Experience blazing-fast performance with our optimized infrastructure.</p>
                </div>
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔒</div>
                    <h4>Bank-Level Security</h4>
                    <p>Your data is protected with enterprise-grade security measures.</p>
                </div>
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                    <h4>AI-Powered</h4>
                    <p>Leverage artificial intelligence to enhance your civic engagement.</p>
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 2rem; margin-top: 2rem;">
                <p style="opacity: 0.7;">
                    © 2024 Civic Catalyst Platform | 
                    <a href="#" style="color: var(--accent-color);">Privacy Policy</a> | 
                    <a href="#" style="color: var(--accent-color);">Terms of Service</a>
                </p>
                <p style="opacity: 0.5; font-size: 0.9rem;">
                    Version 3.0.0 | Powered by Advanced AI & Cloud Technologies
                </p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def create_login_form(cookies):
    """Create a login form with validation and security features"""
    st.markdown("""
    <div class="enhanced-container">
        <h2 style="text-align: center; color: var(--primary-color); margin-bottom: 2rem;">
            🔐 Sign Into Your Account
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Login animation
    login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
    with login_col2:
        login_animation = load_lottie_animation("login")
        if login_animation:
            st_lottie(login_animation, height=150, key="login_form_animation")
    
    with st.form("login_form", clear_on_submit=False):
        # Username input
        st.markdown("### 👤 Username")
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            label_visibility="collapsed",
            help="Your unique username for the platform"
        )
        
        # Password input
        st.markdown("### 🔑 Password")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            label_visibility="collapsed",
            help="Your secure password"
        )
        
        # Login options
        st.markdown("### ⚙️ Login Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remember_me = st.checkbox(
                "🔒 Remember me",
                value=True,
                help=f"Keep you logged in for {SESSION_EXPIRY} days"
            )
        
        with col2:
            stay_signed_in = st.checkbox(
                "💾 Stay signed in",
                value=True,
                help="Maintain session across browser restarts"
            )
        
        # Advanced login options
        with st.expander("🔧 Advanced Options", expanded=False):
            # Voice feedback option
            enable_voice = st.checkbox(
                "🔊 Enable voice feedback",
                value=st.session_state.get("voice_feedback", False),
                help="Get audio confirmation of actions"
            )
            
            # Automatic theme detection
            auto_theme = st.checkbox(
                "🎨 Auto-detect theme preference",
                value=True,
                help="Automatically set theme based on time of day"
            )
            
            # Enhanced security mode
            secure_mode = st.checkbox(
                "🛡️ Enhanced security mode",
                value=False,
                help="Enable additional security measures"
            )
        
        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        
        login_button = st.form_submit_button(
            "🚀 Sign In to Civic Catalyst",
            type="primary",
            use_container_width=True
        )
        
        if login_button:
            # Validate inputs
            if not username or not password:
                NotificationSystem.add("Please enter both username and password", "error")
                return
            
            # Show loading with progress
            with st.spinner("🔐 Authenticating your credentials..."):
                # Simulate authentication process with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Verifying username...")
                progress_bar.progress(25)
                time.sleep(0.3)
                
                status_text.text("Checking password...")
                progress_bar.progress(50)
                time.sleep(0.3)
                
                status_text.text("Validating session...")
                progress_bar.progress(75)
                
                # Actual authentication
                valid, role, message = verify_user(username, password)
                
                progress_bar.progress(100)
                status_text.text("Authentication complete!")
                time.sleep(0.3)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if valid:
                    # Success handling
                    success_animation = load_lottie_animation("success")
                    if success_animation:
                        st_lottie(success_animation, height=120, key="login_success")
                    
                    # Store preferences
                    if enable_voice:
                        st.session_state["voice_feedback"] = True
                    
                    if auto_theme:
                        # Set theme based on time of day
                        current_hour = datetime.now().hour
                        if 6 <= current_hour < 18:
                            st.session_state["theme"] = "cosmic"
                        else:
                            st.session_state["theme"] = "neon"
                    
                    # Create JWT token
                    token = create_jwt_token(username, role, remember_me)
                    
                    if token:
                        # Store session data
                        st.session_state["jwt_token"] = token
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        st.session_state["secure_mode"] = secure_mode
                        
                        # Store in cookies
                        cookies["jwt_token"] = token
                        cookies["username"] = username
                        cookies["role"] = role
                        cookies["secure_mode"] = secure_mode
                        cookies.save()
                        
                        # Success notification
                        NotificationSystem.add(f"Welcome back, {username}! 🎉", "celebration")
                        
                        # Voice feedback if enabled
                        if enable_voice:
                            voice_system = create_voice_system()
                            audio_html = voice_system(
                                f"Welcome back, {username}!",
                                st.session_state.get("site_language", "en"),
                                "excited"
                            )
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Track successful login in Redis
                        cache = get_cache()
                        cache_key = f"stats:login:{username}:{datetime.now().strftime('%Y%m%d')}"
                        login_stats = cache.get(cache_key) or {
                            "count": 0,
                            "last_login": None,
                            "features_used": {}
                        }
                        
                        login_stats["count"] += 1
                        login_stats["last_login"] = datetime.now().isoformat()
                        login_stats["features_used"] = {
                            "remember_me": remember_me,
                            "stay_signed_in": stay_signed_in,
                            "voice_feedback": enable_voice,
                            "auto_theme": auto_theme,
                            "secure_mode": secure_mode
                        }
                        
                        cache.set(cache_key, login_stats, 86400)  # 1 day TTL
                        
                        time.sleep(1)
                        st.rerun()
                        st.stop()
                        return
                    else:
                        NotificationSystem.add("Error creating session. Please try again.", "error")
                else:
                    # Error handling
                    error_animation = load_lottie_animation("error")
                    if error_animation:
                        st_lottie(error_animation, height=100, key="login_error")
                    
                    NotificationSystem.add(message or "Invalid credentials", "error")
    
    # Quick access section
    st.markdown(
        '''
        <div class="enhanced-container" style="margin-top: 2rem;">
            <h3 style="color: var(--primary-color); text-align: center; margin-bottom: 1rem;">⚡ Quick Access</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div class="quick-access-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">👤</div>
                    <div style="font-weight: 600;">Demo User</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">Try the platform</div>
                </div>
                <div class="quick-access-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎭</div>
                    <div style="font-weight: 600;">Guest Access</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">Limited features</div>
                </div>
                <div class="quick-access-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">👑</div>
                    <div style="font-weight: 600;">Admin Demo</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">Full access demo</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Add Streamlit buttons for quick access functionality
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Demo User Access", key="demo_user_btn"):
            st.session_state["username"] = "demo_user"
            st.session_state["role"] = "citizen"
            st.rerun()
            st.stop()
    
    with col2:
        if st.button("Guest Access", key="guest_access_btn"):
            st.session_state["username"] = "guest"
            st.session_state["role"] = "guest"
            st.rerun()
            st.stop()
    
    with col3:
        if st.button("Admin Demo", key="admin_demo_btn"):
            st.session_state["username"] = "admin_demo"
            st.session_state["role"] = "admin"
            st.rerun()
            st.stop()

def create_registration_interface():
    """Create the registration interface with multi-step flow"""
    # Initialize the registration step if not already set
    if "registration_step" not in st.session_state:
        st.session_state["registration_step"] = 1
    
    # Header section
    st.markdown(
        """
        <div class="enhanced-container">
            <h2 style="text-align: center; color: var(--primary-color); margin-bottom: 2rem;">
                ✨ Join the Civic Catalyst Community
            </h2>
            <p style="text-align: center; opacity: 0.8; margin-bottom: 2rem;">
                Create your account in just a few steps and start making a difference in your community.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    step = st.session_state["registration_step"]
    
    # Determine colors based on the current step
    circle1_color = "var(--primary-color)" if step >= 1 else "rgba(255,255,255,0.2)"
    bar1_color    = "var(--primary-color)" if step >= 2 else "rgba(255,255,255,0.2)"
    circle2_color = "var(--primary-color)" if step >= 2 else "rgba(255,255,255,0.2)"
    bar2_color    = "var(--primary-color)" if step >= 3 else "rgba(255,255,255,0.2)"
    circle3_color = "var(--primary-color)" if step >= 3 else "rgba(255,255,255,0.2)"

    # Progress tracker
    progress_html = f"""
    <div class="registration-progress" style="margin: 2rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div class="step" style="
                background: {circle1_color};
                color: white;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                transition: var(--transition);
            ">
                1
            </div>
            <div style="
                flex: 1;
                height: 2px;
                background: {bar1_color};
                margin: 0 1rem;
            "></div>
            <div class="step" style="
                background: {circle2_color};
                color: white;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                transition: var(--transition);
            ">
                2
            </div>
            <div style="
                flex: 1;
                height: 2px;
                background: {bar2_color};
                margin: 0 1rem;
            "></div>
            <div class="step" style="
                background: {circle3_color};
                color: white;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                transition: var(--transition);
            ">
                3
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
            <span>Basic Info</span>
            <span>Security</span>
            <span>Preferences</span>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

    # Render the appropriate step form
    if step == 1:
        create_registration_step_1()
    elif step == 2:
        create_registration_step_2()
    else:
        create_registration_step_3()

def create_registration_step_1():
    """Create the first step of registration - basic information"""
    welcome_animation = load_lottie_animation("welcome")
    if welcome_animation:
        st_lottie(welcome_animation, height=150, key="registration_step1")
    
    st.markdown("### 📝 Basic Information")
    
    with st.form("registration_step_1"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input(
                "First Name *",
                placeholder="Enter your first name",
                help="Your given name"
            )
            
            username = st.text_input(
                "Username *",
                placeholder="Choose a unique username",
                help="3-20 characters, letters, numbers, and some symbols allowed"
            )
        
        with col2:
            last_name = st.text_input(
                "Last Name *",
                placeholder="Enter your last name",
                help="Your family name"
            )
            
            email = st.text_input(
                "Email Address *",
                placeholder="your.email@example.com",
                help="We'll use this to verify your account"
            )
        
        # Additional information
        st.markdown("### 📍 Location & Role")
        
        col3, col4 = st.columns(2)
        
        with col3:
            city = st.text_input(
                "City",
                placeholder="Your city",
                help="Optional: Helps us provide localized content"
            )
            
            role = st.selectbox(
                "Role *",
                ["citizen", "community_leader", "local_official", "researcher", "developer"],
                help="Select your primary role in civic engagement"
            )
        
        with col4:
            country = st.selectbox(
                "Country",
                ["Morocco", "France", "Spain", "United States", "Canada", "Other"],
                help="Your country of residence"
            )
            
            cin = st.text_input(
                "CIN/ID Number",
                placeholder="Optional identification number",
                help="Optional: For enhanced verification"
            )
        
        # Terms and conditions
        st.markdown("### 📋 Terms & Conditions")
        
        terms_accepted = st.checkbox(
            "I accept the Terms of Service and Privacy Policy *",
            help="Required to create an account"
        )
        
        newsletter = st.checkbox(
            "Subscribe to community updates and newsletter",
            value=True,
            help="Stay informed about platform updates and community events"
        )
        
        # Validate and proceed
        col5, col6, col7 = st.columns([1, 2, 1])
        
        with col6:
            if st.form_submit_button("Continue to Security Setup 🔐", use_container_width=True):
                # Validation
                errors = []
                
                if not first_name:
                    errors.append("First name is required")
                if not last_name:
                    errors.append("Last name is required")
                if not username:
                    errors.append("Username is required")
                elif not re.match(r"^[a-zA-Z0-9_.-]{3,20}$", username):
                    errors.append("Username must be 3-20 characters (letters, numbers, _, -, .)")
                if not email:
                    errors.append("Email is required")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    errors.append("Invalid email format")
                if not role:
                    errors.append("Role selection is required")
                if not terms_accepted:
                    errors.append("You must accept the Terms of Service")
                
                if errors:
                    for error in errors:
                        NotificationSystem.add(error, "error")
                else:
                    # Check if username or email already exists
                    db = get_db()
                    if db is not None:
                        existing_user = db.users.find_one({
                            "$or": [
                                {"username": username},
                                {"email": email}
                            ]
                        })
                        
                        if existing_user:
                            if existing_user.get("username") == username:
                                NotificationSystem.add("Username already exists", "error")
                            else:
                                NotificationSystem.add("Email already exists", "error")
                            return
                    
                    # Store step 1 data in cache
                    cache = get_cache()
                    cache_key = f"registration:{username}"
                    cache.set(cache_key, {
                        "first_name": first_name,
                        "last_name": last_name,
                        "username": username,
                        "email": email,
                        "city": city,
                        "country": country,
                        "role": role,
                        "cin": cin,
                        "newsletter": newsletter,
                        "created_at": datetime.now().isoformat()
                    }, 3600)  # 1 hour TTL
                    
                    # Also store in session state as backup
                    st.session_state["registration_data"] = {
                        "first_name": first_name,
                        "last_name": last_name,
                        "username": username,
                        "email": email,
                        "city": city,
                        "country": country,
                        "role": role,
                        "cin": cin,
                        "newsletter": newsletter
                    }
                    
                    st.session_state["registration_step"] = 2
                    NotificationSystem.add("Basic information saved! 🎉", "success")
                    st.rerun()
                    st.stop()

def create_registration_step_2():
    """Create the second step of registration - security setup"""
    security_animation = load_lottie_animation("security")
    if security_animation:
        st_lottie(security_animation, height=150, key="registration_step2")
    
    st.markdown("### 🔐 Security Setup")
    
    with st.form("registration_step_2"):
        # Password creation with strength checking
        st.markdown("#### Create Your Password")
        
        password = st.text_input(
            "Password *",
            type="password",
            placeholder="Create a strong password",
            help=f"Must be at least {PASSWORD_MIN_LENGTH} characters with mix of letters, numbers, and symbols"
        )
        
        # Real-time password strength visualization
        if password:
            strength_html = create_password_strength_visualizer(password)
            st.markdown(strength_html, unsafe_allow_html=True)
        
        confirm_password = st.text_input(
            "Confirm Password *",
            type="password",
            placeholder="Confirm your password",
            help="Must match the password above"
        )
        
        # Security questions (optional but recommended)
        st.markdown("#### Security Questions (Optional but Recommended)")
        
        security_question_1 = st.selectbox(
            "Security Question 1",
            [
                "What was the name of your first pet?",
                "What city were you born in?",
                "What was your childhood nickname?",
                "What is your mother's maiden name?",
                "What was the model of your first car?"
            ]
        )
        
        security_answer_1 = st.text_input(
            "Answer 1",
            placeholder="Your answer (case-sensitive)",
            help="Keep this answer exactly as you type it"
        )
        
        security_question_2 = st.selectbox(
            "Security Question 2",
            [
                "What was the name of your elementary school?",
                "What is your favorite book?",
                "What was your first job?",
                "What city did you meet your spouse?",
                "What is your favorite movie?"
            ]
        )
        
        security_answer_2 = st.text_input(
            "Answer 2",
            placeholder="Your answer (case-sensitive)",
            help="Choose a different question from the first one"
        )
        
        # Two-factor authentication setup
        st.markdown("#### Two-Factor Authentication")
        
        enable_2fa = st.checkbox(
            "Enable Two-Factor Authentication (Recommended)",
            help="Adds an extra layer of security to your account"
        )
        
        if enable_2fa:
            st.info("📱 Two-factor authentication will be configured after account creation.")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("← Back to Basic Info"):
                st.session_state["registration_step"] = 1
                st.rerun()
                st.stop()
        
        with col3:
            if st.form_submit_button("Continue to Preferences →"):
                # Validation
                errors = []
                
                if not password:
                    errors.append("Password is required")
                else:
                    password_check = check_password_strength(password)
                    if not password_check['is_valid']:
                        errors.append("Password does not meet security requirements")
                
                if not confirm_password:
                    errors.append("Password confirmation is required")
                elif password != confirm_password:
                    errors.append("Passwords do not match")
                
                if security_question_1 == security_question_2 and security_answer_1 and security_answer_2:
                    errors.append("Please choose different security questions")
                
                if errors:
                    for error in errors:
                        NotificationSystem.add(error, "error")
                else:
                    # Get step 1 data
                    registration_data = st.session_state.get("registration_data", {})
                    username = registration_data.get("username")
                    
                    if not username:
                        NotificationSystem.add("Missing registration data. Please start over.", "error")
                        st.session_state["registration_step"] = 1
                        st.rerun()
                        st.stop()
                        return
                    
                    # Update registration data in cache
                    cache = get_cache()
                    cache_key = f"registration:{username}"
                    
                    existing_data = cache.get(cache_key) or {}
                    existing_data.update({
                        "password": password,  # Will be hashed before storage in DB
                        "security_question_1": security_question_1,
                        "security_answer_1": security_answer_1,
                        "security_question_2": security_question_2,
                        "security_answer_2": security_answer_2,
                        "enable_2fa": enable_2fa
                    })
                    
                    cache.set(cache_key, existing_data, 3600)  # 1 hour TTL
                    
                    # Also update session state
                    registration_data.update({
                        "password": password,
                        "security_question_1": security_question_1,
                        "security_answer_1": security_answer_1,
                        "security_question_2": security_question_2,
                        "security_answer_2": security_answer_2,
                        "enable_2fa": enable_2fa
                    })
                    
                    st.session_state["registration_data"] = registration_data
                    st.session_state["registration_step"] = 3
                    NotificationSystem.add("Security setup completed! 🛡️", "success")
                    st.rerun()
                    st.stop()

def create_registration_step_3():
    """Create the third step of registration - preferences and completion"""
    celebration_animation = load_lottie_animation("celebration")
    if celebration_animation:
        st_lottie(celebration_animation, height=150, key="registration_step3")
    
    st.markdown("### ⚙️ Preferences & Completion")
    
    with st.form("registration_step_3"):
        # Language preferences
        st.markdown("#### 🌐 Language Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_language = st.selectbox(
                "Primary Language",
                ["en", "fr", "ar", "darija"],
                format_func=lambda x: {
                    "en": "🇬🇧 English",
                    "fr": "🇫🇷 Français", 
                    "ar": "🇲🇦 العربية",
                    "darija": "🇲🇦 الدارجة"
                }[x],
                index=0
            )
        
        with col2:
            secondary_language = st.selectbox(
                "Secondary Language (Optional)",
                ["none", "en", "fr", "ar", "darija"],
                format_func=lambda x: {
                    "none": "None",
                    "en": "🇬🇧 English",
                    "fr": "🇫🇷 Français",
                    "ar": "🇲🇦 العربية", 
                    "darija": "🇲🇦 الدارجة"
                }[x],
                index=0
            )
        
        # Theme preferences
        st.markdown("#### 🎨 Appearance Preferences")
        
        # Theme selector with previews
        selected_theme = st.selectbox(
            "Choose Your Theme",
            AVAILABLE_THEMES,
            format_func=lambda x: x.title(),
            index=0
        )
        
        # Theme preview
        theme_colors = get_theme_colors(selected_theme)
        theme_preview = f"""
        <div style="
            background: {theme_colors['background']};
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
            color: {theme_colors['text']};
            border: 2px solid {theme_colors['primary']};
        ">
            <h4 style="margin: 0; color: {theme_colors['primary']};">{selected_theme.title()} Theme Preview</h4>
            <p style="margin: 0.5rem 0;">This is how your interface will look</p>
        </div>
        """
        st.markdown(theme_preview, unsafe_allow_html=True)
        
        # Accessibility preferences
        st.markdown("#### ♿ Accessibility Preferences")
        
        col3, col4 = st.columns(2)
        
        with col3:
            enable_voice_feedback = st.checkbox(
                "🔊 Enable voice feedback",
                help="Get audio confirmations for actions"
            )
            
            high_contrast = st.checkbox(
                "🔆 High contrast mode",
                help="Enhanced visibility for better readability"
            )
        
        with col4:
            large_text = st.checkbox(
                "📝 Large text mode",
                help="Increase text size for better readability"
            )
            
            reduce_animations = st.checkbox(
                "⚡ Reduce animations",
                help="Minimize motion for better performance or accessibility"
            )
        
        # Notification preferences
        st.markdown("#### 🔔 Notification Preferences")
        
        col5, col6 = st.columns(2)
        
        with col5:
            email_notifications = st.checkbox(
                "📧 Email notifications",
                value=True,
                help="Receive important updates via email"
            )
            
            community_updates = st.checkbox(
                "🏘️ Community updates",
                value=True,
                help="Get notified about local community activities"
            )
        
        with col6:
            system_announcements = st.checkbox(
                "📢 System announcements",
                value=True,
                help="Important platform updates and maintenance notices"
            )
            
            digest_frequency = st.selectbox(
                "Email digest frequency",
                ["daily", "weekly", "monthly", "never"],
                index=1
            )
        
        # Final completion
        st.markdown("---")
        st.markdown("### 🎉 Ready to Join?")
        
        st.info("""
        **You're almost done!** By clicking "Create Account", you'll:
        
        ✅ Join a community of engaged citizens  
        ✅ Access powerful civic engagement tools  
        ✅ Contribute to positive community change  
        ✅ Connect with like-minded individuals  
        """)
        
        # Navigation and completion buttons
        col7, col8, col9 = st.columns([1, 2, 1])
        
        with col7:
            if st.form_submit_button("← Back to Security"):
                st.session_state["registration_step"] = 2
                st.rerun()
                st.stop()
        
        with col8:
            create_account_button = st.form_submit_button(
                "🚀 Create My Account",
                type="primary",
                use_container_width=True
            )
        
        if create_account_button:
            # Final validation and account creation
            registration_data = st.session_state.get("registration_data", {})
            username = registration_data.get("username")
            
            if not username:
                NotificationSystem.add("Registration data missing. Please start over.", "error")
                st.session_state["registration_step"] = 1
                st.rerun()
                st.stop()
                return
            
            # Add step 3 preferences
            registration_data.update({
                "primary_language": primary_language,
                "secondary_language": secondary_language if secondary_language != "none" else None,
                "theme": selected_theme,
                "voice_feedback": enable_voice_feedback,
                "high_contrast": high_contrast,
                "large_text": large_text,
                "reduce_animations": reduce_animations,
                "email_notifications": email_notifications,
                "community_updates": community_updates,
                "system_announcements": system_announcements,
                "digest_frequency": digest_frequency
            })
            
            # Update cache with final data
            cache = get_cache()
            cache_key = f"registration:{username}"
            cache.set(cache_key, registration_data, 3600)
            
            # Show loading spinner
            with st.spinner("🔄 Creating your account..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Validating information...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                status_text.text("Creating user profile...")
                progress_bar.progress(40)
                time.sleep(0.3)
                
                status_text.text("Setting up preferences...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                status_text.text("Configuring security...")
                progress_bar.progress(80)
                time.sleep(0.3)
                
                # Actually create the user
                success = create_user(registration_data)
                
                progress_bar.progress(100)
                status_text.text("Account created successfully!")
                time.sleep(0.3)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                if success:
                    # Success celebration
                    rocket_animation = load_lottie_animation("rocket")
                    if rocket_animation:
                        st_lottie(rocket_animation, height=200, key="registration_success")
                    
                    NotificationSystem.add("🎉 Welcome to Civic Catalyst! Your account has been created successfully.", "celebration")
                    
                    # Voice congratulations if enabled
                    if enable_voice_feedback:
                        voice_system = create_voice_system()
                        audio_html = voice_system(
                            f"Congratulations {registration_data['first_name']}! Welcome to Civic Catalyst!",
                            primary_language,
                            "excited"
                        )
                        if audio_html:
                            st.markdown(audio_html, unsafe_allow_html=True)
                    
                    # Clear registration data from cache
                    cache.delete(cache_key)
                    
                    # Clear registration data from session state
                    del st.session_state["registration_data"]
                    st.session_state["registration_step"] = 1
                    
                    # Set theme and language preferences
                    st.session_state["theme"] = selected_theme
                    st.session_state["site_language"] = primary_language
                    st.session_state["voice_feedback"] = enable_voice_feedback
                    
                    # Show welcome message
                    st.success("✅ Registration completed! You can now log in with your credentials.")
                    
                    time.sleep(2)
                    st.rerun()
                    st.stop()
                else:
                    NotificationSystem.add("❌ Registration failed. Please try again.", "error")

def create_user(registration_data: Dict[str, Any]) -> bool:
    """Create a new user in the database"""
    db = get_db()
    if db is None:
        # Simulate success if DB is not available (demo mode)
        return True
    
    try:
        # Check if username or email already exists
        existing_user = db.users.find_one({
            "$or": [
                {"username": registration_data["username"]},
                {"email": registration_data["email"]}
            ]
        })
        
        if existing_user:
            if existing_user.get("username") == registration_data["username"]:
                NotificationSystem.add("Username already exists", "error")
            else:
                NotificationSystem.add("Email already exists", "error")
            return False
        
        # Hash password
        password_hash, salt = hash_password(registration_data["password"])
        
        # Create user document
        new_user = {
            "username": registration_data["username"],
            "email": registration_data["email"],
            "password_hash": password_hash,
            "password_salt": salt,
            "role": registration_data["role"],
            
            # Personal information
            "personal_info": {
                "first_name": registration_data["first_name"],
                "last_name": registration_data["last_name"],
                "city": registration_data.get("city"),
                "country": registration_data.get("country"),
                "cin": registration_data.get("cin")
            },
            
            # Security settings
            "security": {
                "failed_attempts": 0,
                "locked_until": None,
                "2fa_enabled": registration_data.get("enable_2fa", False),
                "security_questions": {
                    "question_1": registration_data.get("security_question_1"),
                    "answer_1": registration_data.get("security_answer_1"),
                    "question_2": registration_data.get("security_question_2"),
                    "answer_2": registration_data.get("security_answer_2")
                },
                "password_last_changed": datetime.utcnow(),
                "password_expires": datetime.utcnow() + timedelta(days=90)
            },
            
            # Preferences
            "preferences": {
                "language": {
                    "primary": registration_data.get("primary_language", "en"),
                    "secondary": registration_data.get("secondary_language")
                },
                "theme": registration_data.get("theme", DEFAULT_THEME),
                "accessibility": {
                    "voice_feedback": registration_data.get("voice_feedback", False),
                    "high_contrast": registration_data.get("high_contrast", False),
                    "large_text": registration_data.get("large_text", False),
                    "reduce_animations": registration_data.get("reduce_animations", False)
                },
                "notifications": {
                    "email": registration_data.get("email_notifications", True),
                    "community_updates": registration_data.get("community_updates", True),
                    "system_announcements": registration_data.get("system_announcements", True),
                    "digest_frequency": registration_data.get("digest_frequency", "weekly")
                }
            },
            
            # Account metadata
            "account_meta": {
                "created_at": datetime.utcnow(),
                "email_verified": False,
                "newsletter_subscribed": registration_data.get("newsletter", False),
                "onboarding_completed": False,
                "terms_accepted_at": datetime.utcnow(),
                "registration_ip": None,  # Could capture if available
                "registration_source": "web_platform"
            },
            
            # Usage statistics
            "usage_stats": {
                "login_count": 0,
                "last_login": None,
                "total_session_time": 0,
                "features_used": [],
                "achievements": []
            }
        }
        
        # Insert user
        result = db.users.insert_one(new_user)
        
        if result.inserted_id:
            # Log registration
            db.activity_log.insert_one({
                "user": registration_data["username"],
                "action": "user_registration",
                "timestamp": datetime.utcnow(),
                "details": {
                    "registration_method": "form",
                    "role": registration_data["role"],
                    "features_enabled": {
                        "2fa": registration_data.get("enable_2fa", False),
                        "voice_feedback": registration_data.get("voice_feedback", False)
                    }
                }
            })
            
            # Create welcome notification
            db.notifications.insert_one({
                "user": registration_data["username"],
                "type": "welcome",
                "title": "Welcome to Civic Catalyst! 🎉",
                "message": f"Hello {registration_data['first_name']}! Your account has been created successfully. Explore our features and start making a difference in your community.",
                "created_at": datetime.utcnow(),
                "read": False,
                "priority": "high"
            })
            
            # Create cache entry for user preferences
            cache = get_cache()
            cache_key = f"user:preferences:{registration_data['username']}"
            
            cache.set(cache_key, {
                "theme": registration_data.get("theme", DEFAULT_THEME),
                "language": registration_data.get("primary_language", "en"),
                "voice_feedback": registration_data.get("voice_feedback", False),
                "high_contrast": registration_data.get("high_contrast", False),
                "large_text": registration_data.get("large_text", False),
                "reduce_animations": registration_data.get("reduce_animations", False)
            }, CACHE_TTL["user_profile"])
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Registration error: {e}")
        return False

def create_password_reset_interface():
    """Create password reset interface with multiple recovery options"""
    st.markdown("""
    <div class="enhanced-container">
        <h2 style="text-align: center; color: var(--primary-color); margin-bottom: 2rem;">
            🔄 Account Recovery
        </h2>
        <p style="text-align: center; opacity: 0.8; margin-bottom: 2rem;">
            Choose your preferred method to recover your account access.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Password reset animation
    reset_animation = load_lottie_animation("security")
    if reset_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(reset_animation, height=150, key="password_reset_animation")
    
    # Recovery method selector
    recovery_tabs = st.tabs(["📧 Email Reset", "❓ Security Questions", "📱 SMS Recovery"])
    
    with recovery_tabs[0]:
        create_email_reset_form()
    
    with recovery_tabs[1]:
        create_security_questions_form()
    
    with recovery_tabs[2]:
        create_sms_recovery_form()

def create_email_reset_form():
    """Create email-based password reset form"""
    st.markdown("### 📧 Reset via Email")
    st.info("Enter your email address and we'll send you instructions to reset your password.")
    
    with st.form("email_reset_form"):
        email = st.text_input(
            "Email Address",
            placeholder="your.email@example.com",
            help="The email address associated with your account"
        )
        
        # Captcha simulation (in real implementation, use actual captcha)
        st.markdown("#### 🤖 Security Verification")
        import random
        captcha_challenge = random.randint(1000, 9999)
        st.info(f"Please enter the following code: **{captcha_challenge}**")
        
        captcha_input = st.text_input(
            "Verification Code",
            placeholder="Enter the code above",
            help="This helps us verify you're not a robot"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("🔄 Send Reset Email", use_container_width=True):
                if not email:
                    NotificationSystem.add("Please enter your email address", "error")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    NotificationSystem.add("Please enter a valid email address", "error")
                elif captcha_input != str(captcha_challenge):
                    NotificationSystem.add("Incorrect verification code", "error")
                else:
                    with st.spinner("Sending reset email..."):
                        # Cache the email for rate limiting
                        cache = get_cache()
                        cache_key = f"reset_request:{email}"
                        
                        # Check for rate limiting
                        recent_request = cache.get(cache_key)
                        if recent_request:
                            NotificationSystem.add("A reset email was recently sent. Please check your inbox or try again later.", "warning")
                            return
                        
                        # Generate reset token
                        reset_token = generate_password_reset_token(email)
                        
                        if reset_token:
                            # Set rate limiting
                            cache.set(cache_key, {"timestamp": datetime.now().isoformat()}, 300)  # 5 min cooldown
                            
                            st.success("✅ Reset instructions sent! Check your email inbox.")
                            st.info(f"**Demo Token:** {reset_token[:10]}...")
                            NotificationSystem.add("Password reset email sent successfully", "success")
                        else:
                            st.error("❌ Failed to send reset email. Please try again.")
        
        with col2:
            if st.form_submit_button("↩️ Back to Login", use_container_width=True):
                st.rerun()
                st.stop()

def create_security_questions_form():
    """Create security questions-based recovery form"""
    st.markdown("### ❓ Reset via Security Questions")
    st.info("Answer your security questions to reset your password.")
    
    with st.form("security_questions_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            help="Your account username"
        )
        
        if username:
            # Try to get security questions from database
            db = get_db()
            user_data = None
            
            if db is not None:
                user_data = db.users.find_one({"username": username})
            
            if user_data:
                security = user_data.get("security", {})
                security_questions = security.get("security_questions", {})
                
                if security_questions:
                    st.markdown("#### Security Questions")
                    
                    question_1 = security_questions.get("question_1", "What was the name of your first pet?")
                    st.text(f"Question 1: {question_1}")
                    
                    answer_1 = st.text_input(
                        "Answer 1",
                        placeholder="Your answer (case-sensitive)",
                        help="Enter your answer exactly as you set it up",
                        type="password"
                    )
                    
                    question_2 = security_questions.get("question_2", "What was the name of your elementary school?")
                    st.text(f"Question 2: {question_2}")
                    
                    answer_2 = st.text_input(
                        "Answer 2",
                        placeholder="Your answer (case-sensitive)",
                        help="Enter your answer exactly as you set it up",
                        type="password"
                    )
                else:
                    st.warning("No security questions found for this account. Try another recovery method.")
            else:
                # For demo, show simulated questions
                st.markdown("#### Security Questions")
                
                question_1 = st.text("Question 1: What was the name of your first pet?")
                answer_1 = st.text_input(
                    "Answer 1",
                    placeholder="Your answer (case-sensitive)",
                    help="Enter your answer exactly as you set it up",
                    type="password"
                )
                
                question_2 = st.text("Question 2: What was the name of your elementary school?")
                answer_2 = st.text_input(
                    "Answer 2",
                    placeholder="Your answer (case-sensitive)",
                    help="Enter your answer exactly as you set it up",
                    type="password"
                )
        
        if st.form_submit_button("🔓 Verify & Reset Password", use_container_width=True):
            if not username:
                NotificationSystem.add("Please enter your username", "error")
            elif not answer_1 or not answer_2:
                NotificationSystem.add("Please answer both security questions", "error")
            else:
                with st.spinner("Verifying answers..."):
                    # Verify answers against database
                    verified = False
                    
                    if db is not None and user_data:
                        security = user_data.get("security", {})
                        security_questions = security.get("security_questions", {})
                        
                        if (security_questions.get("answer_1") == answer_1 and 
                            security_questions.get("answer_2") == answer_2):
                            verified = True
                    else:
                        # Demo mode
                        verified = (answer_1 == "Fluffy" and answer_2 == "Lincoln Elementary")
                    
                    if verified:
                        # Generate reset token
                        reset_token = str(uuid.uuid4())
                        
                        # Store token in database or cache
                        if db is not None:
                            db.password_resets.insert_one({
                                "token": reset_token,
                                "username": username,
                                "created_at": datetime.utcnow(),
                                "expires": datetime.utcnow() + timedelta(hours=1),
                                "used": False,
                                "method": "security_questions"
                            })
                        else:
                            # Store in cache for demo
                            cache = get_cache()
                            cache.set(f"reset_token:{reset_token}", {
                                "username": username,
                                "expires": (datetime.utcnow() + timedelta(hours=1)).isoformat()
                            }, 3600)
                        
                        st.success("✅ Security questions verified! You can now reset your password.")
                        
                        # Show reset password form
                        st.markdown("### 🔐 Reset Your Password")
                        
                        with st.form("new_password_form"):
                            new_password = st.text_input(
                                "New Password",
                                placeholder="Enter new password",
                                type="password",
                                help=f"Must be at least {PASSWORD_MIN_LENGTH} characters with mix of letters, numbers, and symbols"
                            )
                            
                            # Show password strength
                            if new_password:
                                strength_html = create_password_strength_visualizer(new_password)
                                st.markdown(strength_html, unsafe_allow_html=True)
                            
                            confirm_password = st.text_input(
                                "Confirm New Password",
                                placeholder="Confirm new password",
                                type="password",
                                help="Must match the password above"
                            )
                            
                            if st.form_submit_button("🔐 Reset Password", use_container_width=True):
                                if not new_password:
                                    NotificationSystem.add("Please enter a new password", "error")
                                elif new_password != confirm_password:
                                    NotificationSystem.add("Passwords do not match", "error")
                                else:
                                    password_check = check_password_strength(new_password)
                                    if not password_check['is_valid']:
                                        NotificationSystem.add("Password does not meet security requirements", "error")
                                    else:
                                        # Update password in database
                                        if db is not None:
                                            password_hash, salt = hash_password(new_password)
                                            
                                            db.users.update_one(
                                                {"username": username},
                                                {
                                                    "$set": {
                                                        "password_hash": password_hash,
                                                        "password_salt": salt,
                                                        "security.password_last_changed": datetime.utcnow(),
                                                        "security.password_expires": datetime.utcnow() + timedelta(days=90),
                                                        "security.failed_attempts": 0,
                                                        "security.locked_until": None
                                                    }
                                                }
                                            )
                                            
                                            # Mark reset token as used
                                            db.password_resets.update_one(
                                                {"token": reset_token},
                                                {"$set": {"used": True}}
                                            )
                                            
                                            # Log the action
                                            db.activity_log.insert_one({
                                                "user": username,
                                                "action": "password_reset",
                                                "timestamp": datetime.utcnow(),
                                                "details": {"method": "security_questions"}
                                            })
                                        
                                        st.success("✅ Password reset successful! You can now log in with your new password.")
                                        NotificationSystem.add("Password has been reset successfully", "success")
                    else:
                        NotificationSystem.add("Incorrect security answers", "error")

def create_sms_recovery_form():
    """Create SMS-based recovery form"""
    st.markdown("### 📱 Reset via SMS")
    st.info("Enter your phone number to receive a verification code.")
    
    with st.form("sms_recovery_form"):
        phone_number = st.text_input(
            "Phone Number",
            placeholder="+1 (555) 123-4567",
            help="The phone number associated with your account"
        )
        
        if st.form_submit_button("📲 Send Verification Code", use_container_width=True):
            if not phone_number:
                NotificationSystem.add("Please enter your phone number", "error")
            else:
                with st.spinner("Sending SMS code..."):
                    # Cache the phone for rate limiting
                    cache = get_cache()
                    cache_key = f"sms_reset:{phone_number}"
                    
                    # Check for rate limiting
                    recent_request = cache.get(cache_key)
                    if recent_request:
                        NotificationSystem.add("A verification code was recently sent. Please wait before requesting another.", "warning")
                        return
                    
                    # Generate a verification code
                    verification_code = random.randint(100000, 999999)
                    
                    # Store code in cache
                    cache.set(cache_key, {"code": verification_code, "attempts": 0}, 300)  # 5 min expiry
                    
                    # In production, send the SMS here
                    # ...
                    
                    st.success(f"✅ Code sent! For demo purposes, your code is: **{verification_code}**")
                    
                    # Show verification input
                    st.markdown("#### Enter Verification Code")
                    
                    with st.form("sms_verification_form"):
                        entered_code = st.text_input(
                            "6-Digit Code",
                            placeholder="123456",
                            max_chars=6
                        )
                        
                        if st.form_submit_button("🔐 Verify Code"):
                            if not entered_code:
                                NotificationSystem.add("Please enter the verification code", "error")
                            else:
                                # Get stored code from cache
                                stored_data = cache.get(cache_key)
                                if not stored_data:
                                    NotificationSystem.add("Verification code expired. Please request a new one.", "error")
                                    return
                                
                                stored_code = stored_data.get("code")
                                attempts = stored_data.get("attempts", 0)
                                
                                if attempts >= 3:
                                    cache.delete(cache_key)
                                    NotificationSystem.add("Too many failed attempts. Please request a new code.", "error")
                                    return
                                
                                if str(entered_code) == str(stored_code):
                                    st.success("✅ Phone number verified! You can now reset your password.")
                                    NotificationSystem.add("SMS verification successful", "success")
                                    
                                    # Show password reset form (similar to security questions form)
                                    # ...
                                else:
                                    # Increment failed attempts
                                    stored_data["attempts"] = attempts + 1
                                    cache.set(cache_key, stored_data, 300)
                                    
                                    NotificationSystem.add(f"Invalid verification code. {3 - stored_data['attempts']} attempts remaining.", "error")

def create_about_interface():
    """Create an informative about interface with platform details"""
    st.markdown("""
    <div class="enhanced-container">
        <h2 style="text-align: center; color: var(--primary-color); margin-bottom: 2rem;">
            🌟 About Civic Catalyst
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # About animation
    about_animation = load_lottie_animation("morocco")
    if about_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(about_animation, height=200, key="about_animation")
    
    # Platform information tabs
    info_tabs = st.tabs(["🎯 Mission", "✨ Features", "🏆 Achievements", "📊 Statistics"])
    
    with info_tabs[0]:
        st.markdown("""
        ### Our Mission
        
        Civic Catalyst is dedicated to revolutionizing civic engagement through cutting-edge technology 
        and innovative approaches to community participation. We believe that every citizen has the 
        power to make a meaningful difference in their community.
        
        #### 🎯 Our Goals
        - **Empower Citizens**: Provide tools that make civic participation accessible and impactful
        - **Bridge Communities**: Connect citizens with local government and community organizations
        - **Drive Innovation**: Leverage AI and modern technology for social good
        - **Foster Transparency**: Promote open governance and accountability
        - **Enable Collaboration**: Create platforms for constructive community dialogue
        
        #### 🌍 Global Impact
        Since our launch, we've facilitated thousands of civic initiatives across multiple countries,
        helping communities tackle challenges from local infrastructure to environmental conservation.
        """)
    
    with info_tabs[1]:
        st.markdown("### 🚀 Platform Features")
        
        features_data = [
            {
                "icon": "🤖",
                "title": "AI-Powered Insights",
                "description": "Advanced analytics to understand community trends and needs",
                "status": "active"
            },
            {
                "icon": "📊", 
                "title": "Real-time Analytics",
                "description": "Live dashboards showing community engagement metrics",
                "status": "active"
            },
            {
                "icon": "🔊",
                "title": "Voice Integration",
                "description": "Multi-language voice feedback and accessibility features",
                "status": "active"
            },
            {
                "icon": "🌐",
                "title": "Multi-language Support",
                "description": "Interface available in English, French, Arabic, and Darija",
                "status": "active"
            },
            {
                "icon": "🔒",
                "title": "Advanced Security",
                "description": "Bank-level encryption and security measures",
                "status": "active"
            },
            {
                "icon": "📱",
                "title": "Mobile Optimization",
                "description": "Fully responsive design for all devices",
                "status": "active"
            },
            {
                "icon": "🤝",
                "title": "Collaboration Tools",
                "description": "Real-time collaboration and project management",
                "status": "beta"
            },
            {
                "icon": "🔗",
                "title": "Blockchain Verification",
                "description": "Immutable record keeping for transparency",
                "status": "coming_soon"
            }
        ]
        
        # Create feature grid
        cols = st.columns(2)
        
        for i, feature in enumerate(features_data):
            with cols[i % 2]:
                status_color = {
                    "active": "#27ae60",
                    "beta": "#f39c12", 
                    "coming_soon": "#e74c3c"
                }
                
                status_text = {
                    "active": "✅ Active",
                    "beta": "🧪 Beta",
                    "coming_soon": "🔜 Coming Soon"
                }
                
                st.markdown(f"""
                <div class="feature-showcase" style="
                    background: var(--surface-color);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-left: 4px solid {status_color[feature['status']]};
                    transition: var(--transition);
                ">
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                        <span style="font-size: 2rem;">{feature['icon']}</span>
                        <div>
                            <h4 style="margin: 0; color: var(--primary-color);">{feature['title']}</h4>
                            <span style="
                                background: {status_color[feature['status']]};
                                color: white;
                                padding: 0.2rem 0.5rem;
                                border-radius: 12px;
                                font-size: 0.8rem;
                            ">{status_text[feature['status']]}</span>
                        </div>
                    </div>
                    <p style="margin: 0; opacity: 0.8;">{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with info_tabs[2]:
        st.markdown("### 🏆 Platform Achievements")
        
        achievements_data = [
            {"metric": "Active Users", "value": "25,000+", "icon": "👥"},
            {"metric": "Projects Completed", "value": "1,200+", "icon": "✅"},
            {"metric": "Communities Served", "value": "150+", "icon": "🏘️"},
            {"metric": "Countries", "value": "12", "icon": "🌍"},
            {"metric": "Languages Supported", "value": "4", "icon": "🗣️"},
            {"metric": "Satisfaction Rate", "value": "94.5%", "icon": "⭐"}
        ]
        
        # Create achievements grid
        cols = st.columns(3)
        
        for i, achievement in enumerate(achievements_data):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="achievement-card" style="
                    background: var(--surface-color);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    margin-bottom: 1rem;
                    border: 2px solid var(--primary-color);
                    transition: var(--transition);
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{achievement['icon']}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: var(--primary-color); margin-bottom: 0.5rem;">
                        {achievement['value']}
                    </div>
                    <div style="font-weight: 600; opacity: 0.8;">{achievement['metric']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with info_tabs[3]:
        st.markdown("### 📊 Live Platform Statistics")
        
        # Use cache for statistics
        cache = get_cache()
        cache_key = "stats:platform:summary"
        
        platform_stats = cache.get(cache_key)
        if not platform_stats:
            # Generate some sample statistics if not in cache
            platform_stats = {
                "users": random.randint(10000, 30000),
                "daily_active": random.randint(1000, 5000),
                "projects": random.randint(1000, 3000),
                "avg_engagement": random.randint(25, 90),
                "uptime": random.uniform(99.7, 99.99),
                "response_time": random.randint(50, 150)
            }
            
            # Cache the stats
            cache.set(cache_key, platform_stats, 300)  # 5 minute TTL
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", f"{platform_stats['users']:,}", "+5.2%")
        
        with col2:
            st.metric("Daily Active Users", f"{platform_stats['daily_active']:,}", "+12.7%")
        
        with col3:
            st.metric("Active Projects", f"{platform_stats['projects']:,}", "+3.1%")
        
        # System health metrics
        st.markdown("#### 🖥️ System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Platform Uptime", f"{platform_stats['uptime']:.2f}%", "+0.03%")
        
        with col2:
            st.metric("Avg Response Time", f"{platform_stats['response_time']} ms", "-5 ms")
        
        # Create a sample chart
        # Generate sample daily active user data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        values = [random.randint(800, 5000) for _ in range(30)]
        
        # Create the chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Daily Active Users',
            line=dict(color=get_theme_colors()["primary"], width=3),
            fill='tozeroy',
            fillcolor=f'rgba({int(get_theme_colors()["primary"][1:3], 16)}, {int(get_theme_colors()["primary"][3:5], 16)}, {int(get_theme_colors()["primary"][5:7], 16)}, 0.1)'
        ))
        
        fig.update_layout(
            title='Daily Active Users (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Users',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='var(--text-color)'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_authenticated_interface(username: str, role: str, cookies):
    """Create the main authenticated interface"""
    # Apply theme
    apply_theme()
    
    # Initialize notification system
    NotificationSystem.init()
    NotificationSystem.render()
    
    # Enhanced header with user information
    create_user_header(username, role)
    
    # Main navigation
    main_tabs = st.tabs([
        "🏠 Dashboard",
        "👤 Profile", 
        "🔧 Tools",
        "📊 Analytics",
        "🌐 Community",
        "⚙️ Settings"
    ])
    
    with main_tabs[0]:
        create_dashboard(username, role)
    
    with main_tabs[1]:
        create_user_profile(username, role)
    
    with main_tabs[2]:
        create_tools_interface(username, role)
    
    with main_tabs[3]:
        create_analytics_dashboard(username, role)
    
    with main_tabs[4]:
        create_community_interface(username, role)
    
    with main_tabs[5]:
        create_settings_interface(username, role)
    
    # Enhanced sidebar with quick actions
    create_sidebar(username, role, cookies)

def create_user_header(username: str, role: str):
    """Create user header with info and quick stats"""
    # Get user data from cache or database
    cache = get_cache()
    cache_key = f"user:profile:{username}"
    
    user_data = cache.get(cache_key)
    if not user_data:
        db = get_db()
        if db is not None:
            db_user = db.users.find_one({"username": username})
            if db_user:
                personal_info = db_user.get("personal_info", {})
                user_data = {
                    "first_name": personal_info.get("first_name", ""),
                    "last_name": personal_info.get("last_name", ""),
                    "role": db_user.get("role", role),
                    "last_login": db_user.get("last_login", datetime.now().isoformat()),
                    "login_streak": random.randint(1, 10)  # Sample data
                }
                # Cache the user data
                cache.set(cache_key, user_data, CACHE_TTL["user_profile"])
        
        if not user_data:
            # Fallback data
            user_data = {
                "first_name": "Demo",
                "last_name": "User",
                "role": role,
                "last_login": datetime.now().isoformat(),
                "login_streak": 7
            }
    
    full_name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip() or username
    
    # Create header HTML
    svg_pattern = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.5"/><circle cx="80" cy="80" r="1" fill="white" opacity="0.3"/><circle cx="40" cy="60" r="1" fill="white" opacity="0.4"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>'''
    encoded_svg = base64.b64encode(svg_pattern.encode()).decode()
    
    st.markdown(f"""
    <div class="enhanced-user-header" style="
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.1;">
            <div style="
                background: url('data:image/svg+xml;base64,{encoded_svg}');
                width: 100%;
                height: 100%;
            "></div>
        </div>
        
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div class="floating-element">
                    <div style="
                        width: 100px;
                        height: 100px;
                        border-radius: 50%;
                        background: rgba(255, 255, 255, 0.2);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 3rem;
                        margin-bottom: 1rem;
                        backdrop-filter: blur(10px);
                        border: 3px solid rgba(255, 255, 255, 0.3);
                    ">
                        👋
                    </div>
                </div>
                
                <div style="text-align: left;">
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800;">
                        {t('welcome_back')}, {full_name}!
                    </h1>
                    <p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">
                        {user_data.get('role', role).replace('_', ' ').title()} • Last seen: Today
                    </p>
                    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                        <span style="
                            background: rgba(255, 255, 255, 0.2);
                            padding: 0.3rem 0.8rem;
                            border-radius: 20px;
                            font-size: 0.9rem;
                            backdrop-filter: blur(10px);
                        ">🔥 Active Streak: {user_data.get('login_streak', 7)} days</span>
                        <span style="
                            background: rgba(255, 255, 255, 0.2);
                            padding: 0.3rem 0.8rem;
                            border-radius: 20px;
                            font-size: 0.9rem;
                            backdrop-filter: blur(10px);
                        ">⭐ Level {user_data.get('login_streak', 7) // 2 + 1} Citizen</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_dashboard(username: str, role: str):
    """Create dashboard with personalized content"""
    st.markdown("### 🏠 Your Personal Dashboard")
    
    # Dashboard implementation goes here
    st.info("Dashboard content would be implemented here")
    
    # Show progress tracker
    progress_tracker = create_progress_tracker()
    st.markdown(progress_tracker, unsafe_allow_html=True)

# Placeholders for remaining interface functions
def create_user_profile(username: str, role: str):
    """Create user profile interface"""
    st.markdown("### 👤 User Profile")
    st.info(f"Profile page for {username} ({role})")

def create_tools_interface(username: str, role: str):
    """Create tools interface"""
    st.markdown("### 🔧 Tools")
    st.info("Tools interface would go here")

def create_analytics_dashboard(username: str, role: str):
    """Create analytics dashboard"""
    st.markdown("### 📊 Analytics")
    st.info("Analytics dashboard would go here")

def create_community_interface(username: str, role: str):
    """Create community interface"""
    st.markdown("### 🌐 Community")
    st.info("Community interface would go here")

def create_settings_interface(username: str, role: str):
    """Create settings interface"""
    st.markdown("### ⚙️ Settings")
    st.info("Settings interface would go here")

def create_sidebar(username: str, role: str, cookies):
    """Create sidebar with quick actions"""
    with st.sidebar:
        # User avatar and info
        st.markdown(f"""
        <div style="
            background: var(--surface-color);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
            backdrop-filter: blur(20px);
        ">
            <div style="
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: var(--primary-color);
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.5rem;
                font-size: 1.5rem;
                color: white;
            ">
                {username[0].upper()}
            </div>
            <div style="font-weight: 600; margin-bottom: 0.2rem;">{username}</div>
            <div style="opacity: 0.7; font-size: 0.9rem;">{role.replace('_', ' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        
        if st.button("📝 New Post", use_container_width=True):
            NotificationSystem.add("Opening post editor...", "info")
        
        if st.button("📊 View Analytics", use_container_width=True):
            NotificationSystem.add("Loading analytics...", "info")
        
        if st.button("🔔 Notifications", use_container_width=True):
            NotificationSystem.add("Opening notifications...", "info")
        
        st.markdown("---")
        
        # System status from cache
        cache = get_cache()
        cache_key = "system:status"
        
        system_status = cache.get(cache_key)
        if not system_status:
            # Default status if not in cache
            system_status = {
                "API": "🟢 Online",
                "Database": "🟢 Healthy", 
                "Analytics": "🟡 Maintenance",
                "Support": "🟢 Available"
            }
            cache.set(cache_key, system_status, CACHE_TTL["system_status"])
        
        st.markdown("#### 🖥️ System Status")
        
        for service, status in system_status.items():
            st.markdown(f"**{service}**: {status}")
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            # Add confirmation dialog
            st.markdown("Are you sure you want to logout?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Logout", key="confirm_logout"):
                    logout_user(cookies)
            with col2:
                if st.button("Cancel", key="cancel_logout"):
                    st.rerun()
                    st.stop()
        
        # Version info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; font-size: 0.8rem;">
            Civic Catalyst v3.0.0<br>
            © 2024 Platform
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION FLOW
# =============================================================================
def show_reset_password_interface(reset_token, cookies):
    """Show password reset interface for direct token links"""
    # Apply theme
    apply_theme()
    
    # Initialize notification system
    NotificationSystem.init()
    NotificationSystem.render()
    
    st.markdown("""
    <div class="enhanced-container">
        <h2 style="text-align: center; color: var(--primary-color); margin-bottom: 2rem;">
            🔐 Reset Your Password
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Verify token
    db = get_db()
    token_valid = False
    username = None
    
    if db is not None:
        reset_data = db.password_resets.find_one({
            "token": reset_token,
            "used": False,
            "expires": {"$gt": datetime.utcnow()}
        })
        
        if reset_data:
            token_valid = True
            username = reset_data.get("username")
    else:
        # Check cache for demo mode
        cache = get_cache()
        cache_key = f"reset_token:{reset_token}"
        reset_data = cache.get(cache_key)
        
        if reset_data:
            expires = datetime.fromisoformat(reset_data.get("expires", ""))
            if expires > datetime.utcnow():
                token_valid = True
                username = reset_data.get("username")
    
    if not token_valid:
        st.error("❌ This password reset link is invalid or has expired.")
        
        if st.button("🔄 Request a new reset link", use_container_width=True):
            # Go to forgot password page
            create_password_reset_interface()
        
        return
    
    # Show password reset form
    st.markdown(f"### Reset password for user: **{username}**")
    
    with st.form("reset_password_form"):
        new_password = st.text_input(
            "New Password",
            placeholder="Enter new password",
            type="password",
            help=f"Must be at least {PASSWORD_MIN_LENGTH} characters with mix of letters, numbers, and symbols"
        )
        
        # Show password strength
        if new_password:
            strength_html = create_password_strength_visualizer(new_password)
            st.markdown(strength_html, unsafe_allow_html=True)
        
        confirm_password = st.text_input(
            "Confirm New Password",
            placeholder="Confirm new password",
            type="password",
            help="Must match the password above"
        )
        
        if st.form_submit_button("🔐 Reset Password", use_container_width=True):
            if not new_password:
                NotificationSystem.add("Please enter a new password", "error")
            elif new_password != confirm_password:
                NotificationSystem.add("Passwords do not match", "error")
            else:
                password_check = check_password_strength(new_password)
                if not password_check['is_valid']:
                    NotificationSystem.add("Password does not meet security requirements", "error")
                else:
                    with st.spinner("Updating password..."):
                        success = False
                        
                        if db and username:
                            try:
                                # Hash new password
                                password_hash, salt = hash_password(new_password)
                                
                                # Update user password
                                result = db.users.update_one(
                                    {"username": username},
                                    {
                                        "$set": {
                                            "password_hash": password_hash,
                                            "password_salt": salt,
                                            "security.password_last_changed": datetime.utcnow(),
                                            "security.password_expires": datetime.utcnow() + timedelta(days=90),
                                            "security.failed_attempts": 0,
                                            "security.locked_until": None
                                        }
                                    }
                                )
                                
                                if result.modified_count:
                                    # Mark token as used
                                    db.password_resets.update_one(
                                        {"token": reset_token},
                                        {"$set": {"used": True}}
                                    )
                                    
                                    # Log the password reset
                                    db.activity_log.insert_one({
                                        "user": username,
                                        "action": "password_reset",
                                        "timestamp": datetime.utcnow(),
                                        "details": {"method": "reset_token"}
                                    })
                                    
                                    success = True
                            except Exception as e:
                                print(f"Password reset error: {e}")
                        else:
                            # Demo mode - always succeed
                            time.sleep(1)
                            success = True
                            
                            # Remove token from cache
                            cache = get_cache()
                            cache.delete(f"reset_token:{reset_token}")
                        
                        if success:
                            st.success("✅ Password has been reset successfully! You can now log in with your new password.")
                            
                            if st.button("🔐 Go to Login", use_container_width=True):
                                # Clear query params and redirect to login
                                st.query_params.clear()
                                st.rerun()
                                st.stop()
                        else:
                            st.error("❌ Failed to reset password. Please try again.")

def main():
    """Main application entry point"""
    # Initialize MongoDB and Redis connections
    db = get_db()
    cache = get_cache()
    
    # Initialize session state defaults
    if "theme" not in st.session_state:
        st.session_state["theme"] = DEFAULT_THEME
    
    if "site_language" not in st.session_state:
        st.session_state["site_language"] = "en"
    
    # Initialize cookies
    cookies = get_cookies()
    
    # Check for password reset token in URL params
    params = st.query_params
    reset_token = None
    
    if "token" in params:
        token_val = params["token"]
        reset_token = token_val[0] if isinstance(token_val, list) else token_val
        
        if reset_token:
            show_reset_password_interface(reset_token, cookies)
            return
    
    # Check for existing authentication
    if "username" in st.session_state and "role" in st.session_state:
        # User is authenticated, show main interface
        create_authenticated_interface(
            st.session_state["username"],
            st.session_state["role"], 
            cookies
        )
    else:
        # Try to get authentication from cookies
        if "jwt_token" in cookies and "username" in cookies:
            token = cookies["jwt_token"]
            username = cookies["username"]
            
            # Verify token
            valid, payload = verify_jwt_token(token)
            
            if valid and payload.get("sub") == username:
                # Token is valid, restore session
                st.session_state["jwt_token"] = token
                st.session_state["username"] = username
                st.session_state["role"] = payload.get("role", "user")
                
                # Show authenticated interface
                create_authenticated_interface(
                    username,
                    st.session_state["role"],
                    cookies
                )
                return
        
        # User not authenticated, show login interface
        create_login_interface(cookies)

if __name__ == "__main__":
    main()
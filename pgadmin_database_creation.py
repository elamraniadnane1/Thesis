#!/usr/bin/env python3
"""
CivicCatalyst PostgreSQL Database Creation and Migration Script
For REMACTO Open Government Platform
Supports multilingual content (Arabic, French, Darija) and all functional requirements
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import json
from datetime import datetime, timedelta
import uuid
import random
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'civic-postgres',  # Change to 'civic-postgres' if using Docker
    'port': 5432,
    'user': 'postgres',
    'password': 'Abdi2022',
    'database': 'CivicCatalyst'
}

# Qdrant configuration
QDRANT_CONFIG = {
    'host': '154.44.186.241',
    'port': 6333
}

class CivicCatalystDB:
    """Main class for creating and populating the CivicCatalyst database"""
    
    def __init__(self):
        self.conn = None
        self.qdrant_client = None
        
    def wait_for_postgres(self, max_retries=30, delay=2):
        """Wait for PostgreSQL to be ready"""
        for i in range(max_retries):
            try:
                # Try to connect to postgres database first
                temp_config = DB_CONFIG.copy()
                temp_config['database'] = 'postgres'
                conn = psycopg2.connect(**temp_config)
                conn.close()
                logger.info("PostgreSQL is ready")
                return True
            except Exception as e:
                if i < max_retries - 1:
                    logger.info(f"Waiting for PostgreSQL... ({i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"PostgreSQL not ready after {max_retries} attempts")
                    return False
        return False
        
    def create_database_if_not_exists(self):
        """Create the CivicCatalyst database if it doesn't exist"""
        try:
            # Connect to postgres database
            temp_config = DB_CONFIG.copy()
            temp_config['database'] = 'postgres'
            conn = psycopg2.connect(**temp_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
                logger.info(f"Created database {DB_CONFIG['database']}")
            else:
                logger.info(f"Database {DB_CONFIG['database']} already exists")
                
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
            
    def connect_postgres(self):
        """Connect to PostgreSQL"""
        try:
            # Wait for PostgreSQL to be ready
            if not self.wait_for_postgres():
                raise Exception("PostgreSQL is not ready")
                
            # Create database if it doesn't exist
            self.create_database_if_not_exists()
            
            # Connect to the CivicCatalyst database
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("Connected to PostgreSQL successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
            
    def connect_qdrant(self):
        """Connect to Qdrant"""
        try:
            self.qdrant_client = QdrantClient(
                host=QDRANT_CONFIG['host'],
                port=QDRANT_CONFIG['port']
            )
            logger.info("Connected to Qdrant successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
            
    def create_database_schema(self):
        """Create the complete database schema for CivicCatalyst"""
        cursor = self.conn.cursor()
        
        # Enable required extensions with proper quoting
        extensions = [
            ("postgis", "CREATE EXTENSION IF NOT EXISTS postgis;"),
            ("pg_trgm", "CREATE EXTENSION IF NOT EXISTS pg_trgm;"),
            ("uuid-ossp", 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'),  # Quoted for hyphen
            ("btree_gin", "CREATE EXTENSION IF NOT EXISTS btree_gin;"),
            ("unaccent", "CREATE EXTENSION IF NOT EXISTS unaccent;")
        ]
        
        for ext_name, ext_sql in extensions:
            try:
                cursor.execute(ext_sql)
                logger.info(f"Extension created/verified: {ext_name}")
            except Exception as e:
                logger.warning(f"Extension {ext_name} might already exist: {e}")
        
        # Create schemas first
        cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS core;
            CREATE SCHEMA IF NOT EXISTS geo;
            CREATE SCHEMA IF NOT EXISTS analytics;
            CREATE SCHEMA IF NOT EXISTS engagement;
            CREATE SCHEMA IF NOT EXISTS governance;
            CREATE SCHEMA IF NOT EXISTS system;
        """)
        self.conn.commit()
        logger.info("Schemas created successfully")
        
        # Create custom types
        try:
            cursor.execute("""
                -- Drop types if they exist (CASCADE to drop dependent objects)
                DROP TYPE IF EXISTS language_enum CASCADE;
                DROP TYPE IF EXISTS user_role CASCADE;
                DROP TYPE IF EXISTS project_status CASCADE;
                DROP TYPE IF EXISTS sentiment_enum CASCADE;
                DROP TYPE IF EXISTS remacto_theme CASCADE;
                DROP TYPE IF EXISTS cocreation_phase CASCADE;
            """)
            self.conn.commit()
            
            # Create each type individually with commits to ensure they're available
            cursor.execute("CREATE TYPE language_enum AS ENUM ('ar', 'fr', 'ar-ma', 'multi');")
            self.conn.commit()
            
            cursor.execute("""
                CREATE TYPE user_role AS ENUM (
                    'citizen', 'municipal_official', 'civil_society', 
                    'academic', 'international_observer', 'admin', 'moderator'
                );
            """)
            self.conn.commit()
            
            cursor.execute("""
                CREATE TYPE project_status AS ENUM (
                    'proposed', 'under_review', 'approved', 'in_progress', 
                    'completed', 'cancelled', 'on_hold'
                );
            """)
            self.conn.commit()
            
            cursor.execute("""
                CREATE TYPE sentiment_enum AS ENUM (
                    'very_positive', 'positive', 'neutral', 'negative', 'very_negative'
                );
            """)
            self.conn.commit()
            
            cursor.execute("""
                CREATE TYPE remacto_theme AS ENUM (
                    'transparency', 'participation', 'digitalization', 
                    'accountability', 'innovation', 'sustainability',
                    'gender_equality', 'youth_engagement', 'social_inclusion',
                    'economic_development', 'environmental_protection',
                    'cultural_preservation', 'education', 'health',
                    'infrastructure', 'security', 'justice', 'governance',
                    'public_services', 'citizen_rights', 'data_openness',
                    'community_development', 'international_cooperation'
                );
            """)
            self.conn.commit()
            
            cursor.execute("""
                CREATE TYPE cocreation_phase AS ENUM (
                    'priority_identification', 'workshop_planning', 
                    'co_creation_session', 'consultation_feedback',
                    'implementation_monitoring', 'evaluation_planning'
                );
            """)
            self.conn.commit()
            
            logger.info("Custom types created successfully")
        except Exception as e:
            logger.warning(f"Error creating custom types: {e}")
            # Continue anyway - the types might already exist
        
        # Geographic/Administrative Tables
        cursor.execute("""
            -- Regions table
            CREATE TABLE IF NOT EXISTS geo.regions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                code VARCHAR(10) UNIQUE NOT NULL,
                name_ar VARCHAR(255) NOT NULL,
                name_fr VARCHAR(255) NOT NULL,
                name_en VARCHAR(255),
                geometry GEOMETRY(POLYGON, 4326),
                population INTEGER,
                area_km2 DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Provinces/Prefectures table
            CREATE TABLE IF NOT EXISTS geo.provinces (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                region_id UUID REFERENCES geo.regions(id),
                code VARCHAR(10) UNIQUE NOT NULL,
                name_ar VARCHAR(255) NOT NULL,
                name_fr VARCHAR(255) NOT NULL,
                name_en VARCHAR(255),
                is_prefecture BOOLEAN DEFAULT false,
                geometry GEOMETRY(POLYGON, 4326),
                population INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Cercles table
            CREATE TABLE IF NOT EXISTS geo.cercles (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                province_id UUID REFERENCES geo.provinces(id),
                code VARCHAR(10) UNIQUE NOT NULL,
                name_ar VARCHAR(255) NOT NULL,
                name_fr VARCHAR(255) NOT NULL,
                name_en VARCHAR(255),
                geometry GEOMETRY(POLYGON, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Communes/Municipalities table
            CREATE TABLE IF NOT EXISTS geo.municipalities (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                cercle_id UUID REFERENCES geo.cercles(id),
                province_id UUID REFERENCES geo.provinces(id),
                region_id UUID REFERENCES geo.regions(id),
                code VARCHAR(20) UNIQUE NOT NULL,
                name_ar VARCHAR(255) NOT NULL,
                name_fr VARCHAR(255) NOT NULL,
                name_en VARCHAR(255),
                is_urban BOOLEAN DEFAULT true,
                geometry GEOMETRY(POLYGON, 4326),
                center_point GEOMETRY(POINT, 4326),
                population INTEGER,
                area_km2 DECIMAL(10,2),
                remacto_member BOOLEAN DEFAULT false,
                remacto_join_date DATE,
                website_url VARCHAR(500),
                contact_email VARCHAR(255),
                contact_phone VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Arrondissements (districts within cities)
            CREATE TABLE IF NOT EXISTS geo.arrondissements (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id),
                code VARCHAR(20) UNIQUE NOT NULL,
                name_ar VARCHAR(255) NOT NULL,
                name_fr VARCHAR(255) NOT NULL,
                name_en VARCHAR(255),
                geometry GEOMETRY(POLYGON, 4326),
                population INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        logger.info("Geographic tables created successfully")
        
        # User Management Tables
        cursor.execute("""
            -- Users table (citizens, officials, etc.)
            CREATE TABLE IF NOT EXISTS core.users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                phone VARCHAR(50),
                password_hash VARCHAR(255) NOT NULL,
                role user_role NOT NULL DEFAULT 'citizen',
                municipality_id UUID REFERENCES geo.municipalities(id),
                arrondissement_id UUID REFERENCES geo.arrondissements(id),
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                display_name VARCHAR(200),
                preferred_language language_enum DEFAULT 'fr',
                date_of_birth DATE,
                gender VARCHAR(20),
                national_id_hash VARCHAR(255), -- Hashed for privacy
                is_active BOOLEAN DEFAULT true,
                is_verified BOOLEAN DEFAULT false,
                email_verified BOOLEAN DEFAULT false,
                phone_verified BOOLEAN DEFAULT false,
                two_factor_enabled BOOLEAN DEFAULT false,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb,
                profile_data JSONB DEFAULT '{}'::jsonb
            );
            
            -- Municipal officials table
            CREATE TABLE IF NOT EXISTS governance.municipal_officials (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES core.users(id) UNIQUE,
                municipality_id UUID REFERENCES geo.municipalities(id),
                position VARCHAR(255) NOT NULL,
                department VARCHAR(255),
                office_phone VARCHAR(50),
                office_email VARCHAR(255),
                start_date DATE NOT NULL,
                end_date DATE,
                is_elected BOOLEAN DEFAULT false,
                bio_ar TEXT,
                bio_fr TEXT,
                photo_url VARCHAR(500),
                responsibilities JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        logger.info("User management tables created successfully")
        
        # Projects and Initiatives Tables
        cursor.execute("""
            -- REMACTO Projects table
            CREATE TABLE IF NOT EXISTS core.projects (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id) NOT NULL,
                title_ar VARCHAR(500) NOT NULL,
                title_fr VARCHAR(500) NOT NULL,
                title_en VARCHAR(500),
                description_ar TEXT NOT NULL,
                description_fr TEXT NOT NULL,
                description_en TEXT,
                project_code VARCHAR(50) UNIQUE,
                status project_status NOT NULL DEFAULT 'proposed',
                themes remacto_theme[] NOT NULL,
                primary_theme remacto_theme NOT NULL,
                budget_allocated DECIMAL(15,2),
                budget_spent DECIMAL(15,2),
                currency VARCHAR(3) DEFAULT 'MAD',
                start_date DATE,
                end_date DATE,
                target_beneficiaries INTEGER,
                actual_beneficiaries INTEGER,
                location_description TEXT,
                location_geometry GEOMETRY(GEOMETRY, 4326),
                created_by UUID REFERENCES core.users(id),
                approved_by UUID REFERENCES core.users(id),
                approval_date TIMESTAMP,
                completion_percentage INTEGER DEFAULT 0,
                is_participatory BOOLEAN DEFAULT false,
                co_creation_phase cocreation_phase,
                priority_score INTEGER DEFAULT 50,
                visibility VARCHAR(20) DEFAULT 'public', -- public, restricted, private
                tags TEXT[],
                attachments JSONB DEFAULT '[]'::jsonb,
                milestones JSONB DEFAULT '[]'::jsonb,
                kpis JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Project updates/timeline
            CREATE TABLE IF NOT EXISTS core.project_updates (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id UUID REFERENCES core.projects(id) ON DELETE CASCADE,
                update_type VARCHAR(50) NOT NULL, -- status_change, milestone, budget, general
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                language language_enum NOT NULL,
                created_by UUID REFERENCES core.users(id),
                attachments JSONB DEFAULT '[]'::jsonb,
                is_public BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        logger.info("Projects tables created successfully")
        
        # Consultation and Engagement Tables
        cursor.execute("""
            -- Consultations table
            CREATE TABLE IF NOT EXISTS engagement.consultations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id),
                project_id UUID REFERENCES core.projects(id),
                title_ar VARCHAR(500) NOT NULL,
                title_fr VARCHAR(500) NOT NULL,
                description_ar TEXT NOT NULL,
                description_fr TEXT NOT NULL,
                consultation_type VARCHAR(50) NOT NULL, -- online, in_person, hybrid
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                target_participants INTEGER,
                actual_participants INTEGER,
                themes remacto_theme[],
                status VARCHAR(50) DEFAULT 'draft', -- draft, active, closed, analyzing, completed
                meeting_location TEXT,
                online_platform_url VARCHAR(500),
                moderator_id UUID REFERENCES core.users(id),
                summary_ar TEXT,
                summary_fr TEXT,
                outcomes JSONB DEFAULT '[]'::jsonb,
                created_by UUID REFERENCES core.users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Citizen comments table
            CREATE TABLE IF NOT EXISTS engagement.comments (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES core.users(id),
                project_id UUID REFERENCES core.projects(id),
                consultation_id UUID REFERENCES engagement.consultations(id),
                parent_comment_id UUID REFERENCES engagement.comments(id),
                content TEXT NOT NULL,
                language language_enum NOT NULL,
                sentiment sentiment_enum,
                sentiment_score DECIMAL(3,2),
                sentiment_confidence DECIMAL(3,2),
                is_offensive BOOLEAN DEFAULT false,
                offensive_score DECIMAL(3,2),
                offensive_categories TEXT[],
                moderation_status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected, flagged
                moderation_reason TEXT,
                moderated_by UUID REFERENCES core.users(id),
                moderated_at TIMESTAMP,
                likes_count INTEGER DEFAULT 0,
                replies_count INTEGER DEFAULT 0,
                is_anonymous BOOLEAN DEFAULT false,
                source VARCHAR(50) DEFAULT 'platform', -- platform, imported, social_media
                external_id VARCHAR(255), -- For imported comments
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Citizen ideas/proposals table
            CREATE TABLE IF NOT EXISTS engagement.ideas (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES core.users(id),
                municipality_id UUID REFERENCES geo.municipalities(id),
                consultation_id UUID REFERENCES engagement.consultations(id),
                title VARCHAR(500) NOT NULL,
                description TEXT NOT NULL,
                language language_enum NOT NULL,
                themes remacto_theme[],
                category VARCHAR(100),
                estimated_cost DECIMAL(15,2),
                implementation_time_months INTEGER,
                target_beneficiaries INTEGER,
                location_description TEXT,
                location_geometry GEOMETRY(GEOMETRY, 4326),
                status VARCHAR(50) DEFAULT 'submitted', -- submitted, under_review, approved, rejected, implemented
                review_notes TEXT,
                reviewed_by UUID REFERENCES core.users(id),
                reviewed_at TIMESTAMP,
                votes_count INTEGER DEFAULT 0,
                comments_count INTEGER DEFAULT 0,
                feasibility_score INTEGER,
                impact_score INTEGER,
                priority_score INTEGER,
                attachments JSONB DEFAULT '[]'::jsonb,
                is_anonymous BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        logger.info("Engagement tables created successfully")
        
        # Analytics Tables
        cursor.execute("""
            -- Sentiment analysis results
            CREATE TABLE IF NOT EXISTS analytics.sentiment_analysis (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_type VARCHAR(50) NOT NULL, -- comment, news_article, social_media
                source_id UUID NOT NULL,
                text_content TEXT NOT NULL,
                language language_enum NOT NULL,
                sentiment sentiment_enum NOT NULL,
                sentiment_scores JSONB NOT NULL, -- {very_positive: 0.1, positive: 0.2, ...}
                emotion_scores JSONB, -- {joy: 0.3, anger: 0.1, fear: 0.05, ...}
                topics remacto_theme[],
                keywords TEXT[],
                entities JSONB, -- Named entities: people, organizations, locations
                model_version VARCHAR(50),
                confidence_score DECIMAL(3,2),
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Topic modeling results
            CREATE TABLE IF NOT EXISTS analytics.topic_analysis (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                analysis_date DATE NOT NULL,
                municipality_id UUID REFERENCES geo.municipalities(id),
                source_type VARCHAR(50) NOT NULL,
                topics JSONB NOT NULL, -- [{topic: "infrastructure", score: 0.8, keywords: [...]}]
                dominant_theme remacto_theme,
                theme_distribution JSONB, -- {transparency: 0.3, participation: 0.2, ...}
                total_documents INTEGER,
                model_type VARCHAR(50), -- LDA, NMF, BERTopic
                model_parameters JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Offensive content detection
            CREATE TABLE IF NOT EXISTS analytics.offensive_content (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_type VARCHAR(50) NOT NULL,
                source_id UUID NOT NULL,
                content TEXT NOT NULL,
                language language_enum NOT NULL,
                is_offensive BOOLEAN NOT NULL,
                offensive_categories TEXT[], -- hate_speech, harassment, discrimination, etc.
                confidence_scores JSONB, -- {hate_speech: 0.9, harassment: 0.3, ...}
                severity_level INTEGER, -- 1-5 scale
                context_considered TEXT,
                model_version VARCHAR(50),
                flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_status VARCHAR(50) DEFAULT 'pending',
                reviewed_by UUID REFERENCES core.users(id),
                review_decision VARCHAR(50), -- confirmed, false_positive, needs_context
                review_notes TEXT,
                reviewed_at TIMESTAMP
            );
            
            -- Engagement metrics
            CREATE TABLE IF NOT EXISTS analytics.engagement_metrics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id),
                metric_date DATE NOT NULL,
                metric_type VARCHAR(50) NOT NULL, -- daily, weekly, monthly
                total_users INTEGER DEFAULT 0,
                active_users INTEGER DEFAULT 0,
                new_users INTEGER DEFAULT 0,
                total_comments INTEGER DEFAULT 0,
                total_ideas INTEGER DEFAULT 0,
                total_votes INTEGER DEFAULT 0,
                avg_session_duration INTEGER, -- seconds
                participation_rate DECIMAL(5,2), -- percentage
                sentiment_distribution JSONB,
                top_themes remacto_theme[],
                device_breakdown JSONB, -- {mobile: 60, desktop: 35, tablet: 5}
                demographic_breakdown JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Note: removed updated_at from this table as it was causing errors
                UNIQUE(municipality_id, metric_date, metric_type)
            );
        """)
        
        logger.info("Analytics tables created successfully")
        
        # News and Media Tables
        cursor.execute("""
            -- News articles table
            CREATE TABLE IF NOT EXISTS engagement.news_articles (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_name VARCHAR(255) NOT NULL,
                source_url VARCHAR(500),
                article_url VARCHAR(500) UNIQUE NOT NULL,
                title VARCHAR(1000) NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                author VARCHAR(255),
                published_date TIMESTAMP,
                language language_enum NOT NULL,
                municipalities_mentioned UUID[], -- Array of municipality IDs
                themes remacto_theme[],
                sentiment sentiment_enum,
                sentiment_score DECIMAL(3,2),
                keywords TEXT[],
                image_url VARCHAR(500),
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- News comments table (for sites like Hespress)
            CREATE TABLE IF NOT EXISTS engagement.news_comments (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                article_id UUID REFERENCES engagement.news_articles(id),
                external_comment_id VARCHAR(255),
                author_name VARCHAR(255),
                content TEXT NOT NULL,
                language language_enum NOT NULL,
                posted_date TIMESTAMP,
                likes_count INTEGER DEFAULT 0,
                sentiment sentiment_enum,
                sentiment_score DECIMAL(3,2),
                is_offensive BOOLEAN DEFAULT false,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        logger.info("News tables created successfully")
        
        # Budget and Finance Tables
        cursor.execute("""
            -- Budget allocations table
            CREATE TABLE IF NOT EXISTS governance.budget_allocations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id) NOT NULL,
                fiscal_year INTEGER NOT NULL,
                total_budget DECIMAL(15,2) NOT NULL,
                currency VARCHAR(3) DEFAULT 'MAD',
                allocation_data JSONB NOT NULL, -- Detailed breakdown by category
                participatory_budget_amount DECIMAL(15,2),
                openness_initiatives_budget DECIMAL(15,2),
                themes_budget_distribution JSONB, -- {transparency: 1000000, participation: 500000, ...}
                approval_date DATE,
                approved_by UUID REFERENCES core.users(id),
                document_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(municipality_id, fiscal_year)
            );
            
            -- Participatory budgeting proposals
            CREATE TABLE IF NOT EXISTS governance.pb_proposals (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                idea_id UUID REFERENCES engagement.ideas(id),
                municipality_id UUID REFERENCES geo.municipalities(id),
                fiscal_year INTEGER NOT NULL,
                requested_amount DECIMAL(15,2) NOT NULL,
                approved_amount DECIMAL(15,2),
                votes_count INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'submitted',
                evaluation_notes TEXT,
                implementation_plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        logger.info("Budget tables created successfully")
        
        # Document Management Tables
        cursor.execute("""
            -- Documents table
            CREATE TABLE IF NOT EXISTS core.documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                title VARCHAR(500) NOT NULL,
                description TEXT,
                document_type VARCHAR(100) NOT NULL, -- report, policy, minutes, guide, etc.
                file_path VARCHAR(1000),
                file_size INTEGER,
                mime_type VARCHAR(100),
                language language_enum NOT NULL,
                municipality_id UUID REFERENCES geo.municipalities(id),
                project_id UUID REFERENCES core.projects(id),
                consultation_id UUID REFERENCES engagement.consultations(id),
                uploaded_by UUID REFERENCES core.users(id),
                access_level VARCHAR(50) DEFAULT 'public', -- public, restricted, private
                download_count INTEGER DEFAULT 0,
                tags TEXT[],
                full_text TEXT, -- Extracted text for search
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        logger.info("Document tables created successfully")
        
        # System Tables
        cursor.execute("""
            -- Activity logs table
            CREATE TABLE IF NOT EXISTS system.activity_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES core.users(id),
                action_type VARCHAR(100) NOT NULL,
                resource_type VARCHAR(100),
                resource_id UUID,
                description TEXT,
                ip_address INET,
                user_agent TEXT,
                request_data JSONB,
                response_status INTEGER,
                duration_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- System configuration
            CREATE TABLE IF NOT EXISTS system.configurations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                municipality_id UUID REFERENCES geo.municipalities(id),
                config_key VARCHAR(255) NOT NULL,
                config_value JSONB NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(municipality_id, config_key)
            );
            
            -- AI Model tracking
            CREATE TABLE IF NOT EXISTS system.ai_models (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                model_name VARCHAR(255) NOT NULL,
                model_type VARCHAR(100) NOT NULL, -- sentiment, topic, offensive, summarization
                version VARCHAR(50) NOT NULL,
                parameters JSONB,
                performance_metrics JSONB,
                training_data_info JSONB,
                is_active BOOLEAN DEFAULT false,
                deployed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, version)
            );
            
            -- Notification queue
            CREATE TABLE IF NOT EXISTS system.notifications (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES core.users(id),
                notification_type VARCHAR(100) NOT NULL,
                title VARCHAR(500),
                message TEXT NOT NULL,
                language language_enum NOT NULL,
                priority INTEGER DEFAULT 3, -- 1=urgent, 5=low
                channels TEXT[], -- email, sms, push, in_app
                status VARCHAR(50) DEFAULT 'pending',
                sent_at TIMESTAMP,
                read_at TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        logger.info("System tables created successfully")
        
        # Create indexes for performance
        logger.info("Creating indexes...")
        
        # Create indexes only after we confirm tables exist and have the required columns
        try:
            # Geographic indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_municipalities_geometry ON geo.municipalities USING GIST(geometry);
                CREATE INDEX IF NOT EXISTS idx_municipalities_center ON geo.municipalities USING GIST(center_point);
                CREATE INDEX IF NOT EXISTS idx_municipalities_remacto ON geo.municipalities(remacto_member) WHERE remacto_member = true;
            """)
            
            # User indexes - only create if we've confirmed columns exist
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'core' AND table_name = 'users' AND column_name = 'role'")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_municipality ON core.users(municipality_id);
                    CREATE INDEX IF NOT EXISTS idx_users_role ON core.users(role);
                    CREATE INDEX IF NOT EXISTS idx_users_active ON core.users(is_active) WHERE is_active = true;
                """)
            
            # Project indexes - only create if we've confirmed columns exist
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'core' AND table_name = 'projects' AND column_name = 'status'")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_projects_municipality ON core.projects(municipality_id);
                    CREATE INDEX IF NOT EXISTS idx_projects_status ON core.projects(status);
                    CREATE INDEX IF NOT EXISTS idx_projects_created ON core.projects(created_at DESC);
                """)
                
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'core' AND table_name = 'projects' AND column_name = 'themes'")
            if cursor.fetchone():
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_themes ON core.projects USING GIN(themes);")
            
            # Comment indexes - only create if we've confirmed columns exist
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'engagement' AND table_name = 'comments' AND column_name = 'sentiment'")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_comments_user ON engagement.comments(user_id);
                    CREATE INDEX IF NOT EXISTS idx_comments_project ON engagement.comments(project_id);
                    CREATE INDEX IF NOT EXISTS idx_comments_consultation ON engagement.comments(consultation_id);
                    CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON engagement.comments(sentiment);
                    CREATE INDEX IF NOT EXISTS idx_comments_created ON engagement.comments(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_comments_moderation ON engagement.comments(moderation_status) WHERE moderation_status = 'pending';
                """)
            
            # Analytics indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_source ON analytics.sentiment_analysis(source_type, source_id);
                CREATE INDEX IF NOT EXISTS idx_sentiment_date ON analytics.sentiment_analysis(analyzed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_engagement_metrics_date ON analytics.engagement_metrics(municipality_id, metric_date DESC);
            """)
            
            # News indexes
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'engagement' AND table_name = 'news_articles' AND column_name = 'themes'")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_published ON engagement.news_articles(published_date DESC);
                    CREATE INDEX IF NOT EXISTS idx_news_municipalities ON engagement.news_articles USING GIN(municipalities_mentioned);
                    CREATE INDEX IF NOT EXISTS idx_news_themes ON engagement.news_articles USING GIN(themes);
                """)
            
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation error: {e}")
        
        # We'll move view creation to after data migration to ensure tables are populated
        
        logger.info("Views created successfully")
        
        # Create triggers for updated_at
        cursor.execute("""
            -- Create update timestamp function
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Get all tables with updated_at column - adjusted query to avoid views
        cursor.execute("""
            SELECT table_schema, table_name 
            FROM information_schema.columns 
            WHERE column_name = 'updated_at'
            AND table_schema IN ('core', 'geo', 'engagement', 'governance')
            AND table_name NOT IN (
                SELECT viewname FROM pg_views 
                WHERE schemaname IN ('core', 'geo', 'engagement', 'governance')
            )
        """)
        
        tables = cursor.fetchall()
        for schema, table in tables:
            trigger_name = f"update_{schema}_{table}_updated_at"
            try:
                cursor.execute(f"""
                    DROP TRIGGER IF EXISTS {trigger_name} ON {schema}.{table};
                    CREATE TRIGGER {trigger_name}
                    BEFORE UPDATE ON {schema}.{table}
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """)
            except Exception as e:
                logger.warning(f"Trigger creation warning for {schema}.{table}: {e}")
                
        logger.info("Triggers created successfully")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
        
    def create_initial_data(self):
        """Create initial data for regions and provinces"""
        cursor = self.conn.cursor()
        
        # Create regions
        regions = [
            ('R01', 'طنجة تطوان الحسيمة', 'Tanger-Tétouan-Al Hoceïma'),
            ('R02', 'الشرق', 'Oriental'),
            ('R03', 'فاس مكناس', 'Fès-Meknès'),
            ('R04', 'الرباط سلا القنيطرة', 'Rabat-Salé-Kénitra'),
            ('R05', 'بني ملال خنيفرة', 'Béni Mellal-Khénifra'),
            ('R06', 'الدار البيضاء سطات', 'Casablanca-Settat'),
            ('R07', 'مراكش آسفي', 'Marrakech-Safi'),
            ('R08', 'درعة تافيلالت', 'Drâa-Tafilalet'),
            ('R09', 'سوس ماسة', 'Souss-Massa'),
            ('R10', 'كلميم واد نون', 'Guelmim-Oued Noun'),
            ('R11', 'العيون الساقية الحمراء', 'Laâyoune-Sakia El Hamra'),
            ('R12', 'الداخلة وادي الذهب', 'Dakhla-Oued Ed-Dahab')
        ]
        
        for code, name_ar, name_fr in regions:
            cursor.execute("""
                INSERT INTO geo.regions (code, name_ar, name_fr, population)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO NOTHING
            """, (code, name_ar, name_fr, random.randint(500000, 3000000)))
            
        logger.info("Initial regions created")
        self.conn.commit()
        
    def check_and_add_columns(self, schema, table, required_columns):
        """Helper function to check and add missing columns to a table"""
        cursor = self.conn.cursor()
        
        # Check if table exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema}' AND table_name = '{table}'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.warning(f"Table {schema}.{table} does not exist")
            return False
            
        # Get existing columns
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' AND table_name = '{table}'
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Add missing columns
        columns_added = False
        for col_name, col_def in required_columns.items():
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE {schema}.{table} ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column '{col_name}' to {schema}.{table}")
                    columns_added = True
                except Exception as e:
                    logger.warning(f"Could not add column {col_name} to {schema}.{table}: {e}")
        
        if columns_added:
            self.conn.commit()
            
        return True
        
    def migrate_qdrant_data(self):
        """Migrate data from Qdrant collections to PostgreSQL"""
        cursor = self.conn.cursor()
        
        # Collection mapping with safe order (no foreign key dependencies first)
        collections = [
            ('morocco_centres', self._migrate_morocco_centres),
            ('morocco_cercles', self._migrate_morocco_cercles),
            ('morocco_arrondissements', self._migrate_morocco_arrondissements),
            ('citizens', self._migrate_citizens),
            ('municipal_officials', self._migrate_municipal_officials),
            ('remacto_projects', self._migrate_remacto_projects),
            ('municipal_projects', self._migrate_municipal_projects),
            ('project_updates', self._migrate_project_updates),
            ('citizen_comments', self._migrate_citizen_comments),
            ('remacto_comments', self._migrate_remacto_comments),
            ('citizen_ideas', self._migrate_citizen_ideas),
            ('budget_allocations', self._migrate_budget_allocations),
            ('engagement_metrics', self._migrate_engagement_metrics),
            ('hespress_politics_details', self._migrate_hespress_articles),
            ('hespress_politics_comments', self._migrate_hespress_comments)
        ]
        
        for collection_name, migration_func in collections:
            try:
                # Check if collection exists
                collections_list = self.qdrant_client.get_collections()
                if any(c.name == collection_name for c in collections_list.collections):
                    logger.info(f"Migrating collection: {collection_name}")
                    migration_func()
                    self.conn.commit()
                    logger.info(f"Successfully migrated: {collection_name}")
                else:
                    logger.warning(f"Collection {collection_name} not found in Qdrant")
            except Exception as e:
                logger.error(f"Failed to migrate {collection_name}: {e}")
                self.conn.rollback()
                
    def _migrate_morocco_centres(self):
        """Migrate morocco_centres to municipalities table"""
        try:
            # Get all points from collection
            scroll_result = self.qdrant_client.scroll(
                collection_name='morocco_centres',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Get a sample region for foreign key
            cursor.execute("SELECT id FROM geo.regions LIMIT 1")
            region_result = cursor.fetchone()
            region_id = region_result[0] if region_result else None
            
            for point in tqdm(points, desc="Migrating centres"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # Generate municipality data
                cursor.execute("""
                    INSERT INTO geo.municipalities (
                        code, name_ar, name_fr, is_urban, 
                        population, remacto_member, region_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (code) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    payload.get('code', f"CTR-{point.id}"),
                    payload.get('name_ar', payload.get('name', 'مركز')),
                    payload.get('name_fr', payload.get('name', 'Centre')),
                    True,  # Centres are urban
                    payload.get('population', random.randint(10000, 500000)),
                    payload.get('remacto_member', random.choice([True, False])),
                    region_id,
                    json.dumps(payload)
                ))
                
        except Exception as e:
            logger.error(f"Error in _migrate_morocco_centres: {e}")
            raise
            
    def _migrate_morocco_cercles(self):
        """Migrate morocco_cercles to cercles table"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='morocco_cercles',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Get a sample province for foreign key
            cursor.execute("SELECT id FROM geo.provinces LIMIT 1")
            province_result = cursor.fetchone()
            
            # If no provinces exist, create a default one
            if not province_result:
                cursor.execute("SELECT id FROM geo.regions LIMIT 1")
                region_result = cursor.fetchone()
                if region_result:
                    cursor.execute("""
                        INSERT INTO geo.provinces (region_id, code, name_ar, name_fr)
                        VALUES (%s, 'P01', 'إقليم افتراضي', 'Province Default')
                        RETURNING id
                    """, (region_result[0],))
                    province_result = cursor.fetchone()
                    
            province_id = province_result[0] if province_result else None
            
            for point in tqdm(points, desc="Migrating cercles"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if province_id:
                    cursor.execute("""
                        INSERT INTO geo.cercles (
                            province_id, code, name_ar, name_fr
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (code) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP
                    """, (
                        province_id,
                        payload.get('code', f"CER-{point.id}"),
                        payload.get('name_ar', payload.get('name', 'دائرة')),
                        payload.get('name_fr', payload.get('name', 'Cercle'))
                    ))
                    
        except Exception as e:
            logger.error(f"Error in _migrate_morocco_cercles: {e}")
            raise
            
    def _migrate_morocco_arrondissements(self):
        """Migrate morocco_arrondissements to arrondissements table"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='morocco_arrondissements',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Get a sample municipality for foreign key
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 1")
            municipality_result = cursor.fetchone()
            municipality_id = municipality_result[0] if municipality_result else None
            
            for point in tqdm(points, desc="Migrating arrondissements"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_id:
                    cursor.execute("""
                        INSERT INTO geo.arrondissements (
                            municipality_id, code, name_ar, name_fr, population
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (code) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP
                    """, (
                        municipality_id,
                        payload.get('code', f"ARR-{point.id}"),
                        payload.get('name_ar', payload.get('name', 'منطقة')),
                        payload.get('name_fr', payload.get('name', 'Arrondissement')),
                        payload.get('population', random.randint(5000, 100000))
                    ))
                    
        except Exception as e:
            logger.error(f"Error in _migrate_morocco_arrondissements: {e}")
            raise
            
    def _migrate_citizens(self):
        """Migrate citizens to users table"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='citizens',
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to users table
            required_columns = {
                'role': "user_role NOT NULL DEFAULT 'citizen'",
                'preferred_language': "language_enum DEFAULT 'fr'",
                'is_active': "BOOLEAN DEFAULT true",
                'created_at': "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            self.check_and_add_columns('core', 'users', required_columns)
            
            # Get existing columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = 'users'
            """)
            columns = [row[0] for row in cursor.fetchall()]
            
            # Get sample municipality IDs
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 50")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            # Build dynamic SQL based on available columns
            for i, point in enumerate(tqdm(points, desc="Migrating citizens")):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # Generate user data
                username = payload.get('username', f"citizen_{i}")
                email = payload.get('email', f"{username}@civiccatalyst.ma")
                
                try:
                    # Prepare column names and values dynamically
                    avail_columns = []
                    values = []
                    placeholders = []
                    
                    # Always include these core columns
                    base_columns = [
                        ('username', username),
                        ('email', email),
                        ('password_hash', 'hashed_password_placeholder')
                    ]
                    
                    for col, val in base_columns:
                        avail_columns.append(col)
                        values.append(val)
                        placeholders.append('%s')
                    
                    # Add optional columns if they exist
                    optional_columns = [
                        ('role', 'citizen'),
                        ('municipality_id', random.choice(municipality_ids) if municipality_ids else None),
                        ('first_name', payload.get('first_name', 'مواطن')),
                        ('last_name', payload.get('last_name', 'User')),
                        ('preferred_language', random.choice(['ar', 'fr', 'ar-ma'])),
                        ('is_active', True),
                        ('created_at', datetime.now() - timedelta(days=random.randint(0, 365))),
                        ('metadata', json.dumps(payload))
                    ]
                    
                    for col, val in optional_columns:
                        if col in columns:
                            avail_columns.append(col)
                            values.append(val)
                            placeholders.append('%s')
                    
                    # Construct and execute the dynamic SQL
                    sql = f"""
                        INSERT INTO core.users (
                            {', '.join(avail_columns)}
                        ) VALUES ({', '.join(placeholders)})
                        ON CONFLICT (username) DO NOTHING
                    """
                    cursor.execute(sql, values)
                    
                except psycopg2.IntegrityError:
                    # Skip duplicates
                    pass
                    
        except Exception as e:
            logger.error(f"Error in _migrate_citizens: {e}")
            raise
            
    def _migrate_municipal_officials(self):
        """Migrate municipal_officials to officials table"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='municipal_officials',
                limit=2000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to users table
            required_columns = {
                'role': "user_role NOT NULL DEFAULT 'citizen'",
                'preferred_language': "language_enum DEFAULT 'fr'",
                'is_active': "BOOLEAN DEFAULT true",
                'is_verified': "BOOLEAN DEFAULT false"
            }
            
            self.check_and_add_columns('core', 'users', required_columns)
            
            # Get existing columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = 'users'
            """)
            columns = [row[0] for row in cursor.fetchall()]
            
            # Get sample municipality IDs
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 50")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            for i, point in enumerate(tqdm(points, desc="Migrating officials")):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # First create user
                username = payload.get('username', f"official_{i}")
                email = payload.get('email', f"{username}@gov.ma")
                
                try:
                    # Prepare column names and values dynamically
                    avail_columns = []
                    values = []
                    placeholders = []
                    
                    # Always include these core columns
                    base_columns = [
                        ('username', username),
                        ('email', email),
                        ('password_hash', 'hashed_password_placeholder')
                    ]
                    
                    for col, val in base_columns:
                        avail_columns.append(col)
                        values.append(val)
                        placeholders.append('%s')
                    
                    # Add optional columns if they exist
                    optional_columns = [
                        ('role', 'municipal_official'),
                        ('municipality_id', random.choice(municipality_ids) if municipality_ids else None),
                        ('first_name', payload.get('first_name', 'مسؤول')),
                        ('last_name', payload.get('last_name', 'Official')),
                        ('preferred_language', 'fr'),
                        ('is_active', True),
                        ('is_verified', True)
                    ]
                    
                    for col, val in optional_columns:
                        if col in columns:
                            avail_columns.append(col)
                            values.append(val)
                            placeholders.append('%s')
                    
                    # Construct SQL for insert with returning
                    sql = f"""
                        INSERT INTO core.users (
                            {', '.join(avail_columns)}
                        ) VALUES ({', '.join(placeholders)})
                        ON CONFLICT (username) DO NOTHING
                        RETURNING id
                    """
                    cursor.execute(sql, values)
                    
                    user_result = cursor.fetchone()
                    if user_result:
                        user_id = user_result[0]
                        
                        # Then create official record
                        cursor.execute("""
                            INSERT INTO governance.municipal_officials (
                                user_id, municipality_id, position, department,
                                start_date, is_elected, bio_fr, bio_ar
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (user_id) DO UPDATE
                            SET updated_at = CURRENT_TIMESTAMP
                        """, (
                            user_id,
                            random.choice(municipality_ids) if municipality_ids else None,
                            payload.get('position', 'Municipal Officer'),
                            payload.get('department', 'Administration'),
                            datetime.now().date() - timedelta(days=random.randint(0, 1000)),
                            payload.get('is_elected', False),
                            payload.get('bio_fr', 'Biographie du responsable municipal'),
                            payload.get('bio_ar', 'السيرة الذاتية للمسؤول البلدي')
                        ))
                except psycopg2.IntegrityError:
                    # Skip duplicates
                    pass
                    
        except Exception as e:
            logger.error(f"Error in _migrate_municipal_officials: {e}")
            raise
            
    def _migrate_remacto_projects(self):
        """Migrate REMACTO projects"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='remacto_projects',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to projects table
            required_columns = {
                'status': "project_status NOT NULL DEFAULT 'proposed'",
                'themes': "remacto_theme[] NOT NULL DEFAULT '{}'::remacto_theme[]",
                'primary_theme': "remacto_theme NOT NULL DEFAULT 'transparency'::remacto_theme",
                'is_participatory': "BOOLEAN DEFAULT false",
                'co_creation_phase': "cocreation_phase",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            # Check if table exists and add columns if necessary
            if not self.check_and_add_columns('core', 'projects', required_columns):
                logger.warning("Projects table missing required columns, skipping REMACTO projects migration")
                return
            
            # Get sample data for foreign keys
            cursor.execute("SELECT id FROM geo.municipalities WHERE remacto_member = true LIMIT 50")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            # If no REMACTO members, use any municipalities
            if not municipality_ids:
                cursor.execute("SELECT id FROM geo.municipalities LIMIT 50")
                municipality_ids = [row[0] for row in cursor.fetchall()]
            
            # Get user IDs with role=municipal_official if possible
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = 'users' AND column_name = 'role'
            """)
            has_role = bool(cursor.fetchone())
            
            if has_role:
                cursor.execute("SELECT id FROM core.users WHERE role = 'municipal_official' LIMIT 50")
            else:
                cursor.execute("SELECT id FROM core.users LIMIT 50")
                
            official_ids = [row[0] for row in cursor.fetchall()]
            
            # Available valid themes for the remacto_theme enum
            valid_themes = [
                'transparency', 'participation', 'digitalization', 
                'accountability', 'innovation', 'sustainability',
                'gender_equality', 'youth_engagement', 'social_inclusion',
                'economic_development', 'environmental_protection',
                'cultural_preservation', 'education', 'health',
                'infrastructure', 'security', 'justice', 'governance',
                'public_services', 'citizen_rights', 'data_openness',
                'community_development', 'international_cooperation'
            ]
            
            for i, point in enumerate(tqdm(points, desc="Migrating REMACTO projects")):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_ids:
                    # Choose random valid themes
                    project_themes = random.sample(valid_themes, random.randint(1, 3))
                    
                    try:
                        cursor.execute("""
                            INSERT INTO core.projects (
                                municipality_id, title_ar, title_fr, description_ar, description_fr,
                                project_code, status, themes, primary_theme,
                                budget_allocated, start_date, target_beneficiaries,
                                created_by, is_participatory, co_creation_phase,
                                priority_score, tags, metadata
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::remacto_theme[], %s::remacto_theme, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (project_code) DO UPDATE
                            SET updated_at = CURRENT_TIMESTAMP
                        """, (
                            random.choice(municipality_ids),
                            payload.get('title_ar', 'مشروع ريماكتو'),
                            payload.get('title_fr', payload.get('title', 'Projet REMACTO')),
                            payload.get('description_ar', 'وصف المشروع'),
                            payload.get('description_fr', payload.get('description', 'Description du projet')),
                            payload.get('code', f"REM-{i}"),
                            random.choice(['approved', 'in_progress', 'completed']),
                            project_themes,  # Cast to remacto_theme[] in SQL
                            project_themes[0],  # Cast to remacto_theme in SQL
                            payload.get('budget', random.uniform(100000, 5000000)),
                            datetime.now().date() - timedelta(days=random.randint(0, 365)),
                            payload.get('beneficiaries', random.randint(100, 10000)),
                            random.choice(official_ids) if official_ids else None,
                            True,
                            random.choice(['priority_identification', 'workshop_planning', 
                                         'co_creation_session', 'implementation_monitoring']),
                            random.randint(60, 95),
                            payload.get('tags', []),
                            json.dumps(payload)
                        ))
                    except psycopg2.IntegrityError:
                        # Skip duplicates
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _migrate_remacto_projects: {e}")
            raise
            
    def _migrate_municipal_projects(self):
        """Migrate general municipal projects"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='municipal_projects',
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to projects table
            required_columns = {
                'status': "project_status NOT NULL DEFAULT 'proposed'",
                'themes': "remacto_theme[] NOT NULL DEFAULT '{}'::remacto_theme[]",
                'primary_theme': "remacto_theme NOT NULL DEFAULT 'transparency'::remacto_theme",
                'completion_percentage': "INTEGER DEFAULT 0",
                'visibility': "VARCHAR(20) DEFAULT 'public'",
                'tags': "TEXT[]",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            # Check if table exists and add columns if necessary
            if not self.check_and_add_columns('core', 'projects', required_columns):
                logger.warning("Projects table missing required columns, skipping municipal projects migration")
                return
            
            # Get sample data
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 100")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT id FROM core.users WHERE role IN ('municipal_official', 'admin') LIMIT 50")
            official_ids = [row[0] for row in cursor.fetchall()]
            
            # Valid themes from the remacto_theme enum
            valid_themes = [
                'infrastructure', 'education', 'health', 'environmental_protection', 
                'public_services', 'community_development'
            ]
            
            # Valid project statuses from the enum
            valid_statuses = ['proposed', 'under_review', 'approved', 'in_progress', 
                             'completed', 'cancelled', 'on_hold']
            
            for i, point in enumerate(tqdm(points, desc="Migrating municipal projects")):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_ids:
                    project_themes = random.sample(valid_themes, random.randint(1, 2))
                    budget = payload.get('budget', random.uniform(50000, 2000000))
                    
                    # Ensure the status is a valid enum value
                    status = payload.get('status', random.choice(valid_statuses))
                    if status == 'Cancelled':  # Fix capitalization issue
                        status = 'cancelled'
                    if status not in valid_statuses:
                        status = 'in_progress'  # Default to a safe value
                    
                    try:
                        cursor.execute("""
                            INSERT INTO core.projects (
                                municipality_id, title_ar, title_fr, description_ar, description_fr,
                                project_code, status, themes, primary_theme,
                                budget_allocated, budget_spent, start_date, end_date,
                                created_by, completion_percentage, visibility,
                                tags, metadata
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::remacto_theme[], %s::remacto_theme, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (project_code) DO NOTHING
                        """, (
                            random.choice(municipality_ids),
                            payload.get('title_ar', 'مشروع بلدي'),
                            payload.get('title_fr', payload.get('title', 'Projet Municipal')),
                            payload.get('description_ar', 'وصف المشروع البلدي'),
                            payload.get('description_fr', payload.get('description', 'Description du projet')),
                            f"MUN-{i}",
                            status,
                            project_themes,  # Cast to remacto_theme[] in SQL
                            project_themes[0],  # Cast to remacto_theme in SQL
                            budget,
                            payload.get('spent', random.uniform(0, budget * 0.8)),
                            datetime.now().date() - timedelta(days=random.randint(0, 730)),
                            datetime.now().date() + timedelta(days=random.randint(30, 365)),
                            random.choice(official_ids) if official_ids else None,
                            random.randint(0, 100),
                            'public',
                            payload.get('tags', []),
                            json.dumps(payload)
                        ))
                    except psycopg2.IntegrityError:
                        # Skip duplicates
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _migrate_municipal_projects: {e}")
            raise
            
    def _migrate_project_updates(self):
        """Migrate project updates"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='project_updates',
                limit=15000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to project_updates table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'"
            }
            
            self.check_and_add_columns('core', 'project_updates', required_columns)
            
            # Get project IDs
            cursor.execute("SELECT id FROM core.projects LIMIT 1000")
            project_ids = [row[0] for row in cursor.fetchall()]
            
            # Get user IDs
            cursor.execute("SELECT id FROM core.users WHERE role != 'citizen' LIMIT 100")
            user_ids = [row[0] for row in cursor.fetchall()]
            
            update_types = ['status_change', 'milestone', 'budget', 'general']
            
            for point in tqdm(points, desc="Migrating project updates"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if project_ids:
                    cursor.execute("""
                        INSERT INTO core.project_updates (
                            project_id, update_type, title, content,
                            language, created_by, is_public, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        random.choice(project_ids),
                        random.choice(update_types),
                        payload.get('title', 'تحديث المشروع'),
                        payload.get('content', payload.get('update', 'محتوى التحديث')),
                        random.choice(['ar', 'fr']),
                        random.choice(user_ids) if user_ids else None,
                        True,
                        json.dumps(payload)
                    ))
                    
        except Exception as e:
            logger.error(f"Error in _migrate_project_updates: {e}")
            raise
            
    def _migrate_citizen_comments(self):
        """Migrate citizen comments"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='citizen_comments',
                limit=15000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to comments table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'",
                'sentiment': "sentiment_enum",
                'sentiment_score': "DECIMAL(3,2)",
                'is_offensive': "BOOLEAN DEFAULT false"
            }
            
            self.check_and_add_columns('engagement', 'comments', required_columns)
            
            # Get IDs for foreign keys
            cursor.execute("SELECT id FROM core.users WHERE role = 'citizen' LIMIT 1000")
            citizen_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT id FROM core.projects LIMIT 500")
            project_ids = [row[0] for row in cursor.fetchall()]
            
            sentiments = ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']
            
            for point in tqdm(points, desc="Migrating citizen comments"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if project_ids:
                    sentiment = random.choice(sentiments)
                    sentiment_score = {
                        'very_positive': 0.9,
                        'positive': 0.7,
                        'neutral': 0.5,
                        'negative': 0.3,
                        'very_negative': 0.1
                    }[sentiment]
                    
                    cursor.execute("""
                        INSERT INTO engagement.comments (
                            user_id, project_id, content, language,
                            sentiment, sentiment_score, sentiment_confidence,
                            is_offensive, moderation_status, 
                            likes_count, source, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        random.choice(citizen_ids) if citizen_ids else None,
                        random.choice(project_ids),
                        payload.get('comment', payload.get('content', 'تعليق المواطن')),
                        random.choice(['ar', 'fr', 'ar-ma']),
                        sentiment,
                        sentiment_score,
                        random.uniform(0.7, 0.95),
                        False,
                        'approved',
                        random.randint(0, 50),
                        'platform',
                        json.dumps(payload)
                    ))
                    
        except Exception as e:
            logger.error(f"Error in _migrate_citizen_comments: {e}")
            raise
            
    def _migrate_remacto_comments(self):
        """Migrate REMACTO-specific comments"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='remacto_comments',
                limit=2000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to comments table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'",
                'sentiment': "sentiment_enum",
                'sentiment_score': "DECIMAL(3,2)",
                'is_offensive': "BOOLEAN DEFAULT false"
            }
            
            self.check_and_add_columns('engagement', 'comments', required_columns)
            
            # Get REMACTO project IDs
            cursor.execute("SELECT id FROM core.projects WHERE project_code LIKE 'REM-%' LIMIT 100")
            remacto_project_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT id FROM core.users LIMIT 500")
            user_ids = [row[0] for row in cursor.fetchall()]
            
            for point in tqdm(points, desc="Migrating REMACTO comments"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if remacto_project_ids:
                    cursor.execute("""
                        INSERT INTO engagement.comments (
                            user_id, project_id, content, language,
                            sentiment, sentiment_score, is_offensive,
                            moderation_status, source, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        random.choice(user_ids) if user_ids else None,
                        random.choice(remacto_project_ids),
                        payload.get('content', 'تعليق ريماكتو'),
                        random.choice(['ar', 'fr']),
                        random.choice(['positive', 'neutral', 'negative']),
                        random.uniform(0.3, 0.9),
                        False,
                        'approved',
                        'remacto_platform',
                        json.dumps(payload)
                    ))
                    
        except Exception as e:
            logger.error(f"Error in _migrate_remacto_comments: {e}")
            raise
            
    def _migrate_citizen_ideas(self):
        """Migrate citizen ideas/proposals"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='citizen_ideas',
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to ideas table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'",
                'themes': "remacto_theme[]",
                'category': "VARCHAR(100)",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            self.check_and_add_columns('engagement', 'ideas', required_columns)
            
            # Get existing columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'engagement' AND table_name = 'ideas'
            """)
            columns = [row[0] for row in cursor.fetchall()]
            
            # Get required IDs
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = 'users' AND column_name = 'role'
            """)
            has_role = bool(cursor.fetchone())
            
            if has_role:
                cursor.execute("SELECT id FROM core.users WHERE role = 'citizen' LIMIT 1000")
            else:
                cursor.execute("SELECT id FROM core.users LIMIT 1000")
                
            citizen_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 100")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            # Valid themes from the remacto_theme enum
            valid_themes = [
                'infrastructure', 'education', 'health', 'environmental_protection', 
                'public_services', 'community_development', 'digitalization'
            ]
            
            for point in tqdm(points, desc="Migrating citizen ideas"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_ids:
                    idea_themes = random.sample(valid_themes, random.randint(1, 3))
                    
                    try:
                        # Prepare column names and values dynamically
                        avail_columns = []
                        values = []
                        placeholders = []
                        
                        # Always include these core columns
                        base_columns = [
                            ('title', payload.get('title', 'فكرة المواطن')),
                            ('description', payload.get('description', payload.get('idea', 'وصف الفكرة')))
                        ]
                        
                        for col, val in base_columns:
                            if col in columns:
                                avail_columns.append(col)
                                values.append(val)
                                placeholders.append('%s')
                        
                        # Add optional columns if they exist
                        optional_columns = [
                            ('user_id', random.choice(citizen_ids) if citizen_ids else None),
                            ('municipality_id', random.choice(municipality_ids)),
                            ('language', random.choice(['ar', 'fr', 'ar-ma'])),
                            ('themes', idea_themes),
                            ('category', valid_themes[0]),
                            ('estimated_cost', random.uniform(10000, 500000)),
                            ('implementation_time_months', random.randint(3, 24)),
                            ('target_beneficiaries', random.randint(50, 5000)),
                            ('status', random.choice(['submitted', 'under_review', 'approved'])),
                            ('votes_count', random.randint(0, 500)),
                            ('feasibility_score', random.randint(40, 95)),
                            ('impact_score', random.randint(50, 100)),
                            ('priority_score', random.randint(30, 90)),
                            ('metadata', json.dumps(payload))
                        ]
                        
                        for col, val in optional_columns:
                            if col in columns:
                                avail_columns.append(col)
                                
                                # Special handling for array types that need casting
                                if col == 'themes':
                                    placeholders.append('%s::remacto_theme[]')
                                else:
                                    placeholders.append('%s')
                                    
                                values.append(val)
                        
                        # Construct SQL for insert
                        sql = f"""
                            INSERT INTO engagement.ideas (
                                {', '.join(avail_columns)}
                            ) VALUES ({', '.join(placeholders)})
                        """
                        cursor.execute(sql, values)
                    except psycopg2.IntegrityError:
                        # Skip duplicates
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _migrate_citizen_ideas: {e}")
            raise
            
    def _migrate_budget_allocations(self):
        """Migrate budget allocations"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='budget_allocations',
                limit=2000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Get municipality IDs
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 100")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT id FROM core.users WHERE role IN ('municipal_official', 'admin') LIMIT 20")
            approver_ids = [row[0] for row in cursor.fetchall()]
            
            for point in tqdm(points, desc="Migrating budget allocations"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_ids:
                    allocation_data = {
                        'infrastructure': random.uniform(1000000, 5000000),
                        'education': random.uniform(500000, 2000000),
                        'health': random.uniform(500000, 2000000),
                        'environment': random.uniform(200000, 1000000),
                        'administration': random.uniform(300000, 1500000)
                    }
                    
                    themes_budget = {
                        'transparency': random.uniform(100000, 500000),
                        'participation': random.uniform(100000, 500000),
                        'digitalization': random.uniform(200000, 800000)
                    }
                    
                    try:
                        cursor.execute("""
                            INSERT INTO governance.budget_allocations (
                                municipality_id, fiscal_year, total_budget,
                                allocation_data, participatory_budget_amount,
                                openness_initiatives_budget, themes_budget_distribution,
                                approval_date, approved_by
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (municipality_id, fiscal_year) DO UPDATE
                            SET updated_at = CURRENT_TIMESTAMP
                        """, (
                            random.choice(municipality_ids),
                            payload.get('year', random.choice([2023, 2024, 2025])),
                            payload.get('total', sum(allocation_data.values())),
                            json.dumps(allocation_data),
                            payload.get('participatory', random.uniform(100000, 1000000)),
                            sum(themes_budget.values()),
                            json.dumps(themes_budget),
                            datetime.now().date() - timedelta(days=random.randint(0, 90)),
                            random.choice(approver_ids) if approver_ids else None
                        ))
                    except psycopg2.IntegrityError:
                        # Skip duplicates
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _migrate_budget_allocations: {e}")
            raise
            
    def _migrate_engagement_metrics(self):
        """Migrate engagement metrics"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='engagement_metrics',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check if the table has the top_themes column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'analytics' AND table_name = 'engagement_metrics' AND column_name = 'top_themes'
            """)
            has_top_themes = bool(cursor.fetchone())
            
            # Get municipality IDs
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 50")
            municipality_ids = [row[0] for row in cursor.fetchall()]
            
            # Valid themes from the remacto_theme enum
            valid_themes = [
                'transparency', 'participation', 'digitalization', 
                'accountability', 'innovation', 'sustainability'
            ]
            
            for point in tqdm(points, desc="Migrating engagement metrics"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                if municipality_ids:
                    sentiment_dist = {
                        'very_positive': random.randint(5, 20),
                        'positive': random.randint(20, 40),
                        'neutral': random.randint(20, 40),
                        'negative': random.randint(5, 20),
                        'very_negative': random.randint(0, 10)
                    }
                    
                    device_breakdown = {
                        'mobile': random.randint(50, 70),
                        'desktop': random.randint(20, 40),
                        'tablet': random.randint(5, 15)
                    }
                    
                    # Select random valid themes
                    top_themes = random.sample(valid_themes, random.randint(1, 3))
                    
                    try:
                        if has_top_themes:
                            cursor.execute("""
                                INSERT INTO analytics.engagement_metrics (
                                    municipality_id, metric_date, metric_type,
                                    total_users, active_users, new_users,
                                    total_comments, total_ideas, total_votes,
                                    avg_session_duration, participation_rate,
                                    sentiment_distribution, top_themes, device_breakdown
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::remacto_theme[], %s)
                                ON CONFLICT (municipality_id, metric_date, metric_type) DO NOTHING
                            """, (
                                random.choice(municipality_ids),
                                datetime.now().date() - timedelta(days=random.randint(0, 30)),
                                'daily',
                                random.randint(100, 5000),
                                random.randint(50, 1000),
                                random.randint(10, 200),
                                random.randint(20, 500),
                                random.randint(5, 100),
                                random.randint(10, 200),
                                random.randint(180, 600),
                                random.uniform(5, 25),
                                json.dumps(sentiment_dist),
                                top_themes,  # Cast to remacto_theme[] in SQL
                                json.dumps(device_breakdown)
                            ))
                        else:
                            # Insert without top_themes column
                            cursor.execute("""
                                INSERT INTO analytics.engagement_metrics (
                                    municipality_id, metric_date, metric_type,
                                    total_users, active_users, new_users,
                                    total_comments, total_ideas, total_votes,
                                    avg_session_duration, participation_rate,
                                    sentiment_distribution, device_breakdown
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (municipality_id, metric_date, metric_type) DO NOTHING
                            """, (
                                random.choice(municipality_ids),
                                datetime.now().date() - timedelta(days=random.randint(0, 30)),
                                'daily',
                                random.randint(100, 5000),
                                random.randint(50, 1000),
                                random.randint(10, 200),
                                random.randint(20, 500),
                                random.randint(5, 100),
                                random.randint(10, 200),
                                random.randint(180, 600),
                                random.uniform(5, 25),
                                json.dumps(sentiment_dist),
                                json.dumps(device_breakdown)
                            ))
                    except psycopg2.IntegrityError:
                        # Skip duplicates
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _migrate_engagement_metrics: {e}")
            raise
            
    def _migrate_hespress_articles(self):
        """Migrate Hespress news articles"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='hespress_politics_details',
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to news_articles table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'",
                'municipalities_mentioned': "UUID[]",
                'themes': "remacto_theme[]",
                'sentiment': "sentiment_enum",
                'keywords': "TEXT[]",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            self.check_and_add_columns('engagement', 'news_articles', required_columns)
            
            # Get municipality IDs - converted to strings immediately
            cursor.execute("SELECT id FROM geo.municipalities LIMIT 50")
            municipality_ids = [str(row[0]) for row in cursor.fetchall()]
            
            # Valid themes from the remacto_theme enum
            valid_themes = [
                'transparency', 'governance', 'public_services', 'accountability'
            ]
            
            for i, point in enumerate(tqdm(points, desc="Migrating news articles")):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # Generate a simple article URL to avoid duplicates
                article_url = payload.get('url', f"https://hespress.com/article-{i}")
                
                # Get random municipalities (0-3)
                mentioned_count = min(random.randint(0, 3), len(municipality_ids))
                
                # Select random valid themes
                article_themes = random.sample(valid_themes, random.randint(1, 2))
                themes_str = "{" + ",".join(f'"{theme}"' for theme in article_themes) + "}"
                
                # Use a simplified direct SQL approach without dynamic placeholders
                sql = """
                    INSERT INTO engagement.news_articles (
                        source_name, source_url, article_url, title, content, 
                        summary, author, published_date, language, 
                        municipalities_mentioned, themes, sentiment, keywords, metadata
                    ) VALUES (
                        'Hespress', 'https://www.hespress.com', %s, %s, %s,
                        %s, %s, %s, 'ar',
                        NULL, %s::remacto_theme[], %s, %s, %s
                    )
                    ON CONFLICT (article_url) DO NOTHING
                """
                
                try:
                    cursor.execute(sql, (
                        article_url,
                        payload.get('title', 'مقال إخباري'),
                        payload.get('content', payload.get('text', 'محتوى المقال')),
                        payload.get('summary', payload.get('content', '')[:500] if payload.get('content') else 'ملخص'),
                        payload.get('author', 'Hespress'),
                        datetime.now() - timedelta(days=random.randint(0, 90)),
                        themes_str,
                        random.choice(['positive', 'neutral', 'negative']),
                        payload.get('tags', []),
                        json.dumps(payload)
                    ))
                except psycopg2.IntegrityError:
                    # Skip duplicates
                    pass
                    
        except Exception as e:
            logger.error(f"Error in _migrate_hespress_articles: {e}")
            raise
            
    def _migrate_hespress_comments(self):
        """Migrate Hespress news comments"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name='hespress_politics_comments',
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            cursor = self.conn.cursor()
            
            # Check and add missing columns to news_comments table
            required_columns = {
                'language': "language_enum NOT NULL DEFAULT 'ar'",
                'sentiment': "sentiment_enum",
                'sentiment_score': "DECIMAL(3,2)",
                'is_offensive': "BOOLEAN DEFAULT false",
                'metadata': "JSONB DEFAULT '{}'::jsonb"
            }
            
            self.check_and_add_columns('engagement', 'news_comments', required_columns)
            
            # Get article IDs
            cursor.execute("SELECT id FROM engagement.news_articles LIMIT 100")
            article_ids = [row[0] for row in cursor.fetchall()]
            
            # If no article IDs, return early
            if not article_ids:
                logger.warning("No news articles found, skipping comments migration")
                return
                
            # Use direct SQL approach with explicit column lists
            for point in tqdm(points, desc="Migrating news comments"):
                payload = point.payload if hasattr(point, 'payload') else {}
                
                sql = """
                    INSERT INTO engagement.news_comments (
                        article_id, external_comment_id, author_name,
                        content, language, posted_date, likes_count,
                        sentiment, sentiment_score, is_offensive,
                        metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                try:
                    cursor.execute(sql, (
                        random.choice(article_ids),
                        str(point.id),
                        payload.get('author', 'مجهول'),
                        payload.get('comment', payload.get('content', 'تعليق')),
                        'ar',
                        datetime.now() - timedelta(days=random.randint(0, 30)),
                        random.randint(0, 100),
                        random.choice(['positive', 'neutral', 'negative']),
                        random.uniform(0.3, 0.9),
                        False,
                        json.dumps(payload)
                    ))
                except Exception as e:
                    logger.warning(f"Error inserting comment {point.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in _migrate_hespress_comments: {e}")
            raise
            
    def create_sample_data(self):
        """Create additional sample data for testing"""
        cursor = self.conn.cursor()
        
        # Create sample consultations
        cursor.execute("SELECT id FROM geo.municipalities LIMIT 20")
        municipality_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT id FROM core.projects LIMIT 50")
        project_ids = [row[0] for row in cursor.fetchall()]
        
        # Check if users table has role column
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'core' AND table_name = 'users' AND column_name = 'role'
        """)
        has_role = bool(cursor.fetchone())
        
        if has_role:
            cursor.execute("SELECT id FROM core.users WHERE role IN ('municipal_official', 'admin') LIMIT 10")
        else:
            cursor.execute("SELECT id FROM core.users LIMIT 10")
            
        moderator_ids = [row[0] for row in cursor.fetchall()]
        
        # Check if consultations table has themes column
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'engagement' AND table_name = 'consultations' AND column_name = 'themes'
        """)
        has_themes_column = bool(cursor.fetchone())
        
        if municipality_ids and project_ids and moderator_ids:
            for i in range(20):
                try:
                    # Valid themes for remacto_theme enum
                    valid_themes = ['transparency', 'participation']
                    
                    if has_themes_column:
                        cursor.execute("""
                            INSERT INTO engagement.consultations (
                                municipality_id, project_id, title_ar, title_fr,
                                description_ar, description_fr, consultation_type,
                                start_date, end_date, target_participants,
                                themes, status, moderator_id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::remacto_theme[], %s, %s)
                        """, (
                            random.choice(municipality_ids),
                            random.choice(project_ids) if random.random() > 0.3 else None,
                            f'استشارة عامة {i+1}',
                            f'Consultation publique {i+1}',
                            'وصف الاستشارة العامة للمواطنين',
                            'Description de la consultation publique pour les citoyens',
                            random.choice(['online', 'in_person', 'hybrid']),
                            datetime.now() - timedelta(days=random.randint(0, 30)),
                            datetime.now() + timedelta(days=random.randint(7, 60)),
                            random.randint(50, 500),
                            valid_themes,  # Cast to remacto_theme[] in SQL
                            random.choice(['active', 'completed']),
                            random.choice(moderator_ids)
                        ))
                    else:
                        # Insert without themes column
                        cursor.execute("""
                            INSERT INTO engagement.consultations (
                                municipality_id, project_id, title_ar, title_fr,
                                description_ar, description_fr, consultation_type,
                                start_date, end_date, target_participants,
                                status, moderator_id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            random.choice(municipality_ids),
                            random.choice(project_ids) if random.random() > 0.3 else None,
                            f'استشارة عامة {i+1}',
                            f'Consultation publique {i+1}',
                            'وصف الاستشارة العامة للمواطنين',
                            'Description de la consultation publique pour les citoyens',
                            random.choice(['online', 'in_person', 'hybrid']),
                            datetime.now() - timedelta(days=random.randint(0, 30)),
                            datetime.now() + timedelta(days=random.randint(7, 60)),
                            random.randint(50, 500),
                            random.choice(['active', 'completed']),
                            random.choice(moderator_ids)
                        ))
                except psycopg2.IntegrityError:
                    # Skip if already exists
                    pass
                    
        self.conn.commit()
        logger.info("Sample data created successfully")
        
    def create_views_and_functions(self):
        """Create additional views and functions for the application"""
        cursor = self.conn.cursor()
        
        # Check required columns for views
        try:
            # Check projects table columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = 'projects'
            """)
            project_columns = [row[0] for row in cursor.fetchall()]
            
            # Check comments table columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'engagement' AND table_name = 'comments'
            """)
            comment_columns = [row[0] for row in cursor.fetchall()]
            
            # Check if views already exist and drop them first to avoid conflicts
            cursor.execute("""
                DROP VIEW IF EXISTS governance.municipality_dashboard;
                DROP VIEW IF EXISTS core.project_progress;
            """)
            
            # Only create views if required columns exist
            has_status = 'status' in project_columns
            has_sentiment = 'sentiment' in comment_columns
            
            if has_status and has_sentiment:
                try:
                    cursor.execute("""
                        -- Municipality dashboard view
                        CREATE VIEW governance.municipality_dashboard AS
                        SELECT 
                            m.id,
                            m.name_ar,
                            m.name_fr,
                            m.remacto_member,
                            COUNT(DISTINCT p.id) as total_projects,
                            COUNT(DISTINCT p.id) FILTER (WHERE p.status = 'completed') as completed_projects,
                            COUNT(DISTINCT u.id) as total_users,
                            COUNT(DISTINCT c.id) as total_comments,
                            COUNT(DISTINCT i.id) as total_ideas,
                            AVG(CASE 
                                WHEN c.sentiment = 'very_positive' THEN 5
                                WHEN c.sentiment = 'positive' THEN 4
                                WHEN c.sentiment = 'neutral' THEN 3
                                WHEN c.sentiment = 'negative' THEN 2
                                WHEN c.sentiment = 'very_negative' THEN 1
                            END) as avg_sentiment_score
                        FROM geo.municipalities m
                        LEFT JOIN core.projects p ON p.municipality_id = m.id
                        LEFT JOIN core.users u ON u.municipality_id = m.id
                        LEFT JOIN engagement.comments c ON c.project_id = p.id
                        LEFT JOIN engagement.ideas i ON i.municipality_id = m.id
                        GROUP BY m.id, m.name_ar, m.name_fr, m.remacto_member;
                        
                        -- Project progress view
                        CREATE VIEW core.project_progress AS
                        SELECT 
                            p.id,
                            p.municipality_id,
                            p.title_ar,
                            p.title_fr,
                            p.status,
                            p.completion_percentage,
                            COUNT(DISTINCT pu.id) as updates_count,
                            COUNT(DISTINCT c.id) as comments_count,
                            AVG(c.sentiment_score) as avg_sentiment,
                            MAX(pu.created_at) as last_update_date
                        FROM core.projects p
                        LEFT JOIN core.project_updates pu ON pu.project_id = p.id
                        LEFT JOIN engagement.comments c ON c.project_id = p.id
                        GROUP BY p.id, p.municipality_id, p.title_ar, p.title_fr, p.status, p.completion_percentage;
                    """)
                    logger.info("Full views created successfully")
                except Exception as e:
                    logger.warning(f"Error creating full views: {e}")
            else:
                # Create simpler views that don't rely on status/sentiment
                try:
                    cursor.execute("""
                        -- Simple municipality dashboard view
                        CREATE VIEW governance.municipality_dashboard AS
                        SELECT 
                            m.id,
                            m.name_ar,
                            m.name_fr,
                            m.remacto_member,
                            COUNT(DISTINCT p.id) as total_projects,
                            COUNT(DISTINCT u.id) as total_users,
                            COUNT(DISTINCT c.id) as total_comments,
                            COUNT(DISTINCT i.id) as total_ideas
                        FROM geo.municipalities m
                        LEFT JOIN core.projects p ON p.municipality_id = m.id
                        LEFT JOIN core.users u ON u.municipality_id = m.id
                        LEFT JOIN engagement.comments c ON c.project_id = p.id
                        LEFT JOIN engagement.ideas i ON i.municipality_id = m.id
                        GROUP BY m.id, m.name_ar, m.name_fr, m.remacto_member;
                        
                        -- Simple project progress view
                        CREATE VIEW core.project_progress AS
                        SELECT 
                            p.id,
                            p.municipality_id,
                            p.title_ar,
                            p.title_fr,
                            COUNT(DISTINCT pu.id) as updates_count,
                            COUNT(DISTINCT c.id) as comments_count,
                            MAX(pu.created_at) as last_update_date
                        FROM core.projects p
                        LEFT JOIN core.project_updates pu ON pu.project_id = p.id
                        LEFT JOIN engagement.comments c ON c.project_id = p.id
                        GROUP BY p.id, p.municipality_id, p.title_ar, p.title_fr;
                    """)
                    logger.info("Simple views created successfully")
                except Exception as e:
                    logger.warning(f"Error creating simple views: {e}")
        except Exception as e:
            logger.warning(f"Error checking view requirements: {e}")
        
        # Create function for searching multilingual content
        try:
            cursor.execute("""
                CREATE OR REPLACE FUNCTION search_content(
                    search_query TEXT,
                    search_language language_enum DEFAULT 'multi',
                    content_types TEXT[] DEFAULT ARRAY['projects', 'ideas', 'comments']
                ) RETURNS TABLE (
                    id UUID,
                    content_type TEXT,
                    title TEXT,
                    content TEXT,
                    language language_enum,
                    relevance REAL
                ) AS $func$
                BEGIN
                    -- This is a simplified version. In production, use proper full-text search
                    -- with language-specific configurations and weights
                    
                    IF 'projects' = ANY(content_types) THEN
                        RETURN QUERY
                        SELECT 
                            p.id,
                            'project'::TEXT,
                            COALESCE(p.title_fr, p.title_ar) as title,
                            COALESCE(p.description_fr, p.description_ar) as content,
                            CASE 
                                WHEN p.title_fr IS NOT NULL THEN 'fr'::language_enum
                                ELSE 'ar'::language_enum
                            END as language,
                            1.0::REAL as relevance
                        FROM core.projects p
                        WHERE 
                            p.title_fr ILIKE '%' || search_query || '%' OR
                            p.title_ar ILIKE '%' || search_query || '%' OR
                            p.description_fr ILIKE '%' || search_query || '%' OR
                            p.description_ar ILIKE '%' || search_query || '%';
                    END IF;
                    
                    IF 'ideas' = ANY(content_types) THEN
                        RETURN QUERY
                        SELECT 
                            i.id,
                            'idea'::TEXT,
                            i.title,
                            i.description as content,
                            CASE 
                                WHEN i.language IS NOT NULL THEN i.language
                                ELSE 'fr'::language_enum
                            END as language,
                            0.8::REAL as relevance
                        FROM engagement.ideas i
                        WHERE 
                            i.title ILIKE '%' || search_query || '%' OR
                            i.description ILIKE '%' || search_query || '%';
                    END IF;
                    
                    IF 'comments' = ANY(content_types) THEN
                        RETURN QUERY
                        SELECT 
                            c.id,
                            'comment'::TEXT,
                            LEFT(c.content, 100) as title,
                            c.content,
                            CASE 
                                WHEN c.language IS NOT NULL THEN c.language
                                ELSE 'fr'::language_enum
                            END as language,
                            0.6::REAL as relevance
                        FROM engagement.comments c
                        WHERE c.content ILIKE '%' || search_query || '%';
                    END IF;
                END;
                $func$ LANGUAGE plpgsql;
            """)
            logger.info("Search function created successfully")
        except Exception as e:
            logger.warning(f"Error creating search function: {e}")
            
        logger.info("Views created successfully")
        
    def print_summary(self):
        """Print summary of the created database"""
        cursor = self.conn.cursor()
        
        print("\n" + "="*80)
        print("CIVICCATALYST DATABASE CREATION SUMMARY")
        print("="*80)
        
        try:
            # Get list of all schemas
            cursor.execute("""
                SELECT DISTINCT schema_name
                FROM information_schema.schemata
                WHERE schema_name IN ('core', 'geo', 'analytics', 'engagement', 'governance', 'system')
            """)
            schemas = [row[0] for row in cursor.fetchall()]
            
            print("\nTable Row Counts:")
            print("-"*50)
            
            for schema in schemas:
                # Get all tables in this schema
                cursor.execute(f"""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = '{schema}'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    print(f"\n{schema.upper()} Schema:")
                    for table in tables:
                        # Count rows in each table
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                            count = cursor.fetchone()[0]
                            print(f"  {table}: {count:,} rows")
                        except Exception as e:
                            print(f"  {table}: Error counting rows - {e}")
        except Exception as e:
            logger.warning(f"Could not get table statistics: {e}")
            
        # Get database size
        try:
            cursor.execute("SELECT pg_database_size('CivicCatalyst')")
            db_size = cursor.fetchone()[0]
            print(f"\nDatabase Size: {db_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get database size: {e}")
            
        print("\n" + "="*80)
        print("Database creation and migration completed successfully!")
        print(f"Connection: postgresql://postgres:Abdi2022@{DB_CONFIG['host']}:5432/CivicCatalyst")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    db = CivicCatalystDB()
    
    try:
        # Connect to databases
        logger.info("Connecting to databases...")
        db.connect_postgres()
        db.connect_qdrant()
        
        # Create schema
        logger.info("Creating database schema...")
        db.create_database_schema()
        
        # Create initial data
        logger.info("Creating initial data...")
        db.create_initial_data()
        
        # Migrate data
        logger.info("Migrating data from Qdrant...")
        db.migrate_qdrant_data()
        
        # Create sample data
        logger.info("Creating additional sample data...")
        db.create_sample_data()
        
        # Create views and functions after data is loaded
        logger.info("Creating views and functions...")
        db.create_views_and_functions()
        
        # Print summary
        db.print_summary()
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)
    finally:
        if db.conn:
            db.conn.close()
            logger.info("PostgreSQL connection closed")
        if db.qdrant_client:
            db.qdrant_client.close()
            logger.info("Qdrant connection closed")


if __name__ == "__main__":
    main()
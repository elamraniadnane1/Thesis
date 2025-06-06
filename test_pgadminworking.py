#!/usr/bin/env python3
"""
CivicCatalyst Database Testing Script
This script tests the connectivity and functionality of the CivicCatalyst PostgreSQL database
and verifies that pgAdmin is accessible.
"""

import argparse
import psycopg2
import requests
import json
import sys
import time
from datetime import datetime
from tabulate import tabulate
import socket
import uuid

# Configure argument parser
parser = argparse.ArgumentParser(description='Test PostgreSQL and pgAdmin connectivity and functionality')
parser.add_argument('--host', default='154.44.186.241', help='PostgreSQL host address')
parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port')
parser.add_argument('--dbname', default='CivicCatalyst', help='Database name')
parser.add_argument('--user', default='postgres', help='Database user')
parser.add_argument('--password', default='Abdi2022', help='Database password')
parser.add_argument('--pgadmin-url', default='http://154.44.186.241:80', help='pgAdmin URL')
parser.add_argument('--pgadmin-user', default='elamraniadnane1@gmail.com', help='pgAdmin username')
parser.add_argument('--pgadmin-pass', default='Abdi2022', help='pgAdmin password')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

args = parser.parse_args()

# Test results storage
test_results = []
verbose = args.verbose

def log(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level}: {message}")

def add_test_result(test_name, status, message=""):
    """Add a test result to the results list"""
    test_results.append({
        "test_name": test_name,
        "status": status,
        "message": message,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    if verbose or status == "FAIL":
        status_color = "\033[92m" if status == "PASS" else "\033[91m"  # Green for PASS, Red for FAIL
        print(f"{status_color}[{status}]\033[0m {test_name}: {message}")

def test_network_connectivity():
    """Test network connectivity to the PostgreSQL server"""
    try:
        # Try to create a socket connection to the PostgreSQL server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((args.host, args.port))
        sock.close()
        
        if result == 0:
            add_test_result("Network Connectivity", "PASS", f"Port {args.port} is open on {args.host}")
            return True
        else:
            add_test_result("Network Connectivity", "FAIL", f"Port {args.port} is closed on {args.host}")
            return False
    except Exception as e:
        add_test_result("Network Connectivity", "FAIL", f"Error testing connection: {str(e)}")
        return False

def test_postgres_connection():
    """Test connection to PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        add_test_result("PostgreSQL Connection", "PASS", f"Connected successfully. Server version: {version[0]}")
        return True
    except Exception as e:
        add_test_result("PostgreSQL Connection", "FAIL", f"Connection failed: {str(e)}")
        return False

def test_pgadmin_availability():
    """Test if pgAdmin is available"""
    try:
        response = requests.get(args.pgadmin_url, timeout=5)
        if response.status_code == 200:
            add_test_result("pgAdmin Availability", "PASS", "pgAdmin is accessible")
            return True
        else:
            add_test_result("pgAdmin Availability", "FAIL", f"pgAdmin returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        add_test_result("pgAdmin Availability", "FAIL", f"Error accessing pgAdmin: {str(e)}")
        return False

def test_database_schemas():
    """Test that all required schemas exist"""
    expected_schemas = ['core', 'geo', 'analytics', 'engagement', 'governance', 'system']
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN %s
        """, (tuple(expected_schemas),))
        
        existing_schemas = [row[0] for row in cursor.fetchall()]
        missing_schemas = set(expected_schemas) - set(existing_schemas)
        
        if not missing_schemas:
            add_test_result("Database Schemas", "PASS", f"All schemas exist: {', '.join(existing_schemas)}")
        else:
            add_test_result("Database Schemas", "FAIL", f"Missing schemas: {', '.join(missing_schemas)}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        add_test_result("Database Schemas", "FAIL", f"Error checking schemas: {str(e)}")

def test_tables_exist():
    """Test that all expected tables exist in each schema"""
    schema_tables = {
        'geo': ['regions', 'provinces', 'cercles', 'municipalities', 'arrondissements'],
        'core': ['users', 'projects', 'project_updates', 'documents'],
        'engagement': ['consultations', 'comments', 'ideas', 'news_articles', 'news_comments'],
        'analytics': ['sentiment_analysis', 'topic_analysis', 'offensive_content', 'engagement_metrics'],
        'governance': ['municipal_officials', 'budget_allocations', 'pb_proposals'],
        'system': ['activity_logs', 'configurations', 'ai_models', 'notifications']
    }
    
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        
        all_tables_exist = True
        
        for schema, tables in schema_tables.items():
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
            """, (schema,))
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            missing_tables = set(tables) - set(existing_tables)
            
            if not missing_tables:
                add_test_result(f"Tables in {schema}", "PASS", f"All tables exist: {len(existing_tables)}/{len(tables)}")
            else:
                all_tables_exist = False
                add_test_result(f"Tables in {schema}", "FAIL", f"Missing tables: {', '.join(missing_tables)}")
        
        cursor.close()
        conn.close()
        
        return all_tables_exist
    except Exception as e:
        add_test_result("Tables Existence", "FAIL", f"Error checking tables: {str(e)}")
        return False

def test_sample_data():
    """Test that tables have sample data"""
    key_tables = [
        ('geo.municipalities', 'Code, Name (FR), Name (AR)', 'code, name_fr, name_ar'),
        ('core.users', 'Username, Email, Role', 'username, email, role'),
        ('core.projects', 'Title (FR), Status', 'title_fr, status'),
        ('engagement.ideas', 'Title, Description (sample)', 'title, left(description, 50) as description_sample'),
        ('governance.budget_allocations', 'Municipality ID, Fiscal Year, Total Budget', 'municipality_id, fiscal_year, total_budget')
    ]
    
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        
        all_data_present = True
        
        for table, description, columns in key_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Get sample rows
                    cursor.execute(f"SELECT {columns} FROM {table} LIMIT 3")
                    sample = cursor.fetchall()
                    sample_str = "\n".join([str(row) for row in sample])
                    
                    add_test_result(f"Data in {table}", "PASS", f"Found {count} rows. Sample data ({description}):\n{sample_str}")
                else:
                    all_data_present = False
                    add_test_result(f"Data in {table}", "FAIL", f"No data found in {table}")
            except Exception as e:
                all_data_present = False
                add_test_result(f"Data in {table}", "FAIL", f"Error querying {table}: {str(e)}")
        
        cursor.close()
        conn.close()
        
        return all_data_present
    except Exception as e:
        add_test_result("Sample Data", "FAIL", f"Error checking sample data: {str(e)}")
        return False

def test_views_and_functions():
    """Test that views and functions exist and work"""
    views = [
        'governance.municipality_dashboard',
        'core.project_progress'
    ]
    
    functions = [
        'search_content'
    ]
    
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        
        # Test views
        for view in views:
            try:
                cursor.execute(f"SELECT * FROM {view} LIMIT 1")
                if cursor.rowcount > 0:
                    add_test_result(f"View {view}", "PASS", f"View exists and returns data")
                else:
                    add_test_result(f"View {view}", "PASS", f"View exists but returns no data")
            except Exception as e:
                add_test_result(f"View {view}", "FAIL", f"Error querying view: {str(e)}")
        
        # Test functions
        for function in functions:
            try:
                cursor.execute("""
                    SELECT routine_name 
                    FROM information_schema.routines 
                    WHERE routine_type = 'FUNCTION' 
                    AND routine_name = %s
                """, (function,))
                
                if cursor.rowcount > 0:
                    # Try to execute the function
                    if function == 'search_content':
                        cursor.execute("SELECT * FROM search_content('test', 'multi', ARRAY['projects', 'ideas']) LIMIT 5")
                        add_test_result(f"Function {function}", "PASS", f"Function exists and executes successfully")
                else:
                    add_test_result(f"Function {function}", "FAIL", f"Function does not exist")
            except Exception as e:
                add_test_result(f"Function {function}", "FAIL", f"Error testing function: {str(e)}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        add_test_result("Views and Functions", "FAIL", f"Error checking views and functions: {str(e)}")

def test_insert_and_relationships():
    """Test insert capability and foreign key relationships"""
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        cursor = conn.cursor()
        
        # 1. Get a municipality ID
        cursor.execute("SELECT id FROM geo.municipalities LIMIT 1")
        municipality_id = cursor.fetchone()[0]
        
        # 2. Insert a test user
        test_username = f"test_user_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO core.users (
                username, email, password_hash, role, municipality_id, 
                first_name, last_name, preferred_language
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            test_username, 
            f"{test_username}@test.com", 
            "test_password_hash", 
            "citizen", 
            municipality_id,
            "Test",
            "User",
            "fr"
        ))
        user_id = cursor.fetchone()[0]
        
        add_test_result("Insert User", "PASS", f"Successfully inserted test user with ID: {user_id}")
        
        # 3. Insert a test project
        cursor.execute("""
            INSERT INTO core.projects (
                municipality_id, title_ar, title_fr, description_ar, description_fr,
                project_code, status, themes, primary_theme, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            municipality_id,
            "مشروع اختبار",
            "Projet de Test",
            "وصف المشروع التجريبي",
            "Description du projet de test",
            f"TEST-{uuid.uuid4().hex[:8]}",
            "proposed",
            ["transparency", "digitalization"],
            "transparency",
            user_id
        ))
        project_id = cursor.fetchone()[0]
        
        add_test_result("Insert Project", "PASS", f"Successfully inserted test project with ID: {project_id}")
        
        # 4. Insert a test comment
        cursor.execute("""
            INSERT INTO engagement.comments (
                user_id, project_id, content, language, sentiment,
                sentiment_score, is_offensive, moderation_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            user_id,
            project_id,
            "This is a test comment",
            "fr",
            "positive",
            0.8,
            False,
            "approved"
        ))
        comment_id = cursor.fetchone()[0]
        
        add_test_result("Insert Comment", "PASS", f"Successfully inserted test comment with ID: {comment_id}")
        
        # 5. Clean up test data to avoid cluttering the database
        cursor.execute("DELETE FROM engagement.comments WHERE id = %s", (comment_id,))
        cursor.execute("DELETE FROM core.projects WHERE id = %s", (project_id,))
        cursor.execute("DELETE FROM core.users WHERE id = %s", (user_id,))
        
        add_test_result("Foreign Key Relationships", "PASS", "Successfully tested foreign key relationships")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        add_test_result("Insert and Relationships", "FAIL", f"Error testing inserts: {str(e)}")
        return False

def print_summary():
    """Print a summary of all test results"""
    pass_count = sum(1 for result in test_results if result['status'] == 'PASS')
    fail_count = sum(1 for result in test_results if result['status'] == 'FAIL')
    
    print("\n" + "="*80)
    print(f"CIVICCATALYST DATABASE TEST SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Host: {args.host}:{args.port}")
    print(f"Database: {args.dbname}")
    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {pass_count} ({pass_count/len(test_results)*100:.1f}%)")
    print(f"Failed: {fail_count} ({fail_count/len(test_results)*100:.1f}%)")
    print("="*80)
    
    # Create a table of results
    table_data = []
    for result in test_results:
        status_symbol = "✅" if result['status'] == 'PASS' else "❌"
        # Truncate message if it's too long
        message = result['message']
        if len(message) > 60:
            message = message[:57] + "..."
        
        table_data.append([
            status_symbol,
            result['test_name'],
            message
        ])
    
    print(tabulate(table_data, headers=["Status", "Test Name", "Message"]))
    print("="*80)
    
    if fail_count > 0:
        print("\nFAILED TESTS DETAILS:")
        for result in test_results:
            if result['status'] == 'FAIL':
                print(f"\n❌ {result['test_name']}:")
                print(f"   {result['message']}")
    
    print("\n" + "="*80)
    
    # Return True if all tests passed, False otherwise
    return fail_count == 0

def main():
    """Main function to run all tests"""
    log("Starting CivicCatalyst database tests")
    
    # Basic connectivity tests
    if not test_network_connectivity():
        log("Network connectivity test failed. Aborting further tests.", "ERROR")
        return False
    
    if not test_postgres_connection():
        log("PostgreSQL connection test failed. Aborting further tests.", "ERROR")
        return False
    
    # Test pgAdmin availability (optional)
    test_pgadmin_availability()
    
    # Test database structure
    test_database_schemas()
    
    if not test_tables_exist():
        log("Some required tables are missing. Continuing with limited tests.", "WARNING")
    
    # Test data and functionality
    test_sample_data()
    test_views_and_functions()
    test_insert_and_relationships()
    
    # Print summary and return success/failure
    return print_summary()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
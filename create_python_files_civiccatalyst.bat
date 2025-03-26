@echo off
REM Change directory to the target folder
cd /d "C:\Users\DELL\OneDrive\Desktop\Thesis\pages"

REM Create empty Python files
type nul > app.py
type nul > config.py
type nul > data_ingestion.py
type nul > nlp_processing.py
type nul > dashboard_overview.py
type nul > dashboard_sentiment.py
type nul > dashboard_topic_modeling.py
type nul > dashboard_news_documents.py
type nul > dashboard_admin.py
type nul > utils.py

echo Empty Python files have been created in %CD%.
pause

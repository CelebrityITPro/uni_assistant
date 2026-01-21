"""
Configuration file for University RAG Assistant
Centralized settings for all components
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# SCRAPER SETTINGS
# ============================================

UNIVERSITY_NAME = "Conestoga College"

ALLOWED_DOMAINS = [
    "conestogac.on.ca",          # Main site
    "conestogastudents.com",     # Student portal
    # Add more domains as needed:
    # "library.conestogac.on.ca",
    # "careers.conestogac.on.ca",
]

# Pages to start scraping from
START_URLS = [
        "https://www.conestogac.on.ca/",
        "https://www.conestogac.on.ca/future-students/",
        "https://www.conestogac.on.ca/international/",
        "https://www.conestogac.on.ca/admissions/",
        "https://www.conestogac.on.ca/campus-life-and-services/",
        "https://www.conestogac.on.ca/about/",
        "https://www.conestogac.on.ca/programs-and-courses/",

        # Student portal
        "https://www.conestogastudents.com/",
        "https://www.conestogastudents.com/about-us",
        "https://www.conestogastudents.com/getinvolved",
        "https://www.conestogastudents.com/studentlife",
        "https://www.conestogastudents.com/wellness",
        "https://www.conestogastudents.com/representation",
        ]

# Scraping limits
MAX_PAGES = 500
SCRAPE_DELAY = 2  # Seconds between requests

# ============================================
# TEXT PROCESSING SETTINGS
# ============================================

# Chunking parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_CHUNK_SIZE = 1500  # Maximum chunk size

# ============================================
# EMBEDDING SETTINGS
# ============================================

# Embedding model (from sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast
# Alternative: "all-mpnet-base-v2"  # Better quality, slower

EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
# Use 768 if using all-mpnet-base-v2

# ============================================
# VECTOR DATABASE SETTINGS
# ============================================

# ChromaDB collection name
COLLECTION_NAME = "university_knowledge"

# Retrieval parameters
TOP_K_RESULTS = 5  # Number of chunks to retrieve for each query

# ============================================
# LLM SETTINGS
# ============================================

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # Change to "mistral:7b" for better quality

# Generation parameters
TEMPERATURE = 0.3  # Lower = more focused answers
MAX_TOKENS = 500  # Maximum response length
TOP_P = 0.9

# System prompt template
SYSTEM_PROMPT = f"""You are a helpful virtual assistant for {UNIVERSITY_NAME}.

Your role is to answer questions about the university based ONLY on the provided context.

Guidelines:
- Be friendly and helpful
- Answer only based on the context provided
- If the context doesn't contain the answer, say "I don't have that information in my knowledge base. I recommend contacting the admissions office at [contact info]."
- Always cite your sources by mentioning which page or section the information came from
- Be concise but complete
- Use bullet points for lists when appropriate

Context:
{{context}}

Question: {{question}}

Answer:"""

# ============================================
# CONVERSATION SETTINGS
# ============================================

# Maximum conversation history to maintain
MAX_CONVERSATION_HISTORY = 10

# ============================================
# STREAMLIT UI SETTINGS
# ============================================

APP_TITLE = f"{UNIVERSITY_NAME} Virtual Assistant"
APP_DESCRIPTION = "Ask me anything about the university!"

# UI Colors (optional)
PRIMARY_COLOR = "#1f77b4"
BACKGROUND_COLOR = "#ffffff"

# ============================================
# SCHEDULING SETTINGS
# ============================================

# How often to re-scrape the website
SCRAPE_SCHEDULE = "weekly"  # Options: daily, weekly, monthly

# ============================================
# LOGGING SETTINGS
# ============================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = PROJECT_ROOT / "app.log"
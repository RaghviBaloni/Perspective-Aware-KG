"""
Configuration file for TACO Perspective Annotation Pipeline
"""

import os
from pathlib import Path

# ======================================
# PATHS
# ======================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TACO_DATA_DIR = DATA_DIR / "TACO"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, TACO_DATA_DIR, PROCESSED_DATA_DIR, 
                  ANNOTATIONS_DIR, LOG_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =================================
# DATA FILES
# =================================

# Input files
CONVERSATIONS_FILE = TACO_DATA_DIR / "conversations.csv"
WORKER_DECISIONS_FILE = TACO_DATA_DIR / "worker_decisions.csv"
MAJORITY_VOTES_FILE = TACO_DATA_DIR / "majority_votes.csv"
BACKUP_TWEETS_FILE = TACO_DATA_DIR / "backup_tweets.csv"

# Processed files
PREPROCESSED_TWEETS_FILE = PROCESSED_DATA_DIR / "preprocessed_tweets.csv"
URL_METADATA_FILE = PROCESSED_DATA_DIR / "url_metadata.json"
DATA_STATISTICS_FILE = RESULTS_DIR / "data_statistics.json"

# =================================
# TACO DATASET PARAMETERS
# =================================

# TACO class labels
TACO_CLASSES = ['Statement', 'Reason', 'Notification', 'None', 'Undecided']

# Worker IDs (from dataset description)
WORKERS = ['A', 'B', 'C', 'D', 'E', 'F']  # A and E are same person

# ====================================
# PREPROCESSING PARAMETERS
# ====================================

# Text cleaning
MIN_TWEET_LENGTH = 5  # Minimum characters after cleaning
MAX_TWEET_LENGTH = 500  # Twitter's limit (conservative)

# URL handling
URL_PATTERNS = {
    'standard': r'https?://[^\s]+',
    'shortened': r'(?:https?://)?(?:t\.co|bit\.ly|tinyurl\.com)/[^\s]+',
}

# URL placeholders (if you want to replace URLs during cleaning)
URL_PLACEHOLDER = "[URL]"
MENTION_PLACEHOLDER = "[USER]"

# Whether to keep or replace
KEEP_URLS = True  # Set False to replace with placeholder
KEEP_MENTIONS = True  # Set False to replace with placeholder

# ====================================
# URL CLASSIFICATION
# ====================================

# Domain categories for URL classification
MEDIA_EXTENSIONS = ['.gif', '.jpg', '.jpeg', '.png', '.mp4', '.webm', '.mov']

SOCIAL_DOMAINS = [
    'twitter.com', 't.co', 'facebook.com', 'instagram.com', 
    'reddit.com', 'linkedin.com', 'tiktok.com'
]

NEWS_DOMAINS = [
    'nytimes.com', 'bbc.com', 'cnn.com', 'reuters.com', 
    'apnews.com', 'theguardian.com', 'washingtonpost.com',
    'bloomberg.com', 'wsj.com', 'foxnews.com', 'nbcnews.com',
    'abcnews.com', 'usatoday.com', 'npr.org'
]

ACADEMIC_DOMAINS = [
    'arxiv.org', 'doi.org', 'pubmed.gov', 'scholar.google.com',
    'researchgate.net', 'academia.edu', 'sciencedirect.com',
    'jstor.org', 'ieee.org', 'acm.org'
]

SHORTENED_DOMAINS = [
    't.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly',
    'buff.ly', 'is.gd', 'tiny.cc'
]

# ==========================
# VLLM PARAMETERS
# ==========================

# Model configuration
VLLM_MODELS = {
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'mistral-large': 'mistralai/Mistral-Large-Instruct-2407',
    'qwen-2.5-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'llama-3.1-70b-awq': 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
}

# Default model for Pass 1
DEFAULT_MODEL = 'llama-3.1-70b-awq'

# Inference parameters
INFERENCE_PARAMS = {
    'temperature': 0.3,
    'top_p': 0.95,
    'max_tokens': 1500,
    'stop': ['\n}', '}\n', '#'],
}

# Batching
BATCH_SIZE = 16  # Number of tweets per batch
MAX_CONCURRENT_REQUESTS = 4  # For URL fetching

# ===============================
# ANNOTATION PARAMETERS
# ===============================

# Perspective labels
PERSPECTIVE_LABELS = [
    'Information Providing',
    'Information Seeking',
    'Epistemic/Factual',
    'Experiential',
    'Opinion/Belief',
    'AOT'
]

# Confidence thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
LOW_CONFIDENCE_THRESHOLD = 0.5  # Flag annotations below this

# Retry logic
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# =======================
# LOGGING
# =======================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "pipeline.log"

# ==========================
# PASS 1 vs PASS 2 PARAMETERS
# ==========================

# Pass 1: No URL fetching
PASS_1_FETCH_URLS = False
PASS_1_OUTPUT_FILE = ANNOTATIONS_DIR / "pass1_annotations.json"

# Pass 2: Selective URL fetching
PASS_2_FETCH_URLS = True
PASS_2_OUTPUT_FILE = ANNOTATIONS_DIR / "pass2_annotations.json"
PASS_2_SAMPLE_SIZE = 500  # Number of tweets to re-annotate with URLs

# ============================
# VALIDATION PARAMETERS
# ============================

# Evidence span validation
STRICT_EVIDENCE_MATCHING = True  # Require exact substring match
ALLOW_WHITESPACE_DIFF = True  # Allow minor whitespace differences

# Output validation
REQUIRE_ALL_FIELDS = True  # All fields must be present
VALIDATE_CONFIDENCE_RANGE = True  # Confidence must be in [0, 1]

# =============================
# ANALYSIS PARAMETERS
# =============================

# Statistical tests
SIGNIFICANCE_LEVEL = 0.05  # For hypothesis tests

# Sampling for manual review
MANUAL_REVIEW_SAMPLE_SIZE = 100

# Inter-rater reliability
KAPPA_INTERPRETATION = {
    'poor': (0.0, 0.20),
    'fair': (0.20, 0.40),
    'moderate': (0.40, 0.60),
    'substantial': (0.60, 0.80),
    'almost_perfect': (0.80, 1.00)
}

# ==============================
# HELPER FUNCTIONS
# ==============================

def get_annotation_output_file(model_name, pass_number=1):
    """Generate output filename for annotations"""
    return ANNOTATIONS_DIR / f"pass{pass_number}_{model_name}_annotations.json"

def get_log_file(script_name):
    """Generate log filename for specific script"""
    return LOG_DIR / f"{script_name}.log"

# ================================
# VALIDATION
# ================================

def validate_config():
    """Validate that all required files exist"""
    required_files = [
        CONVERSATIONS_FILE,
        WORKER_DECISIONS_FILE,
        MAJORITY_VOTES_FILE,
        BACKUP_TWEETS_FILE
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print("WARNING: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("All required data files found")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("TACO PERSPECTIVE ANNOTATION - CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Logs Directory: {LOG_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\n Default Model: {DEFAULT_MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Pass 1 Fetch URLs: {PASS_1_FETCH_URLS}")
    print(f"Pass 2 Fetch URLs: {PASS_2_FETCH_URLS}")
    print("\n" + "=" * 70)
    validate_config()

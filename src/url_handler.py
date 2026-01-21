"""
URL Handler - Extraction, Classification, and Resolution
Pass 1: URL extraction and classification only (no fetching)
Pass 2: Add fetching capabilities
"""

import re
import logging
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
import time
import pandas as pd

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import *

logger = logging.getLogger(__name__)


class URLHandler:
    """
    Handles URL extraction, classification, and (optionally) fetching
    """
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # For detecting t.co and other shortened URLs
        self.shortened_pattern = re.compile(
            r'(?:https?://)?(?:' + '|'.join(SHORTENED_DOMAINS) + r')/\S+'
        )
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs from text
        
        Args:
            text: Tweet text
            
        Returns:
            List of URLs found
        """
        if not text or pd.isna(text):
            return []
        
        urls = self.url_pattern.findall(text)
        return urls
    
    def extract_domain(self, url: str) -> str:
        """
        Extract domain from URL
        
        Args:
            url: Full URL
            
        Returns:
            Domain name (e.g., 'twitter.com')
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except Exception as e:
            logger.debug(f"Error parsing domain from {url}: {e}")
            return ""
    
    def classify_url_type(self, url: str) -> str:
        """
        Classify URL type without fetching
        
        Args:
            url: Full URL
            
        Returns:
            URL type: 'media', 'social', 'news', 'academic', 'shortened', 'other'
        """
        # Check for media extensions
        url_lower = url.lower()
        for ext in MEDIA_EXTENSIONS:
            if url_lower.endswith(ext):
                return 'media'
        
        # Extract domain
        domain = self.extract_domain(url)
        
        if not domain:
            return 'other'
        
        # Check domain categories
        if any(d in domain for d in SHORTENED_DOMAINS):
            return 'shortened'
        elif any(d in domain for d in SOCIAL_DOMAINS):
            return 'social'
        elif any(d in domain for d in NEWS_DOMAINS):
            return 'news'
        elif any(d in domain for d in ACADEMIC_DOMAINS):
            return 'academic'
        else:
            return 'other'
    
    def is_shortened_url(self, url: str) -> bool:
        """Check if URL is a shortened URL (t.co, bit.ly, etc.)"""
        return bool(self.shortened_pattern.search(url))
    
    def should_fetch_url(
        self, 
        url_type: str, 
        taco_class: str, 
        tweet_text: str
    ) -> bool:
        """
        Decide if URL should be fetched based on heuristics
        (For Pass 2 implementation)
        
        Args:
            url_type: Type from classify_url_type()
            taco_class: TACO label (statement, reason, notification, none)
            tweet_text: Full tweet text
            
        Returns:
            Boolean indicating whether to fetch
        """
        # Never fetch media or social URLs
        if url_type in ['media', 'social']:
            return False
        
        # Always fetch for reason tweets (likely evidential)
        if taco_class == 'reason':
            return True
        
        # For statements, check for epistemic markers
        epistemic_markers = [
            'study', 'studies', 'research', 'data', 'report', 'reports',
            'according to', 'shows that', 'evidence', 'findings',
            'analysis', 'statistics', 'survey', 'poll'
        ]
        
        if taco_class == 'statement':
            text_lower = tweet_text.lower()
            if any(marker in text_lower for marker in epistemic_markers):
                return True
            else:
                return False
        
        # For notification, only fetch news/academic
        if taco_class == 'notification':
            return url_type in ['news', 'academic']
        
        # For none, skip fetching
        if taco_class == 'none':
            return False
        
        # Default: don't fetch
        return False
    
    def analyze_tweet_urls(
        self, 
        tweet_text: str, 
        taco_class: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive URL analysis for a single tweet
        
        Args:
            tweet_text: Tweet text
            taco_class: Optional TACO class label
            
        Returns:
            Dictionary with URL metadata
        """
        urls = self.extract_urls(tweet_text)
        
        if not urls:
            return {
                'has_urls': False,
                'url_count': 0,
                'urls': []
            }
        
        url_metadata = []
        for url in urls:
            url_type = self.classify_url_type(url)
            domain = self.extract_domain(url)
            is_shortened = self.is_shortened_url(url)
            
            # Determine if should fetch (for Pass 2)
            should_fetch = False
            if taco_class:
                should_fetch = self.should_fetch_url(url_type, taco_class, tweet_text)
            
            url_metadata.append({
                'url': url,
                'type': url_type,
                'domain': domain,
                'is_shortened': is_shortened,
                'should_fetch': should_fetch
            })
        
        return {
            'has_urls': True,
            'url_count': len(urls),
            'urls': url_metadata
        }
    
    def remove_urls(self, text: str, placeholder: str = "[URL]") -> str:
        """
        Remove URLs from text and replace with placeholder
        
        Args:
            text: Original text
            placeholder: Replacement string
            
        Returns:
            Text with URLs replaced
        """
        return self.url_pattern.sub(placeholder, text)
    
    def extract_text_without_urls(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract text and URLs separately
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (text_without_urls, list_of_urls)
        """
        urls = self.extract_urls(text)
        text_clean = self.url_pattern.sub('', text)
        
        # Clean up extra whitespace
        text_clean = ' '.join(text_clean.split())
        
        return text_clean, urls


# Placeholder for Pass 2: URL Content Fetcher
class URLContentFetcher:
    """
    Fetches and extracts content from URLs
    (To be implemented for Pass 2)
    """
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        logger.info("URLContentFetcher initialized (Pass 2 functionality)")
    
    def resolve_shortened_url(self, url: str) -> str:
        """
        Resolve shortened URL to final destination
        (Placeholder for Pass 2)
        """
        # TODO: Implement URL resolution
        logger.debug(f"URL resolution not yet implemented: {url}")
        return url
    
    def fetch_url_content(self, url: str, max_length: int = 500) -> Optional[Dict]:
        """
        Fetch and extract content from URL
        (Placeholder for Pass 2)
        
        Args:
            url: URL to fetch
            max_length: Maximum content length to extract
            
        Returns:
            Dictionary with title, summary, content_type or None
        """
        # TODO: Implement using newspaper3k or trafilatura
        logger.debug(f"URL fetching not yet implemented: {url}")
        return None
    
    def check_url_alive(self, url: str, timeout: int = 3) -> bool:
        """
        Check if URL is accessible
        (Placeholder for Pass 2)
        """
        # TODO: Implement HEAD request
        logger.debug(f"URL checking not yet implemented: {url}")
        return True


def test_url_handler():
    """Test URL handler functionality"""
    print("="*70)
    print("TESTING URL HANDLER")
    print("="*70)
    
    handler = URLHandler()
    
    # Test cases
    test_tweets = [
        {
            'text': 'Check this out: https://www.nytimes.com/article/climate-change',
            'class': 'statement',
            'expected_type': 'news'
        },
        {
            'text': 'Study shows important findings https://arxiv.org/abs/1234.5678',
            'class': 'reason',
            'expected_type': 'academic'
        },
        {
            'text': 'LOL 😂 https://t.co/abc123def',
            'class': 'none',
            'expected_type': 'shortened'
        },
        {
            'text': 'Check out this GIF https://example.com/funny.gif',
            'class': 'notification',
            'expected_type': 'media'
        },
        {
            'text': 'No URLs here just plain text',
            'class': 'statement',
            'expected_type': None
        }
    ]
    
    for i, test in enumerate(test_tweets, 1):
        print(f"\nTest {i}:")
        print(f"  Text: {test['text'][:50]}...")
        print(f"  Class: {test['class']}")
        
        # Analyze URLs
        analysis = handler.analyze_tweet_urls(test['text'], test['class'])
        
        print(f"  Has URLs: {analysis['has_urls']}")
        print(f"  URL Count: {analysis['url_count']}")
        
        if analysis['has_urls']:
            for url_info in analysis['urls']:
                print(f"    - URL: {url_info['url'][:40]}...")
                print(f"      Type: {url_info['type']}")
                print(f"      Domain: {url_info['domain']}")
                print(f"      Shortened: {url_info['is_shortened']}")
                print(f"      Should Fetch: {url_info['should_fetch']}")
                
                # Verify expected type
                if test['expected_type']:
                    if url_info['type'] == test['expected_type']:
                        print(f"      ✓ Type classification correct!")
                    else:
                        print(f"      ✗ Expected {test['expected_type']}, got {url_info['type']}")
    
    print("\n" + "="*70)
    print("URL HANDLER TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Setup logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_url_handler()
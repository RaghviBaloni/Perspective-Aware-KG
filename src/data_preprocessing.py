"""
Tweet Preprocessing Pipeline
Cleans tweets, handles URLs, extracts features, prepares for annotation
"""

import pandas as pd
import numpy as np
import re
import html
import logging
from typing import Dict, List, Tuple
import json

# Import configuration and URL handler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import *
from url_handler import URLHandler

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(get_log_file('data_preprocessing')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TweetPreprocessor:
    """
    Comprehensive tweet preprocessing for perspective annotation
    """
    
    def __init__(self, keep_urls=True, keep_mentions=True):
        self.keep_urls = keep_urls
        self.keep_mentions = keep_mentions
        self.url_handler = URLHandler()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'cleaned': 0,
            'too_short': 0,
            'too_long': 0,
            'missing_text': 0,
            'html_decoded': 0,
            'urls_found': 0,
            'mentions_found': 0,
            'hashtags_found': 0
        }
    
    def decode_html_entities(self, text: str) -> str:
        """
        Decode HTML entities (&amp; → &, &lt; → <, etc.)
        
        Args:
            text: Raw tweet text
            
        Returns:
            Decoded text
        """
        if not text or pd.isna(text):
            return text
        
        try:
            decoded = html.unescape(text)
            if decoded != text:
                self.stats['html_decoded'] += 1
            return decoded
        except Exception as e:
            logger.warning(f"HTML decode error: {e}")
            return text
    
    def clean_whitespace(self, text: str) -> str:
        """
        Normalize whitespace (tabs, newlines, multiple spaces)
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        if not text or pd.isna(text):
            return text
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def handle_retweets(self, text: str) -> str:
        """
        Handle RT indicators (keep or remove)
        
        Args:
            text: Tweet text
            
        Returns:
            Processed text
        """
        if not text or pd.isna(text):
            return text
        
        # Pattern: "RT @username: " at start
        rt_pattern = r'^RT\s+@\w+:\s*'
        
        if re.match(rt_pattern, text):
            # Remove RT prefix but keep the content
            text = re.sub(rt_pattern, '', text)
        
        return text
    
    def handle_urls(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Handle URLs based on configuration
        
        Args:
            text: Tweet text
            
        Returns:
            Tuple of (processed_text, url_metadata)
        """
        if not text or pd.isna(text):
            return text, []
        
        # Extract URL metadata
        url_analysis = self.url_handler.analyze_tweet_urls(text)
        
        if url_analysis['has_urls']:
            self.stats['urls_found'] += 1
        
        # Process text based on configuration
        if self.keep_urls:
            processed_text = text
        else:
            processed_text = self.url_handler.remove_urls(text, URL_PLACEHOLDER)
        
        return processed_text, url_analysis.get('urls', [])
    
    def handle_mentions(self, text: str) -> str:
        """
        Handle @mentions based on configuration
        
        Args:
            text: Tweet text
            
        Returns:
            Processed text
        """
        if not text or pd.isna(text):
            return text
        
        # Count mentions
        mentions = re.findall(r'@\w+', text)
        if mentions:
            self.stats['mentions_found'] += 1
        
        # Process based on configuration
        if not self.keep_mentions:
            text = re.sub(r'@\w+', MENTION_PLACEHOLDER, text)
        
        return text
    
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags (keep in text, just track them)
        
        Args:
            text: Tweet text
            
        Returns:
            List of hashtags
        """
        if not text or pd.isna(text):
            return []
        
        hashtags = re.findall(r'#\w+', text)
        if hashtags:
            self.stats['hashtags_found'] += 1
        
        return hashtags
    
    def validate_tweet_length(self, text: str) -> Tuple[bool, str]:
        """
        Validate tweet length meets requirements
        
        Args:
            text: Cleaned tweet text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not text or pd.isna(text):
            return False, 'missing_text'
        
        length = len(text)
        
        if length < MIN_TWEET_LENGTH:
            return False, 'too_short'
        elif length > MAX_TWEET_LENGTH:
            return False, 'too_long'
        else:
            return True, 'valid'
    
    def clean_single_tweet(self, text: str) -> str:
        """
        Apply all cleaning steps to a single tweet
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return text
        
        # Step 1: Decode HTML entities
        text = self.decode_html_entities(text)
        
        # Step 2: Handle retweets
        text = self.handle_retweets(text)
        
        # Step 3: Handle mentions
        text = self.handle_mentions(text)
        
        # Step 4: Normalize whitespace
        text = self.clean_whitespace(text)
        
        return text
    
    def preprocess_tweet(
        self, 
        tweet_id: str,
        text: str, 
        taco_class: str,
        topic: str,
        confidence: float
    ) -> Dict:
        """
        Complete preprocessing pipeline for a single tweet
        
        Args:
            tweet_id: Tweet ID
            text: Raw tweet text
            taco_class: TACO class label
            topic: Topic label
            confidence: Annotation confidence
            
        Returns:
            Dictionary with processed tweet and metadata
        """
        self.stats['total_processed'] += 1
        
        # Store original
        original_text = text
        
        # Extract hashtags before cleaning
        hashtags = self.extract_hashtags(text)
        
        # Clean text (but keep URLs initially)
        cleaned_text = self.clean_single_tweet(text)
        
        # Handle URLs and get metadata
        if self.keep_urls:
            processed_text = cleaned_text
            url_analysis = self.url_handler.analyze_tweet_urls(
                cleaned_text, 
                taco_class
            )
        else:
            processed_text, url_metadata = self.handle_urls(cleaned_text)
            url_analysis = {
                'has_urls': len(url_metadata) > 0,
                'url_count': len(url_metadata),
                'urls': url_metadata
            }
        
        # Validate length
        is_valid, validation_reason = self.validate_tweet_length(processed_text)
        
        if not is_valid:
            self.stats[validation_reason] += 1
        else:
            self.stats['cleaned'] += 1
        
        # Build output dictionary
        result = {
            'tweet_id': tweet_id,
            'text_original': original_text,
            'text_cleaned': processed_text,
            'taco_class': taco_class,
            'topic': topic,
            'confidence': confidence,
            'is_valid': is_valid,
            'validation_reason': validation_reason,
            'text_length': len(processed_text) if processed_text else 0,
            'word_count': len(processed_text.split()) if processed_text else 0,
            'has_urls': url_analysis['has_urls'],
            'url_count': url_analysis['url_count'],
            'url_metadata': url_analysis.get('urls', []),
            'hashtags': hashtags,
            'hashtag_count': len(hashtags)
        }
        
        return result
    
    def preprocess_dataset(self, input_file: Path) -> pd.DataFrame:
        """
        Preprocess entire TACO dataset
        
        Args:
            input_file: Path to merged TACO data CSV
            
        Returns:
            DataFrame with preprocessed tweets
        """
        logger.info(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        logger.info(f"Preprocessing {len(df)} tweets...")
        
        # Process each tweet
        processed_tweets = []
        
        for idx, row in df.iterrows():
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processed {idx + 1}/{len(df)} tweets...")
            
            result = self.preprocess_tweet(
                tweet_id=str(row['tweet_id']),
                text=row['text'],
                taco_class=row['class'],
                topic=row['topic'],
                confidence=row['confidence']
            )
            
            processed_tweets.append(result)
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_tweets)
        
        logger.info("Preprocessing complete!")
        
        return processed_df
    
    def print_statistics(self):
        """Print preprocessing statistics"""
        print("\n" + "="*70)
        print("PREPROCESSING STATISTICS")
        print("="*70)
        print(f"  Total processed:     {self.stats['total_processed']}")
        print(f"  Successfully cleaned: {self.stats['cleaned']}")
        print(f"  Missing text:        {self.stats['missing_text']}")
        print(f"  Too short:           {self.stats['too_short']}")
        print(f"  Too long:            {self.stats['too_long']}")
        print(f"  HTML decoded:        {self.stats['html_decoded']}")
        print(f"  Tweets with URLs:    {self.stats['urls_found']}")
        print(f"  Tweets with mentions: {self.stats['mentions_found']}")
        print(f"  Tweets with hashtags: {self.stats['hashtags_found']}")
        print("="*70)
        
        # Calculate percentages
        total = self.stats['total_processed']
        if total > 0:
            valid_pct = (self.stats['cleaned'] / total) * 100
            print(f"\n✓ Valid tweets: {valid_pct:.1f}%")
            
            if self.stats['too_short'] > 0:
                short_pct = (self.stats['too_short'] / total) * 100
                print(f"⚠️  Too short: {short_pct:.1f}%")
            
            if self.stats['too_long'] > 0:
                long_pct = (self.stats['too_long'] / total) * 100
                print(f"⚠️  Too long: {long_pct:.1f}%")
    
    def save_url_metadata(self, processed_df: pd.DataFrame):
        """
        Save URL metadata separately for Pass 2 analysis
        
        Args:
            processed_df: Processed DataFrame
        """
        logger.info("Extracting URL metadata...")
        
        # Filter tweets with URLs
        tweets_with_urls = processed_df[processed_df['has_urls'] == True].copy()
        
        if len(tweets_with_urls) == 0:
            logger.warning("No tweets with URLs found")
            return
        
        # Build URL metadata structure
        url_metadata = {}
        
        for _, row in tweets_with_urls.iterrows():
            tweet_id = row['tweet_id']
            url_metadata[tweet_id] = {
                'taco_class': row['taco_class'],
                'topic': row['topic'],
                'text': row['text_cleaned'],
                'url_count': row['url_count'],
                'urls': row['url_metadata']
            }
        
        # Save to JSON
        with open(URL_METADATA_FILE, 'w') as f:
            json.dump(url_metadata, f, indent=2)
        
        logger.info(f"✓ URL metadata saved to {URL_METADATA_FILE}")
        logger.info(f"  Total tweets with URLs: {len(url_metadata)}")
        
        # Print URL type distribution
        url_types = {}
        for tweet_data in url_metadata.values():
            for url_info in tweet_data['urls']:
                url_type = url_info['type']
                url_types[url_type] = url_types.get(url_type, 0) + 1
        
        print("\nURL TYPE DISTRIBUTION:")
        for url_type, count in sorted(url_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {url_type:15s}: {count:4d}")


def main():
    """Main preprocessing pipeline"""
    logger.info("="*70)
    logger.info("STARTING TWEET PREPROCESSING")
    logger.info("="*70)
    
    # Check if merged data exists
    merged_file = PROCESSED_DATA_DIR / "merged_taco_data.csv"
    if not merged_file.exists():
        logger.error(f"Merged data file not found: {merged_file}")
        logger.error("Please run data_stats.py first!")
        return 1
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor(
        keep_urls=KEEP_URLS,
        keep_mentions=KEEP_MENTIONS
    )
    
    # Preprocess dataset
    processed_df = preprocessor.preprocess_dataset(merged_file)
    
    # Print statistics
    preprocessor.print_statistics()
    
    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {PREPROCESSED_TWEETS_FILE}...")
    processed_df.to_csv(PREPROCESSED_TWEETS_FILE, index=False)
    logger.info("Preprocessed data saved")
    
    # Save URL metadata for Pass 2
    preprocessor.save_url_metadata(processed_df)
    
    # Print summary by class
    print("\n" + "="*70)
    print("VALID TWEETS BY CLASS")
    print("="*70)
    valid_df = processed_df[processed_df['is_valid'] == True]
    class_counts = valid_df['taco_class'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls:15s}: {count:4d}")
    print("="*70)
    
    logger.info("="*70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*70)
    
    print("\nPreprocessing complete! Ready for annotation.")
    print(f"Preprocessed data: {PREPROCESSED_TWEETS_FILE}")
    print(f"URL metadata: {URL_METADATA_FILE}")
    print(f"Log file: {get_log_file('data_preprocessing')}")
    
    return 0


if __name__ == "__main__":
    exit(main())
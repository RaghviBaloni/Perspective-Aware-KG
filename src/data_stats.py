"""
TACO Dataset Exploration Script
Loads all CSV files, merges them, and generates comprehensive statistics
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
import logging

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(get_log_file('taco_stats')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TACOExplorer:
    """
    Comprehensive exploration of TACO dataset
    """
    
    def __init__(self):
        self.conversations = None
        self.worker_decisions = None
        self.majority_votes = None
        self.backup_tweets = None
        self.merged_data = None
        self.statistics = {}
    
    def load_data(self):
        """Load all TACO CSV files"""
        logger.info("Loading TACO dataset files...")
        
        try:
            # Load conversations
            self.conversations = pd.read_csv(CONVERSATIONS_FILE, dtype={"tweet_id": str})
            logger.info(f"Loaded conversations: {len(self.conversations)} rows")
            
            # Load worker decisions
            self.worker_decisions = pd.read_csv(WORKER_DECISIONS_FILE, dtype={"tweet_id": str})
            logger.info(f"Loaded worker decisions: {len(self.worker_decisions)} rows")
            
            # Load majority votes (ground truth)
            self.majority_votes = pd.read_csv(MAJORITY_VOTES_FILE, dtype={"tweet_id": str}, keep_default_na=False, na_values=[''])
            logger.info(f"Loaded majority votes: {len(self.majority_votes)} rows")
            #print("="*70)
            df = pd.read_csv(MAJORITY_VOTES_FILE)
            print("CLASS COLUMN DIAGNOSTICS")
            print("="*70)

            #Added to preprocess None class
            self.majority_votes['class'] = (
                self.majority_votes['class']
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({'nan': 'None'})
            )


            # Show unique values (including whitespace/case issues)
            print("\nUnique class values (with repr to show hidden chars):")
            for val in df['class'].unique():
                print(f"  {repr(val)}")

            # Count including NaN
            print("\nValue counts (dropna=False):")
            print(df['class'].value_counts(dropna=False))

            # Check for None as string vs Python None
            print(f"\nPython None count: {df['class'].isna().sum()}")
            print(f"String 'None' count: {(df['class'] == 'None').sum()}")
            print(f"String 'none' count: {(df['class'] == 'none').sum()}")

            print("="*70)

            #Load the backup tweets
            self.backup_tweets = pd.read_csv( BACKUP_TWEETS_FILE, engine="python", quotechar='"', encoding="utf-8", dtype={"tweet_id": str} ) 
            logger.info(f"✓ Loaded backup tweets: {len(self.backup_tweets)} rows")
            
            '''# Load backup tweets (actual text)
            self.backup_tweets = pd.read_csv(BACKUP_TWEETS_FILE)
            logger.info(f"✓ Loaded backup tweets: {len(self.backup_tweets)} rows")
            '''
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"✗ Missing file: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            return False
    
    def merge_data(self):
        """Merge all datasets on tweet_id"""
        logger.info("Merging datasets...")
        
        # Start with majority votes (ground truth)
        self.merged_data = self.majority_votes.copy()
        
        # Merge with backup tweets to get text
        self.merged_data = self.merged_data.merge(
            self.backup_tweets,
            on='tweet_id',
            how='left'
        )
        
        # Merge with conversations to get thread structure
        self.merged_data = self.merged_data.merge(
            self.conversations[['tweet_id', 'conversation_id', 'parent_id']],
            on='tweet_id',
            how='left'
        )
        
        logger.info(f"✓ Merged dataset: {len(self.merged_data)} tweets")
        
        # Check for missing text
        missing_text = self.merged_data['text'].isna().sum()
        if missing_text > 0:
            logger.warning(f"{missing_text} tweets missing text")
        
        return self.merged_data
    
    def analyze_class_distribution(self):
        """Analyze TACO class label distribution"""
        logger.info("Analyzing class distribution...")
        
        class_counts = self.merged_data['class'].value_counts()
        class_percentages = self.merged_data['class'].value_counts(normalize=True) * 100
        
        self.statistics['class_distribution'] = {
            'counts': class_counts.to_dict(),
            'percentages': class_percentages.to_dict()
        }
        
        # Print summary
        print("\n" + "="*70)
        print("TACO CLASS DISTRIBUTION")
        print("="*70)
        for cls in TACO_CLASSES:
            if cls in class_counts:
                count = class_counts[cls]
                pct = class_percentages[cls]
                print(f"  {cls:15s}: {count:5d} ({pct:5.2f}%)")
        print("="*70)
        
        # Check class balance
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class
        
        self.statistics['class_imbalance'] = {
            'min_count': int(min_class),
            'max_count': int(max_class),
            'imbalance_ratio': float(imbalance_ratio)
        }
        
        if imbalance_ratio > 3:
            logger.warning(f"Class imbalance detected: {imbalance_ratio:.2f}x")
        
        return class_counts
    
    def analyze_text_properties(self):
        """Analyze tweet text properties"""
        logger.info("Analyzing text properties...")
        
        # Remove rows with missing text
        valid_texts = self.merged_data.dropna(subset=['text'])
        
        # Calculate text lengths
        valid_texts['text_length'] = valid_texts['text'].str.len()
        valid_texts['word_count'] = valid_texts['text'].str.split().str.len()
        
        # Overall statistics
        text_stats = {
            'total_tweets': len(valid_texts),
            'avg_length': float(valid_texts['text_length'].mean()),
            'median_length': float(valid_texts['text_length'].median()),
            'min_length': int(valid_texts['text_length'].min()),
            'max_length': int(valid_texts['text_length'].max()),
            'avg_word_count': float(valid_texts['word_count'].mean()),
            'median_word_count': float(valid_texts['word_count'].median())
        }
        
        self.statistics['text_properties'] = text_stats
        
        print("\n" + "="*70)
        print("TEXT PROPERTIES")
        print("="*70)
        print(f"  Total tweets:        {text_stats['total_tweets']}")
        print(f"  Avg length:          {text_stats['avg_length']:.1f} chars")
        print(f"  Median length:       {text_stats['median_length']:.1f} chars")
        print(f"  Length range:        {text_stats['min_length']}-{text_stats['max_length']} chars")
        print(f"  Avg word count:      {text_stats['avg_word_count']:.1f} words")
        print(f"  Median word count:   {text_stats['median_word_count']:.1f} words")
        print("="*70)
        
        # Per-class statistics
        class_text_stats = {}
        for cls in TACO_CLASSES:
            if cls in valid_texts['class'].values:
                cls_data = valid_texts[valid_texts['class'] == cls]
                class_text_stats[cls] = {
                    'avg_length': float(cls_data['text_length'].mean()),
                    'avg_word_count': float(cls_data['word_count'].mean())
                }
        
        self.statistics['text_properties_by_class'] = class_text_stats
        
        print("\nTEXT PROPERTIES BY CLASS:")
        for cls, stats in class_text_stats.items():
            print(f"  {cls:15s}: {stats['avg_length']:6.1f} chars, {stats['avg_word_count']:5.1f} words")
        
        return text_stats
    
    def analyze_urls(self):
        """Analyze URL presence and patterns"""
        logger.info("Analyzing URLs...")
        
        valid_texts = self.merged_data.dropna(subset=['text'])
        
        # URL pattern (comprehensive)
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Detect URLs
        valid_texts['has_url'] = valid_texts['text'].str.contains(url_pattern, regex=True)
        valid_texts['url_count'] = valid_texts['text'].str.findall(url_pattern).str.len()
        
        # Overall URL statistics
        tweets_with_urls = valid_texts['has_url'].sum()
        total_urls = valid_texts['url_count'].sum()
        
        url_stats = {
            'tweets_with_urls': int(tweets_with_urls),
            'percentage_with_urls': float(tweets_with_urls / len(valid_texts) * 100),
            'total_url_occurrences': int(total_urls),
            'avg_urls_per_tweet': float(total_urls / len(valid_texts))
        }
        
        self.statistics['url_statistics'] = url_stats
        
        print("\n" + "="*70)
        print("URL STATISTICS")
        print("="*70)
        print(f"  Tweets with URLs:    {url_stats['tweets_with_urls']} ({url_stats['percentage_with_urls']:.1f}%)")
        print(f"  Total URL count:     {url_stats['total_url_occurrences']}")
        print(f"  Avg URLs per tweet:  {url_stats['avg_urls_per_tweet']:.2f}")
        print("="*70)
        
        # URL statistics by class
        url_by_class = {}
        for cls in TACO_CLASSES:
            if cls in valid_texts['class'].values:
                cls_data = valid_texts[valid_texts['class'] == cls]
                tweets_with_urls_cls = cls_data['has_url'].sum()
                url_by_class[cls] = {
                    'count': int(tweets_with_urls_cls),
                    'percentage': float(tweets_with_urls_cls / len(cls_data) * 100)
                }
        
        self.statistics['url_by_class'] = url_by_class
        
        print("\nURL PRESENCE BY CLASS:")
        for cls, stats in url_by_class.items():
            print(f"  {cls:15s}: {stats['count']:4d} tweets ({stats['percentage']:5.1f}%)")
        
        # Extract sample URLs for domain analysis
        all_urls = []
        for text in valid_texts['text'].dropna():
            urls = url_pattern.findall(text)
            all_urls.extend(urls)
        
        # Analyze domains
        domains = []
        for url in all_urls:
            try:
                # Extract domain
                domain = re.search(r'://([^/]+)', url)
                if domain:
                    domains.append(domain.group(1))
            except:
                continue
        
        domain_counts = Counter(domains)
        top_domains = dict(domain_counts.most_common(20))
        
        self.statistics['top_domains'] = top_domains
        
        print("\nTOP 20 DOMAINS:")
        for domain, count in list(top_domains.items())[:20]:
            print(f"  {domain:30s}: {count:4d}")
        
        return url_stats
    
    def analyze_special_characters(self):
        """Analyze special characters (mentions, hashtags, emojis)"""
        logger.info("Analyzing special characters...")
        
        valid_texts = self.merged_data.dropna(subset=['text'])
        
        # Mentions
        valid_texts['has_mention'] = valid_texts['text'].str.contains(r'@\w+', regex=True)
        mentions = valid_texts['has_mention'].sum()
        
        # Hashtags
        valid_texts['has_hashtag'] = valid_texts['text'].str.contains(r'#\w+', regex=True)
        hashtags = valid_texts['has_hashtag'].sum()
        
        # Basic emoji detection (Unicode ranges)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", 
            flags=re.UNICODE
        )
        valid_texts['has_emoji'] = valid_texts['text'].str.contains(emoji_pattern, regex=True)
        emojis = valid_texts['has_emoji'].sum()
        
        special_char_stats = {
            'mentions': int(mentions),
            'hashtags': int(hashtags),
            'emojis': int(emojis),
            'percentage_mentions': float(mentions / len(valid_texts) * 100),
            'percentage_hashtags': float(hashtags / len(valid_texts) * 100),
            'percentage_emojis': float(emojis / len(valid_texts) * 100)
        }
        
        self.statistics['special_characters'] = special_char_stats
        
        print("\n" + "="*70)
        print("SPECIAL CHARACTERS")
        print("="*70)
        print(f"  Mentions (@):        {special_char_stats['mentions']} ({special_char_stats['percentage_mentions']:.1f}%)")
        print(f"  Hashtags (#):        {special_char_stats['hashtags']} ({special_char_stats['percentage_hashtags']:.1f}%)")
        print(f"  Emojis:              {special_char_stats['emojis']} ({special_char_stats['percentage_emojis']:.1f}%)")
        print("="*70)
        
        return special_char_stats
    
    def analyze_thread_structure(self):
        """Analyze conversation thread structure"""
        logger.info("Analyzing thread structure...")
        
        # Top-level tweets (tweet_id == conversation_id)
        self.merged_data['is_top_level'] = (
            self.merged_data['tweet_id'] == self.merged_data['conversation_id']
        )
        
        top_level_count = self.merged_data['is_top_level'].sum()
        reply_count = len(self.merged_data) - top_level_count
        
        # Conversation sizes
        conv_sizes = self.merged_data.groupby('conversation_id').size()
        
        thread_stats = {
            'top_level_tweets': int(top_level_count),
            'reply_tweets': int(reply_count),
            'unique_conversations': int(len(conv_sizes)),
            'avg_conversation_size': float(conv_sizes.mean()),
            'median_conversation_size': float(conv_sizes.median()),
            'max_conversation_size': int(conv_sizes.max())
        }
        
        self.statistics['thread_structure'] = thread_stats
        
        print("\n" + "="*70)
        print("THREAD STRUCTURE")
        print("="*70)
        print(f"  Top-level tweets:    {thread_stats['top_level_tweets']}")
        print(f"  Reply tweets:        {thread_stats['reply_tweets']}")
        print(f"  Unique conversations: {thread_stats['unique_conversations']}")
        print(f"  Avg conv size:       {thread_stats['avg_conversation_size']:.1f} tweets")
        print(f"  Median conv size:    {thread_stats['median_conversation_size']:.1f} tweets")
        print(f"  Max conv size:       {thread_stats['max_conversation_size']} tweets")
        print("="*70)
        
        return thread_stats
    
    def identify_problematic_tweets(self):
        """Identify tweets that might be problematic for annotation"""
        logger.info("Identifying problematic tweets...")
        
        valid_texts = self.merged_data.dropna(subset=['text'])
        
        problems = {
            'missing_text': len(self.merged_data) - len(valid_texts),
            'very_short': len(valid_texts[valid_texts['text'].str.len() < MIN_TWEET_LENGTH]),
            'very_long': len(valid_texts[valid_texts['text'].str.len() > MAX_TWEET_LENGTH]),
            'only_url': len(valid_texts[valid_texts['text'].str.match(r'^\s*https?://.*\s*$')]),
            'only_mention': len(valid_texts[valid_texts['text'].str.match(r'^\s*@\w+\s*$')]),
        }
        
        self.statistics['problematic_tweets'] = problems
        
        total_problematic = sum(problems.values())
        
        print("\n" + "="*70)
        print("PROBLEMATIC TWEETS")
        print("="*70)
        for issue, count in problems.items():
            pct = count / len(self.merged_data) * 100
            print(f"  {issue:20s}: {count:4d} ({pct:5.2f}%)")
        print(f"  {'TOTAL':20s}: {total_problematic:4d} ({total_problematic/len(self.merged_data)*100:5.2f}%)")
        print("="*70)
        
        return problems
    
    def save_statistics(self):
        """Save statistics to JSON file"""
        logger.info("Saving statistics...")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        stats_clean = convert_types(self.statistics)
        
        with open(DATA_STATISTICS_FILE, 'w') as f:
            json.dump(stats_clean, f, indent=2)
        
        logger.info(f"✓ Statistics saved to {DATA_STATISTICS_FILE}")
    
    def save_merged_data(self):
        """Save merged dataset for preprocessing"""
        logger.info("Saving merged dataset...")
        
        # Select relevant columns
        output_df = self.merged_data[[
            'tweet_id', 'class', 'topic', 'confidence', 
            'text', 'conversation_id', 'parent_id'
        ]].copy()
        
        # Save to processed directory
        output_file = PROCESSED_DATA_DIR / "merged_taco_data.csv"
        output_df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Merged data saved to {output_file}")
        
        return output_file
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("=" * 70)
        logger.info("STARTING TACO DATASET EXPLORATION")
        logger.info("=" * 70)
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Exiting.")
            return False
        
        # Merge datasets
        self.merge_data()
        
        # Run all analyses
        self.analyze_class_distribution()
        self.analyze_text_properties()
        self.analyze_urls()
        self.analyze_special_characters()
        self.analyze_thread_structure()
        self.identify_problematic_tweets()
        
        # Save results
        self.save_statistics()
        self.save_merged_data()
        
        logger.info("=" * 70)
        logger.info("EXPLORATION COMPLETE")
        logger.info("=" * 70)
        
        print("\nAnalysis complete! Check the following files:")
        print(f"Statistics: {DATA_STATISTICS_FILE}")
        print(f"Merged data: {PROCESSED_DATA_DIR / 'merged_taco_data.csv'}")
        print(f"Log file: {get_log_file('taco_stats')}")
        
        return True


def main():
    """Main execution"""
    explorer = TACOExplorer()
    success = explorer.run_full_analysis()
    
    if success:
        print("\nReady to proceed with preprocessing!")
    else:
        print("\nExploration failed. Please check the logs.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
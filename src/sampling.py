"""
Sampling utilities for creating stratified test samples
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(get_log_file('sampling')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 100,
    exclude_classes: list = ['undecided'],
    class_column: str = 'taco_class',
    random_state: int = 42
) -> tuple:
    """
    Create stratified sample maintaining class proportions
    
    Args:
        df: Input DataFrame
        n_samples: Total number of samples to draw
        exclude_classes: Classes to exclude (e.g., 'Undecided')
        class_column: Column name containing class labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (sample_df, remaining_df)
    """
    logger.info(f"Creating stratified sample of {n_samples} tweets...")
    
    # Filter out excluded classes
    df_filtered = df[~df[class_column].isin(exclude_classes)].copy()
    logger.info(f"After excluding {exclude_classes}: {len(df_filtered)} tweets")
    
    # Calculate samples per class (proportional)
    class_counts = df_filtered[class_column].value_counts()
    total_filtered = len(df_filtered)
    
    samples_per_class = {}
    for cls, count in class_counts.items():
        proportion = count / total_filtered
        n_class = max(1, int(round(n_samples * proportion)))  # Ensure at least 1 sample per class
        samples_per_class[cls] = n_class
    
    # Adjust to ensure total = n_samples (handle rounding)
    total_assigned = sum(samples_per_class.values())
    if total_assigned > n_samples:
        # Remove from largest classes first
        sorted_classes = sorted(samples_per_class.items(), key=lambda x: x[1], reverse=True)
        diff = total_assigned - n_samples
        for cls, count in sorted_classes:
            if count > 1 and diff > 0:
                reduction = min(count - 1, diff)
                samples_per_class[cls] -= reduction
                diff -= reduction
                if diff == 0:
                    break
    elif total_assigned < n_samples:
        # Add to largest class
        largest_class = max(samples_per_class, key=samples_per_class.get)
        samples_per_class[largest_class] += (n_samples - total_assigned)
    
    logger.info("Samples per class:")
    for cls, n in samples_per_class.items():
        logger.info(f"  {cls:15s}: {n:3d} samples")
    
    # Sample from each class
    sampled_dfs = []
    for cls, n in samples_per_class.items():
        cls_df = df_filtered[df_filtered[class_column] == cls]
        
        # Check if we have enough samples
        if len(cls_df) < n:
            logger.warning(f"Class '{cls}' has only {len(cls_df)} tweets, requested {n}")
            n = len(cls_df)
        
        cls_sample = cls_df.sample(n=n, random_state=random_state)
        sampled_dfs.append(cls_sample)
    
    # Combine samples
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Get remaining data
    remaining_df = df_filtered[~df_filtered.index.isin(sample_df.index)].copy()
    
    logger.info(f"Created sample: {len(sample_df)} tweets")
    logger.info(f"Remaining: {len(remaining_df)} tweets")
    
    return sample_df, remaining_df


def create_test_sample(
    preprocessed_file: Path = PREPROCESSED_TWEETS_FILE,
    output_file: Path = None,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Create and save test sample for pipeline validation
    
    Args:
        preprocessed_file: Path to preprocessed tweets CSV
        output_file: Path to save test sample (optional)
        n_samples: Number of samples
        
    Returns:
        Test sample DataFrame
    """
    logger.info("="*70)
    logger.info("CREATING TEST SAMPLE")
    logger.info("="*70)
    
    # Load preprocessed data
    logger.info(f"Loading preprocessed data from {preprocessed_file}...")
    df = pd.read_csv(
        preprocessed_file,
        keep_default_na=False,
        na_values=['']
    )
    
    # Filter only valid tweets
    df_valid = df[df['is_valid'] == True].copy()
    logger.info(f"Valid tweets: {len(df_valid)}/{len(df)}")
    
    # Create stratified sample
    test_sample, remaining = create_stratified_sample(
        df_valid,
        n_samples=n_samples,
        exclude_classes=['undecided']
    )
    
    # Print class distribution
    print("\n" + "="*70)
    print("TEST SAMPLE CLASS DISTRIBUTION")
    print("="*70)
    class_dist = test_sample['taco_class'].value_counts()
    for cls, count in class_dist.items():
        pct = (count / len(test_sample)) * 100
        print(f"  {cls:15s}: {count:3d} ({pct:5.1f}%)")
    print("="*70)
    
    # Save if output file specified
    if output_file:
        test_sample.to_csv(output_file, index=False)
        logger.info(f"Test sample saved to {output_file}")
    
    logger.info("="*70)
    logger.info("TEST SAMPLE CREATION COMPLETE")
    logger.info("="*70)
    
    return test_sample


def load_annotation_data(
    preprocessed_file: Path = PREPROCESSED_TWEETS_FILE,
    test_mode: bool = False,
    test_sample_size: int = 100
) -> pd.DataFrame:
    """
    Load data for annotation (either test sample or full dataset)
    
    Args:
        preprocessed_file: Path to preprocessed tweets
        test_mode: If True, return test sample only
        test_sample_size: Size of test sample
        
    Returns:
        DataFrame ready for annotation
    """
    logger.info(f"Loading annotation data (test_mode={test_mode})...")
    
    # Load preprocessed data
    df = pd.read_csv(
        preprocessed_file,
        keep_default_na=False,
        na_values=['']
    )
    
    # Filter valid tweets and exclude Undecided
    df_valid = df[
        (df['is_valid'] == True) & 
        (df['taco_class'] != 'undecided')
    ].copy()
    
    logger.info(f"Valid tweets for annotation: {len(df_valid)}")
    
    if test_mode:
        # Create test sample
        test_sample, _ = create_stratified_sample(
            df_valid,
            n_samples=test_sample_size,
            exclude_classes=[]  # Already filtered out Undecided
        )
        logger.info(f"Test mode: Using {len(test_sample)} tweets")
        return test_sample
    else:
        logger.info(f"Full mode: Using all {len(df_valid)} tweets")
        return df_valid


def analyze_sample_quality(sample_df: pd.DataFrame, full_df: pd.DataFrame):
    """
    Analyze how representative the sample is
    
    Args:
        sample_df: Sample DataFrame
        full_df: Full DataFrame
    """
    print("\n" + "="*70)
    print("SAMPLE QUALITY ANALYSIS")
    print("="*70)
    
    # Class distribution comparison
    print("\nClass Distribution Comparison:")
    print(f"{'Class':<15} {'Sample %':<12} {'Full %':<12} {'Difference'}")
    print("-" * 60)
    
    sample_dist = sample_df['taco_class'].value_counts(normalize=True) * 100
    full_dist = full_df['taco_class'].value_counts(normalize=True) * 100
    
    for cls in sample_dist.index:
        sample_pct = sample_dist[cls]
        full_pct = full_dist.get(cls, 0)
        diff = sample_pct - full_pct
        print(f"{cls:<15} {sample_pct:>6.2f}%     {full_pct:>6.2f}%     {diff:>+6.2f}%")
    
    # Text length comparison
    print("\nText Length Comparison:")
    print(f"{'Metric':<20} {'Sample':<15} {'Full Dataset'}")
    print("-" * 50)
    print(f"{'Avg length':<20} {sample_df['text_length'].mean():>8.1f} chars {full_df['text_length'].mean():>8.1f} chars")
    print(f"{'Median length':<20} {sample_df['text_length'].median():>8.1f} chars {full_df['text_length'].median():>8.1f} chars")
    
    # URL presence comparison
    sample_url_pct = (sample_df['has_urls'].sum() / len(sample_df)) * 100
    full_url_pct = (full_df['has_urls'].sum() / len(full_df)) * 100
    print(f"{'URLs present':<20} {sample_url_pct:>8.1f}%      {full_url_pct:>8.1f}%")
    
    print("="*70)


def main():
    """Main execution for testing"""
    
    # Check if preprocessed data exists
    if not PREPROCESSED_TWEETS_FILE.exists():
        logger.error(f"Preprocessed data not found: {PREPROCESSED_TWEETS_FILE}")
        logger.error("Please run data_preprocessing.py first!")
        return 1
    
    # Create test sample
    test_sample_file = PROCESSED_DATA_DIR / "test_sample_100.csv"
    test_sample = create_test_sample(
        preprocessed_file=PREPROCESSED_TWEETS_FILE,
        output_file=test_sample_file,
        n_samples=100
    )
    
    # Load full valid dataset for comparison
    full_df = pd.read_csv(
        PREPROCESSED_TWEETS_FILE,
        keep_default_na=False,
        na_values=['']
    )
    full_valid = full_df[
        (full_df['is_valid'] == True) & 
        (full_df['taco_class'] != 'undecided')
    ]
    
    # Analyze sample quality
    analyze_sample_quality(test_sample, full_valid)
    
    print(f"\nTest sample created: {test_sample_file}")
    print(f"   Ready for annotation pipeline testing!")
    
    return 0


if __name__ == "__main__":
    exit(main())
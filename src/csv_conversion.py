"""
Convert LLM JSON annotations to CSV format for classification
Handles multi-label, multiple evidence spans per label
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import config
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

def aggregate_perspectives(annotations_list, confidence_threshold=0.7):
    """
    Aggregate multiple annotations for a single tweet
    
    Args:
        annotations_list: List of annotation dicts from LLM
        confidence_threshold: Minimum confidence to count perspective (default 0.7)
        
    Returns:
        Dictionary with aggregated features
    """
    # Initialize features
    features = {
        'information_providing': 0,
        'information_seeking': 0,
        'epistemic': 0,
        'experiential': 0,
        'opinion': 0,
        'aot': 0,
        
        'max_conf_information_providing': 0.0,
        'max_conf_information_seeking': 0.0,
        'max_conf_epistemic': 0.0,
        'max_conf_experiential': 0.0,
        'max_conf_opinion': 0.0,
        'max_conf_aot': 0.0,
        
        'count_information_providing': 0,
        'count_information_seeking': 0,
        'count_epistemic': 0,
        'count_experiential': 0,
        'count_opinion': 0,
        'count_aot': 0,
        
        'num_perspectives': 0,
        'avg_confidence': 0.0,
        'total_annotations': len(annotations_list)
    }
    
    # Label normalization
    label_map = {
        'Information Providing': 'information_providing',
        'Information Seeking': 'information_seeking',
        'Epistemic/Factual': 'epistemic',
        'Experiential': 'experiential',
        'Opinion/Belief': 'opinion',
        'AOT': 'aot'
    }
    
    confidences = []
    perspective_set = set()
    
    for ann in annotations_list:
        label_orig = ann.get('label', '')
        confidence = float(ann.get('confidence', 0.0))
        
        # Normalize label
        label_normalized = label_map.get(label_orig)
        
        if label_normalized and confidence >= confidence_threshold:
            # Binary presence
            features[f'{label_normalized}'] = 1
            
            # Max confidence
            current_max = features[f'max_conf_{label_normalized}']
            features[f'max_conf_{label_normalized}'] = max(current_max, confidence)
            
            # Count occurrences
            features[f'count_{label_normalized}'] += 1
            
            # Track for aggregates
            confidences.append(confidence)
            perspective_set.add(label_normalized)
    
    # Aggregate statistics
    features['num_perspectives'] = len(perspective_set)
    features['avg_confidence'] = np.mean(confidences) if confidences else 0.0
    
    return features

def convert_json_to_csv(
    json_file: Path,
    taco_csv: Path,
    output_csv: Path,
    confidence_threshold: float = 0.7
):
    """
    Convert JSON annotations to CSV with binary features
    
    Args:
        json_file: Path to LLM annotations JSON
        taco_csv: Path to original TACO merged CSV
        output_csv: Path to save output CSV
        confidence_threshold: Minimum confidence to count (0.7 = 70%)
    """
    print("="*70)
    print("CONVERTING JSON ANNOTATIONS TO CSV")
    print("="*70)
    
    # Load JSON annotations
    print(f"\nLoading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations_dict = data.get('annotations', {})
    print(f"✓ Loaded {len(annotations_dict)} annotated tweets")
    
    # Load original TACO data
    print(f"\nLoading TACO data from {taco_csv}...")
    taco_df = pd.read_csv(
        taco_csv,
        dtype={'tweet_id': str},
        keep_default_na=False,
        na_values=['']
    )
    print(f"✓ Loaded {len(taco_df)} TACO tweets")
    
    # Process each tweet
    rows = []
    
    '''
    for tweet_id, ann_data in annotations_dict.items():
        # Get TACO metadata
        taco_row = taco_df[taco_df['tweet_id'] == str(tweet_id)]
        
        if len(taco_row) == 0:
            print(f"⚠️ Tweet {tweet_id} not found in TACO data")
            continue
        
        taco_row = taco_row.iloc[0]
        
        # Extract annotations
        tweet_annotations = ann_data['annotation'].get(str(tweet_id), [])
    '''

    for _, taco_row in taco_df.iterrows():
        tweet_id = str(taco_row['tweet_id'])
        ann_data = annotations_dict.get(tweet_id, None)
        if ann_data is None:
            tweet_annotations = []
        else:
            if 'annotation' in ann_data:
                tweet_annotations = ann_data["annotation"].get(tweet_id, [])
            else:
                tweet_annotations = []    
        # Aggregate perspectives
        features = aggregate_perspectives(tweet_annotations, confidence_threshold)
        
        # Build row
        row = {
            'tweet_id': tweet_id,
            'taco_class': taco_row['class'],
            'topic': taco_row['topic'],
            'text': taco_row['text'],
            'text_length': len(taco_row['text']),
            'word_count': len(taco_row['text'].split()),
            'has_url': int('http' in taco_row['text'].lower()),
            **features  # Add all perspective features
        }
        
        rows.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Sort columns for readability
    col_order = [
        'tweet_id', 'taco_class', 'topic', 'text',
        'text_length', 'word_count', 'has_url',
        'information_providing', 'information_seeking',
        'epistemic', 'experiential', 'opinion', 'aot',
        'max_conf_information_providing', 'max_conf_information_seeking',
        'max_conf_epistemic', 'max_conf_experiential', 
        'max_conf_opinion', 'max_conf_aot',
        'count_information_providing', 'count_information_seeking',
        'count_epistemic', 'count_experiential', 'count_opinion', 'count_aot',
        'num_perspectives', 'avg_confidence', 'total_annotations'
    ]
    
    result_df = result_df[col_order]
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved perspective-enriched CSV to {output_csv}")
    
    # Print statistics
    print("\n" + "="*70)
    print("PERSPECTIVE STATISTICS")
    print("="*70)
    
    for perspective in ['information_providing', 'information_seeking', 
                        'epistemic', 'experiential', 'opinion', 'aot']:
        count = result_df[f'{perspective}'].sum()
        pct = (count / len(result_df)) * 100
        avg_conf = result_df[result_df[f'{perspective}'] == 1][f'max_conf_{perspective}'].mean()
        print(f"{perspective:25s}: {count:4d} tweets ({pct:5.1f}%), avg conf: {avg_conf:.3f}")
    
    print(f"\nAvg perspectives per tweet: {result_df['num_perspectives'].mean():.2f}")
    print(f"Avg confidence overall:     {result_df['avg_confidence'].mean():.3f}")
    
    # By TACO class
    print("\n" + "="*70)
    print("PERSPECTIVES BY TACO CLASS")
    print("="*70)
    
    for taco_class in ['reason', 'statement', 'notification', 'none', 'undecided']:
        class_df = result_df[result_df['taco_class'] == taco_class]
        if len(class_df) == 0:
            continue
        avg_persp = class_df['num_perspectives'].mean()
        print(f"{taco_class:15s}: {len(class_df):4d} tweets, avg {avg_persp:.2f} perspectives")

    # By TACO class - detailed perspective breakdown
    print("\n" + "="*70)
    print("PERSPECTIVE DISTRIBUTION BY CLASS")
    print("="*70)
    
    for taco_class in ['reason', 'statement', 'notification', 'none']:
        class_df = result_df[result_df['taco_class'] == taco_class]
        if len(class_df) == 0:
            continue
        print(f"\n{taco_class.upper()}:")
        for perspective in ['information_providing', 'information_seeking', 
                            'epistemic', 'experiential', 'opinion', 'aot']:
            count = class_df[f'{perspective}'].sum()
            pct = (count / len(class_df)) * 100
            avg_conf = class_df[class_df[f'{perspective}'] == 1][f'max_conf_{perspective}'].mean() if count > 0 else 0
            print(f"  {perspective:25s}: {count:3d}/{len(class_df):3d} ({pct:5.1f}%), avg conf: {avg_conf:.3f}")
    
    print("="*70)
    
    return result_df

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON annotations to CSV')
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='Path to JSON annotations file'
    )
    parser.add_argument(
        '--taco-csv',
        type=str,
        default=str(PROCESSED_DATA_DIR / 'merged_taco_data.csv'),
        help='Path to TACO merged CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(PROCESSED_DATA_DIR / 'perspective_enriched.csv'),
        help='Output CSV path'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='Minimum confidence to count perspective (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Convert
    df = convert_json_to_csv(
        json_file=Path(args.json),
        taco_csv=Path(args.taco_csv),
        output_csv=Path(args.output),
        confidence_threshold=args.confidence_threshold
    )
    
    print(f"\n Conversion complete!")
    print(f"   Output: {args.output}")
    print(f"   Ready for baseline comparison!")
    
    return 0

if __name__ == "__main__":
    exit(main())
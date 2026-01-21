"""
Main annotation pipeline for TACO perspective annotation
Orchestrates: sampling, prompt building, vLLM inference, validation, output
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# Import all components
import sys
sys.path.append(str(Path(__file__).parent))
from config import *
from sampling import load_annotation_data
from prompt_builder import PromptBuilder
from vllm_client import VLLMAnnotator
from output_validator import AnnotationValidator

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(get_log_file('annotation_pipeline')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """
    Main pipeline for perspective annotation
    """
    
    def __init__(
        self,
        model_name: str = None,
        batch_size: int = BATCH_SIZE,
        test_mode: bool = False,
        test_sample_size: int = 100
    ):
        """
        Initialize annotation pipeline
        
        Args:
            model_name: vLLM model to use
            batch_size: Number of tweets per batch
            test_mode: If True, annotate only test sample
            test_sample_size: Size of test sample
        """
        self.test_mode = test_mode
        self.test_sample_size = test_sample_size
        self.batch_size = batch_size
        
        logger.info("="*70)
        logger.info("INITIALIZING ANNOTATION PIPELINE")
        logger.info("="*70)
        logger.info(f"Mode: {'TEST' if test_mode else 'FULL'}")
        logger.info(f"Batch size: {batch_size}")
        
        # Initialize components
        logger.info("Loading components...")
        
        self.prompt_builder = PromptBuilder()
        logger.info("Prompt builder ready")
        
        self.llm = VLLMAnnotator(model_name=model_name)
        logger.info("vLLM model ready")
        
        self.validator = AnnotationValidator(strict_mode=False)
        logger.info("Validator ready")
        
        # Storage for results
        self.annotations = {}
        self.failed_tweets = []
        self.validation_errors = []
        
        logger.info("="*70)
    
    def load_data(self) -> pd.DataFrame:
        """Load tweets for annotation"""
        logger.info("Loading annotation data...")
        
        df = load_annotation_data(
            preprocessed_file=PREPROCESSED_TWEETS_FILE,
            test_mode=self.test_mode,
            test_sample_size=self.test_sample_size
        )
        
        logger.info(f"Loaded {len(df)} tweets for annotation")
        
        return df
    
    def create_batches(self, df: pd.DataFrame) -> list:
        """
        Create batches for processing
        
        Args:
            df: DataFrame with tweets
            
        Returns:
            List of DataFrame batches
        """
        batches = []
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            batches.append(batch_df)
        
        logger.info(f"Created {len(batches)} batches (batch_size={self.batch_size})")
        return batches
    
    def annotate_batch(self, batch_df: pd.DataFrame) -> dict:
        """
        Annotate a single batch
        
        Args:
            batch_df: DataFrame batch
            
        Returns:
            Dictionary of results
        """
        batch_results = {
            'annotations': {},
            'failed': [],
            'validation_errors': []
        }
        
        # Build prompts
        prompts_data = self.prompt_builder.build_batch_prompts(batch_df)
        prompts = [prompt for _, prompt in prompts_data]
        tweet_ids = [tweet_id for tweet_id, _ in prompts_data]
        
        # Run inference
        responses = self.llm.annotate_batch(prompts, show_progress=False)
        
        # Validate each response
        for i, (tweet_id, (response, success)) in enumerate(zip(tweet_ids, responses)):
            tweet_text = batch_df.iloc[i]['text_cleaned']
            
            if not success:
                batch_results['failed'].append({
                    'tweet_id': tweet_id,
                    'error': 'LLM inference failed',
                    'response': response
                })
                continue
            
            # Validate annotation
            is_valid, parsed, errors = self.validator.validate_annotation(
                tweet_id, tweet_text, response
            )
            
            if is_valid:
                batch_results['annotations'][tweet_id] = {
                    'annotation': parsed,
                    'raw_response': response,
                    'validation_errors': errors if errors else []
                }
            else:
                batch_results['failed'].append({
                    'tweet_id': tweet_id,
                    'error': 'Validation failed',
                    'errors': errors,
                    'raw_response': response,
                    'tweet_text': tweet_text
                })
        
        return batch_results
    
    def run(self, output_file: Path = None) -> dict:
        """
        Run complete annotation pipeline
        
        Args:
            output_file: Path to save annotations
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("STARTING ANNOTATION PIPELINE")
        logger.info("="*70)
        
        # Load data
        df = self.load_data()
        total_tweets = len(df)
        
        # Create batches
        batches = self.create_batches(df)
        
        # Process batches with progress bar
        logger.info(f"Processing {len(batches)} batches...")
        
        with tqdm(total=total_tweets, desc="Annotating tweets") as pbar:
            for batch_idx, batch_df in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}...")
                
                batch_results = self.annotate_batch(batch_df)
                
                # Accumulate results
                self.annotations.update(batch_results['annotations'])
                self.failed_tweets.extend(batch_results['failed'])
                
                # Update progress
                pbar.update(len(batch_df))
                
                # Log progress
                success_count = len(batch_results['annotations'])
                failed_count = len(batch_results['failed'])
                logger.info(f"  Batch {batch_idx + 1}: {success_count} success, {failed_count} failed")
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        summary = self.generate_summary(total_tweets, elapsed_time)
        
        # Save results
        if output_file:
            self.save_results(output_file, summary)
        
        # Print summary
        self.print_summary(summary)
        
        logger.info("="*70)
        logger.info("ANNOTATION PIPELINE COMPLETE")
        logger.info("="*70)
        
        return summary
    
    def generate_summary(self, total_tweets: int, elapsed_time: float) -> dict:
        """Generate summary statistics"""
        successful = len(self.annotations)
        failed = len(self.failed_tweets)
        
        # Get component statistics
        llm_stats = self.llm.get_statistics()
        validation_stats = self.validator.get_statistics()
        
        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'test' if self.test_mode else 'full',
                'model': self.llm.model_name,
                'batch_size': self.batch_size
            },
            'totals': {
                'total_tweets': total_tweets,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total_tweets * 100) if total_tweets > 0 else 0
            },
            'timing': {
                'total_time_seconds': elapsed_time,
                'avg_time_per_tweet': elapsed_time / total_tweets if total_tweets > 0 else 0,
                'tweets_per_second': total_tweets / elapsed_time if elapsed_time > 0 else 0
            },
            'llm_statistics': llm_stats,
            'validation_statistics': validation_stats
        }
        
        return summary
    
    def save_results(self, output_file: Path, summary: dict):
        """Save annotations and summary"""
        logger.info(f"Saving results to {output_file}...")
        
        output_data = {
            'metadata': summary['metadata'],
            'summary': summary,
            'annotations': self.annotations,
            'failed_tweets': self.failed_tweets
        }
        
        # Save main annotations file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Save failed tweets separately for review
        if self.failed_tweets:
            failed_file = output_file.parent / f"{output_file.stem}_failed.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_tweets, f, indent=2)
            logger.info(f"Failed tweets saved to {failed_file}")
    
    def print_summary(self, summary: dict):
        """Print summary to console"""
        print("\n" + "="*70)
        print("ANNOTATION SUMMARY")
        print("="*70)
        
        # Totals
        totals = summary['totals']
        print(f"\nTotals:")
        print(f"  Total tweets:        {totals['total_tweets']}")
        print(f"  Successful:          {totals['successful']}")
        print(f"  Failed:              {totals['failed']}")
        print(f"  Success rate:        {totals['success_rate']:.2f}%")
        
        # Timing
        timing = summary['timing']
        print(f"\nTiming:")
        print(f"  Total time:          {timing['total_time_seconds']:.2f}s")
        print(f"  Avg time/tweet:      {timing['avg_time_per_tweet']:.3f}s")
        print(f"  Tweets/second:       {timing['tweets_per_second']:.2f}")
        
        # LLM statistics
        llm_stats = summary['llm_statistics']
        print(f"\nLLM Inference:")
        print(f"  Tokens generated:    {llm_stats['total_tokens']}")
        print(f"  Tokens/second:       {llm_stats['tokens_per_second']:.1f}")
        
        # Validation statistics
        val_stats = summary['validation_statistics']
        print(f"\nValidation:")
        print(f"  Validation rate:     {val_stats['validation_rate']:.2f}%")
        
        if val_stats.get('errors_by_type'):
            print(f"  Common errors:")
            for error_type, count in list(val_stats['errors_by_type'].items())[:5]:
                print(f"    {error_type:25s}: {count}")
        
        print("="*70)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TACO Perspective Annotation Pipeline')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (100 tweets only)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for inference (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        mode_str = 'test' if args.test else 'full'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = ANNOTATIONS_DIR / f"pass1_{args.model}_{mode_str}_{timestamp}.json"
    
    # Check prerequisites
    if not PREPROCESSED_TWEETS_FILE.exists():
        logger.error(f"Preprocessed data not found: {PREPROCESSED_TWEETS_FILE}")
        logger.error("Please run data_preprocessing.py first!")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = AnnotationPipeline(
            model_name=VLLM_MODELS.get(args.model, args.model),
            batch_size=args.batch_size,
            test_mode=args.test,
            test_sample_size=100
        )
        
        # Run annotation
        summary = pipeline.run(output_file=output_file)
        
        print(f"\nAnnotation complete!")
        print(f"   Results: {output_file}")
        
        # Recommendations based on results
        if summary['totals']['failed'] > 0:
            print(f"\n{summary['totals']['failed']} tweets failed annotation.")
            print(f"   Review failed tweets in: {output_file.parent / f'{output_file.stem}_failed.json'}")
        
        if args.test:
            print(f"\nTest mode completed successfully!")
            #print(f"   To annotate full dataset, run: python annotation_pipeline.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
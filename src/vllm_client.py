"""
vLLM Client for batch inference with Llama models
"""

import logging
import time
from typing import List, Tuple, Optional
from pathlib import Path

from vllm import LLM, SamplingParams

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

logger = logging.getLogger(__name__)


class VLLMAnnotator:
    """
    Wrapper for vLLM inference optimized for perspective annotation
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.6,
        max_model_len=4096,
    ):
        """
        Initialize vLLM model
        
        Args:
            model_name: HuggingFace model identifier
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        """
        if model_name is None:
            model_name = VLLM_MODELS[DEFAULT_MODEL]
        
        logger.info("="*70)
        logger.info("INITIALIZING vLLM MODEL")
        logger.info("="*70)
        logger.info(f"Model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-p: {top_p}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

        # Initialize vLLM
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
                quantization="awq"
                #device ="cuda:0"
            )
            logger.info("✓ vLLM model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load vLLM model: {e}")
            raise
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=None  # No special stop sequences
        )
        
        self.model_name = model_name
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
    
    def annotate_single(self, prompt: str) -> Tuple[str, bool]:
        """
        Annotate a single tweet
        
        Args:
            prompt: Full annotation prompt
            
        Returns:
            Tuple of (response_text, success)
        """
        try:
            start_time = time.time()
            
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            elapsed = time.time() - start_time
            
            # Extract response
            response_text = outputs[0].outputs[0].text
            
            # Update stats
            self.stats['total_requests'] += 1
            self.stats['successful'] += 1
            self.stats['total_tokens'] += len(outputs[0].outputs[0].token_ids)
            self.stats['total_time'] += elapsed
            
            return response_text, True
            
        except Exception as e:
            logger.error(f"Error in single annotation: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed'] += 1
            return str(e), False
    
    def annotate_batch(
        self, 
        prompts: List[str],
        show_progress: bool = True
    ) -> List[Tuple[str, bool]]:
        """
        Annotate a batch of tweets
        
        Args:
            prompts: List of annotation prompts
            show_progress: Whether to log progress
            
        Returns:
            List of (response_text, success) tuples
        """
        try:
            if show_progress:
                logger.info(f"Annotating batch of {len(prompts)} tweets...")
            
            start_time = time.time()
            
            # Generate annotations
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            elapsed = time.time() - start_time
            
            # Extract responses
            results = []
            total_tokens = 0
            
            for output in outputs:
                response_text = output.outputs[0].text
                results.append((response_text, True))
                total_tokens += len(output.outputs[0].token_ids)
            
            # Update stats
            self.stats['total_requests'] += len(prompts)
            self.stats['successful'] += len(prompts)
            self.stats['total_tokens'] += total_tokens
            self.stats['total_time'] += elapsed
            
            if show_progress:
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                logger.info(f"✓ Batch completed in {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch annotation: {e}")
            self.stats['total_requests'] += len(prompts)
            self.stats['failed'] += len(prompts)
            
            # Return error for all prompts
            return [(str(e), False) for _ in prompts]
    
    def annotate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Tuple[str, bool]:
        """
        Annotate with retry logic
        
        Args:
            prompt: Annotation prompt
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            
        Returns:
            Tuple of (response_text, success)
        """
        for attempt in range(max_retries):
            response, success = self.annotate_single(prompt)
            
            if success:
                return response, True
            
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        logger.error(f"All {max_retries} attempts failed")
        return response, False
    
    def get_statistics(self) -> dict:
        """
        Get inference statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['total_requests']) * 100
        else:
            stats['success_rate'] = 0.0
        
        if stats['total_time'] > 0:
            stats['avg_time_per_request'] = stats['total_time'] / stats['total_requests']
            stats['tokens_per_second'] = stats['total_tokens'] / stats['total_time']
        else:
            stats['avg_time_per_request'] = 0.0
            stats['tokens_per_second'] = 0.0
        
        return stats
    
    def print_statistics(self):
        """Print inference statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("vLLM INFERENCE STATISTICS")
        print("="*70)
        print(f"  Total requests:      {stats['total_requests']}")
        print(f"  Successful:          {stats['successful']}")
        print(f"  Failed:              {stats['failed']}")
        print(f"  Success rate:        {stats['success_rate']:.2f}%")
        print(f"  Total tokens:        {stats['total_tokens']}")
        print(f"  Total time:          {stats['total_time']:.2f}s")
        print(f"  Avg time/request:    {stats['avg_time_per_request']:.3f}s")
        print(f"  Tokens/second:       {stats['tokens_per_second']:.1f}")
        print("="*70)


def test_vllm_client():
    """Test vLLM client with simple prompts"""
    print("="*70)
    print("TESTING vLLM CLIENT")
    print("="*70)
    
    # Initialize client
    try:
        client = VLLMAnnotator(
            model_name=VLLM_MODELS[DEFAULT_MODEL],
            temperature=0.7,
            max_tokens=500
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize vLLM: {e}")
        print("  Make sure vLLM is properly installed and GPU is available")
        return 1
    
    # Test prompt
    test_prompt = """You are a helpful assistant. 

Respond with a valid JSON object containing:
{
  "message": "Hello, I am working!",
  "status": "success"
}

Provide only the JSON, no other text."""
    
    print("\nTest 1: Single inference")
    print(f"Prompt: {test_prompt[:100]}...")
    
    response, success = client.annotate_single(test_prompt)
    
    if success:
        print(f"✓ Response received:")
        print(f"  {response[:200]}...")
    else:
        print(f"✗ Failed: {response}")
    
    # Test batch inference
    print("\nTest 2: Batch inference (3 prompts)")
    batch_prompts = [test_prompt] * 3
    
    results = client.annotate_batch(batch_prompts, show_progress=True)
    
    print(f"✓ Batch completed: {len(results)} responses")
    for i, (response, success) in enumerate(results, 1):
        status = "✓" if success else "✗"
        print(f"  {status} Response {i}: {len(response)} chars")
    
    # Print statistics
    client.print_statistics()
    
    print("\n" + "="*70)
    print("vLLM CLIENT TEST COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    # Setup logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit(test_vllm_client())
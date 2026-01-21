"""
Prompt builder for perspective annotation
Constructs prompts from template + tweet text
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

logger = logging.getLogger(__name__)


# Base prompt template (from your document)
PERSPECTIVE_ANNOTATION_TEMPLATE = """You are an expert in cognitive psychology, epistemic cognition, and social media discourse analysis. Your task is to annotate short texts (tweets, comments, or hypotheses) with the cognitive perspectives they express.

A single text may reflect multiple perspectives. Your job is to identify all applicable perspectives, extract the exact evidence spans, assign a confidence score, and provide a brief rationale.

Use the following psychologically grounded definitions:

**Information Providing**: While often studied in the context of communication or information behavior, from a cognitive perspective, it involves the active encoding, organization, and external transfer of one's existing knowledge or data to another person or system. This behavior is often motivated by a perceived need in others or a goal to share information, involving the cognitive processes of memory retrieval, synthesis, and communication strategy.

**Information Seeking**: This is the purposive and conscious effort to identify, locate, and acquire information in response to a perceived need or a "gap" in one's current knowledge to solve a problem or satisfy a goal. It is an active process that involves various cognitive activities such as formulating queries, browsing, extracting, and evaluating information, and is influenced by cognitive (thoughts), affective (feelings), and physical factors.

**Epistemic/Factual Perspective**: This perspective relates to epistemic cognition, which involves an individual's understanding of knowledge and knowing, including beliefs about the certainty, sources, and justification of knowledge. An epistemic/factual perspective specifically emphasizes knowledge that is objective, verifiable, and evidence-based, focusing on the rigorous evaluation of information against established standards of justification to determine its accuracy and reliability.

**Experiential Perspective**: This approach emphasizes that knowledge and understanding are derived from and tested through direct, personal experience. The cognitive perspective here involves a process of continuous learning through a cycle of experiencing, reflecting, thinking, and acting, where knowledge is created through the transaction between personal experience and social knowledge. It often contrasts with purely abstract, detached learning.

**Opinion/Belief**: In a psychological context, opinions and beliefs are mental representations or stances that a person holds to be true, but which may not meet rigorous epistemic standards of factual knowledge. They are influenced by an individual's personal experiences, social interactions, and cognitive biases, and can affect behavior and emotions. They are often more subjective and can be more resistant to change than factual knowledge.

**Actively Open-Minded Thinking (AOT)**: This is a cognitive disposition or personality trait characterized by the willingness to sincerely consider different perspectives, challenge one's own beliefs, and weigh evidence fairly when making decisions or forming conclusions. AOT involves metacognitive awareness and the volitional effort to overcome biases that favor one's existing beliefs (such as confirmation bias), with the goal of achieving a more accurate and grounded understanding of reality.

For each input text:
1. Identify all applicable perspectives from the list above.
2. For each perspective, extract an evidence span:
   - Must be copied verbatim from the input text.
   - Must be the minimal phrase that justifies the label.
3. Provide a confidence score between 0.0 and 1.0.
4. Provide a brief rationale (1–2 sentences) explaining why the perspective applies.

Output must follow the exact JSON format below:
{{
  "tweet_id": [
    {{
      "label": "SINGLE_LABEL_ONLY",
      "evidence_span": "...",
      "confidence": 0.0,
      "rationale": "..."
    }}
  ]
}}

CRITICAL: Each annotation object must have EXACTLY ONE label from this list:
- "Information Providing"
- "Information Seeking"
- "Epistemic/Factual"
- "Experiential"
- "Opinion/Belief"
- "AOT"

If multiple perspectives apply, create SEPARATE annotation objects for each perspective.
DO NOT combine labels with pipes (|) or commas.

Notes:
- Use multiple objects in the list if multiple perspectives apply.
- Evidence spans must be substrings of the input text.
- Confidence must be a float between 0 and 1.
- Do not include labels that are not supported by evidence.
- Do not hallucinate content not present in the text.

Example:
Input: "I'm not sure if this policy will work. Does anyone have data on its past performance?"
Output:
{{
  "tweet_001": [
    {{
      "label": "Information Seeking",
      "evidence_span": "Does anyone have data on its past performance?",
      "confidence": 0.91,
      "rationale": "The text explicitly asks for data, indicating a knowledge gap and a request for information."
    }},
    {{
      "label": "Opinion/Belief",
      "evidence_span": "I'm not sure if this policy will work",
      "confidence": 0.78,
      "rationale": "The speaker expresses uncertainty and a subjective evaluation of the policy."
    }}
  ]
}}

IMPORTANT FORMATTING RULES:
1. Return ONLY valid JSON - no comments, no preamble, no markdown
2. Each "label" field contains EXACTLY ONE perspective label
3. If a tweet has multiple perspectives, create separate objects in the array
4. Evidence spans must be verbatim excerpts from the tweet text
5. Do not add any text after the closing brace

Now annotate the following tweet:

Tweet ID: {tweet_id}
Tweet Text: "{tweet_text}"

Provide your annotation in valid JSON format only. Do not include any preamble or explanation outside the JSON.
"""


class PromptBuilder:
    """
    Builds annotation prompts for tweets
    """
    
    def __init__(self, template: str = PERSPECTIVE_ANNOTATION_TEMPLATE):
        self.template = template
        logger.info("PromptBuilder initialized")
    
    def build_prompt(
        self, 
        tweet_id: str, 
        tweet_text: str,
        url_content: Optional[Dict] = None
    ) -> str:
        """
        Build annotation prompt for a single tweet
        
        Args:
            tweet_id: Tweet identifier
            tweet_text: Tweet text (cleaned)
            url_content: Optional URL content for Pass 2 (not used in Pass 1)
            
        Returns:
            Complete prompt string
        """
        # For Pass 1: Simple substitution
        if url_content is None:
            prompt = self.template.format(
                tweet_id=tweet_id,
                tweet_text=tweet_text
            )
        else:
            # For Pass 2: Augmented prompt with URL content
            prompt = self._build_augmented_prompt(tweet_id, tweet_text, url_content)
        
        return prompt
    
    def _build_augmented_prompt(
        self, 
        tweet_id: str, 
        tweet_text: str, 
        url_content: Dict
    ) -> str:
        """
        Build augmented prompt with URL content (for Pass 2)
        
        Args:
            tweet_id: Tweet identifier
            tweet_text: Tweet text
            url_content: Dictionary with 'title', 'summary', 'content_type'
            
        Returns:
            Augmented prompt string
        """
        # Augmented template for Pass 2
        augmented_section = f"""
The tweet includes a URL linking to:
Title: {url_content.get('title', 'N/A')}
Content: {url_content.get('summary', 'N/A')}

Consider both the tweet text AND the linked content when annotating perspectives.
If the URL provides evidence or factual information, this may indicate "Information Providing" or "Epistemic/Factual" perspectives.
"""
        
        # Insert augmented section before the annotation request
        base_prompt = self.template.format(
            tweet_id=tweet_id,
            tweet_text=tweet_text
        )
        
        # Insert augmented section before "Now annotate..."
        parts = base_prompt.split("Now annotate the following tweet:")
        augmented_prompt = parts[0] + augmented_section + "\nNow annotate the following tweet:" + parts[1]
        
        return augmented_prompt
    
    def build_batch_prompts(
        self, 
        tweets_df, 
        url_contents: Optional[Dict] = None
    ) -> list:
        """
        Build prompts for a batch of tweets
        
        Args:
            tweets_df: DataFrame with 'tweet_id' and 'text_cleaned' columns
            url_contents: Optional dict mapping tweet_id to URL content (Pass 2)
            
        Returns:
            List of (tweet_id, prompt) tuples
        """
        prompts = []
        
        for _, row in tweets_df.iterrows():
            tweet_id = str(row['tweet_id'])
            tweet_text = row['text_cleaned']
            
            # Get URL content if available (Pass 2)
            url_content = None
            if url_contents and tweet_id in url_contents:
                url_content = url_contents[tweet_id]
            
            prompt = self.build_prompt(tweet_id, tweet_text, url_content)
            prompts.append((tweet_id, prompt))
        
        return prompts
    
    def validate_prompt_length(self, prompt: str, max_tokens: int = 4000) -> bool:
        """
        Check if prompt is within token limits
        
        Args:
            prompt: Full prompt string
            max_tokens: Maximum allowed tokens (approximate)
            
        Returns:
            True if within limits
        """
        # Rough approximation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) / 4
        
        if estimated_tokens > max_tokens:
            logger.warning(f"Prompt exceeds token limit: {estimated_tokens:.0f} > {max_tokens}")
            return False
        
        return True
    
    def get_system_message(self) -> str:
        """
        Get system message for chat-based models (if needed)
        
        Returns:
            System message string
        """
        return "You are an expert annotator for cognitive perspectives in social media text. You always respond with valid JSON and follow instructions precisely."


def test_prompt_builder():
    """Test prompt building functionality"""
    print("="*70)
    print("TESTING PROMPT BUILDER")
    print("="*70)
    
    builder = PromptBuilder()
    
    # Test case 1: Simple tweet
    test_tweet_1 = {
        'id': 'tweet_001',
        'text': 'Climate change is real and we need to act now!'
    }
    
    prompt_1 = builder.build_prompt(test_tweet_1['id'], test_tweet_1['text'])
    
    print(f"\nTest 1: Simple tweet")
    print(f"Tweet: {test_tweet_1['text']}")
    print(f"Prompt length: {len(prompt_1)} chars")
    print(f"Within limits: {builder.validate_prompt_length(prompt_1)}")
    print(f"\nFirst 500 chars of prompt:")
    print(prompt_1[:500] + "...")
    
    # Test case 2: Tweet with question
    test_tweet_2 = {
        'id': 'tweet_002',
        'text': 'Does anyone know if this study is legitimate? Need sources.'
    }
    
    prompt_2 = builder.build_prompt(test_tweet_2['id'], test_tweet_2['text'])
    
    print(f"\n{'='*70}")
    print(f"Test 2: Tweet with question")
    print(f"Tweet: {test_tweet_2['text']}")
    print(f"Prompt length: {len(prompt_2)} chars")
    
    # Test case 3: Very long tweet
    test_tweet_3 = {
        'id': 'tweet_003',
        'text': 'This is a very long tweet ' * 20  # Simulate long content
    }
    
    prompt_3 = builder.build_prompt(test_tweet_3['id'], test_tweet_3['text'])
    
    print(f"\n{'='*70}")
    print(f"Test 3: Long tweet")
    print(f"Tweet length: {len(test_tweet_3['text'])} chars")
    print(f"Prompt length: {len(prompt_3)} chars")
    print(f"Within limits: {builder.validate_prompt_length(prompt_3)}")
    
    print(f"\n{'='*70}")
    print("PROMPT BUILDER TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Setup logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_prompt_builder()
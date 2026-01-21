"""
Output validator for perspective annotations
Validates JSON format, evidence spans, confidence scores
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import *

logger = logging.getLogger(__name__)


class AnnotationValidator:
    """
    Validates LLM-generated perspective annotations
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, enforce strict validation rules
        """
        self.strict_mode = strict_mode
        self.valid_labels = set(PERSPECTIVE_LABELS)
        
        # Statistics
        self.stats = {
            'total_validated': 0,
            'valid': 0,
            'invalid': 0,
            'errors_by_type': {}
        }
    
    def extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON from LLM response (handles markdown fences, preambles)
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted JSON string or None
        """
        # Remove markdown code fences
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Try to find JSON object
        # Pattern: { ... }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Return the largest match (most likely to be complete)
            return max(matches, key=len)
        
        # If no match, return cleaned response
        return response.strip()
    
    def validate_json_structure(self, json_str: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Validate JSON can be parsed and has correct structure
        
        Args:
            json_str: JSON string
            
        Returns:
            Tuple of (is_valid, parsed_json, error_message)
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {e}"
        
        # Check structure: should be dict with one key (tweet_id)
        if not isinstance(data, dict):
            return False, None, "JSON must be a dictionary"
        
        if len(data) != 1:
            return False, None, f"JSON should have exactly 1 tweet_id key, found {len(data)}"
        
        # Get the annotations list
        tweet_id = list(data.keys())[0]
        annotations = data[tweet_id]
        
        if not isinstance(annotations, list):
            return False, None, "Annotations must be a list"
        
        return True, data, ""
    
    def validate_evidence_span(
        self, 
        evidence_span: str, 
        tweet_text: str,
        allow_fuzzy: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate evidence span is a substring of tweet text
        
        Args:
            evidence_span: Claimed evidence span
            tweet_text: Original tweet text
            allow_fuzzy: Allow minor whitespace differences
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not evidence_span or not tweet_text:
            return False, "Empty evidence span or tweet text"
        
        # Exact match
        if evidence_span in tweet_text:
            return True, ""
        
        # Fuzzy match (if enabled)
        if allow_fuzzy:
            # Normalize whitespace
            evidence_normalized = ' '.join(evidence_span.split())
            tweet_normalized = ' '.join(tweet_text.split())
            
            if evidence_normalized in tweet_normalized:
                return True, ""
        
        return False, f"Evidence span not found in tweet: '{evidence_span[:50]}...'"
    
    def validate_confidence(self, confidence: float) -> Tuple[bool, str]:
        """
        Validate confidence score is in valid range
        
        Args:
            confidence: Confidence score
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(confidence, (int, float)):
            return False, f"Confidence must be numeric, got {type(confidence)}"
        
        if confidence < MIN_CONFIDENCE or confidence > MAX_CONFIDENCE:
            return False, f"Confidence {confidence} outside range [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]"
        
        return True, ""
    
    def validate_label(self, label: str) -> Tuple[bool, str]:
        """
        Validate label is from allowed set
        
        Args:
            label: Perspective label
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if label contains pipes (multi-label error)
        if '|' in label:
            return False, f"Invalid label format: contains '|'. Each annotation must have ONE label. Found: '{label}'"
        
        # Check if valid single label
        if label not in self.valid_labels:
            return False, f"Invalid label '{label}'. Must be one of: {self.valid_labels}"
        
        return True, ""
    
    def validate_annotation(
        self, 
        tweet_id: str,
        tweet_text: str,
        llm_response: str
    ) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Complete validation pipeline for a single annotation
        
        Args:
            tweet_id: Tweet identifier
            tweet_text: Original tweet text
            llm_response: Raw LLM response
            
        Returns:
            Tuple of (is_valid, parsed_annotation, list_of_errors)
        """
        self.stats['total_validated'] += 1
        errors = []
        
        # Step 1: Extract JSON
        json_str = self.extract_json_from_response(llm_response)
        if not json_str:
            errors.append("No JSON found in response")
            self._record_error("no_json")
            return False, None, errors
        
        # Step 2: Parse and validate JSON structure
        is_valid_json, parsed, error_msg = self.validate_json_structure(json_str)
        if not is_valid_json:
            errors.append(error_msg)
            self._record_error("invalid_json_structure")
            return False, None, errors
        
        # Get annotations list
        response_tweet_id = list(parsed.keys())[0]
        annotations = parsed[response_tweet_id]
        
        # Step 3: Validate each annotation
        valid_annotations = []
        
        for i, ann in enumerate(annotations):
            ann_errors = []
            
            # Check required fields
            required_fields = ['label', 'evidence_span', 'confidence', 'rationale']
            for field in required_fields:
                if field not in ann:
                    ann_errors.append(f"Missing field: {field}")
            
            if ann_errors:
                errors.extend([f"Annotation {i}: {e}" for e in ann_errors])
                continue
            
            # Validate label
            is_valid_label, error_msg = self.validate_label(ann['label'])
            if not is_valid_label:
                ann_errors.append(error_msg)
            
            # Validate evidence span
            is_valid_evidence, error_msg = self.validate_evidence_span(
                ann['evidence_span'],
                tweet_text,
                allow_fuzzy=ALLOW_WHITESPACE_DIFF
            )
            if not is_valid_evidence:
                ann_errors.append(error_msg)
            
            # Validate confidence
            is_valid_conf, error_msg = self.validate_confidence(ann['confidence'])
            if not is_valid_conf:
                ann_errors.append(error_msg)
            
            # Validate rationale exists and is non-empty
            if not ann.get('rationale') or len(ann['rationale'].strip()) < 10:
                ann_errors.append("Rationale missing or too short (<10 chars)")
            
            if ann_errors:
                errors.extend([f"Annotation {i}: {e}" for e in ann_errors])
                self._record_error("invalid_annotation_fields")
            else:
                valid_annotations.append(ann)
        
        # Step 4: Final validation
        if not valid_annotations and self.strict_mode:
            errors.append("No valid annotations found")
            self.stats['invalid'] += 1
            return False, None, errors
        
        # Reconstruct with validated annotations only
        validated_output = {
            tweet_id: valid_annotations
        }
        
        # Record success if at least one valid annotation
        if valid_annotations or not self.strict_mode:
            self.stats['valid'] += 1
            return True, validated_output, errors
        else:
            self.stats['invalid'] += 1
            return False, None, errors
    
    def _record_error(self, error_type: str):
        """Record error type for statistics"""
        self.stats['errors_by_type'][error_type] = \
            self.stats['errors_by_type'].get(error_type, 0) + 1
    
    def validate_batch(
        self,
        annotations: List[Tuple[str, str, str]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a batch of annotations
        
        Args:
            annotations: List of (tweet_id, tweet_text, llm_response) tuples
            
        Returns:
            Tuple of (valid_annotations, invalid_annotations_with_errors)
        """
        valid = []
        invalid = []
        
        for tweet_id, tweet_text, llm_response in annotations:
            is_valid, parsed, errors = self.validate_annotation(
                tweet_id, tweet_text, llm_response
            )
            
            if is_valid:
                valid.append({
                    'tweet_id': tweet_id,
                    'annotation': parsed,
                    'errors': errors  # May have non-critical errors
                })
            else:
                invalid.append({
                    'tweet_id': tweet_id,
                    'tweet_text': tweet_text,
                    'llm_response': llm_response,
                    'errors': errors
                })
        
        return valid, invalid
    
    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        stats = self.stats.copy()
        
        if stats['total_validated'] > 0:
            stats['validation_rate'] = (stats['valid'] / stats['total_validated']) * 100
        else:
            stats['validation_rate'] = 0.0
        
        return stats
    
    def print_statistics(self):
        """Print validation statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("VALIDATION STATISTICS")
        print("="*70)
        print(f"  Total validated:     {stats['total_validated']}")
        print(f"  Valid:               {stats['valid']}")
        print(f"  Invalid:             {stats['invalid']}")
        print(f"  Validation rate:     {stats['validation_rate']:.2f}%")
        
        if stats['errors_by_type']:
            print("\n  Errors by type:")
            for error_type, count in stats['errors_by_type'].items():
                print(f"    {error_type:30s}: {count:4d}")
        
        print("="*70)


def test_validator():
    """Test validation functionality"""
    print("="*70)
    print("TESTING ANNOTATION VALIDATOR")
    print("="*70)
    
    validator = AnnotationValidator(strict_mode=True)
    
    tweet_text = "Climate change is real and we need to act now!"
    tweet_id = "test_001"
    
    # Test 1: Valid annotation
    print("\nTest 1: Valid annotation")
    valid_response = '''
    {
      "test_001": [
        {
          "label": "Opinion/Belief",
          "evidence_span": "Climate change is real",
          "confidence": 0.85,
          "rationale": "The speaker expresses a definitive belief about climate change."
        }
      ]
    }
    '''
    
    is_valid, parsed, errors = validator.validate_annotation(tweet_id, tweet_text, valid_response)
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Test 2: Invalid evidence span
    print("\nTest 2: Invalid evidence span")
    invalid_response = '''
    {
      "test_001": [
        {
          "label": "Opinion/Belief",
          "evidence_span": "This span does not exist",
          "confidence": 0.85,
          "rationale": "Testing invalid evidence span."
        }
      ]
    }
    '''
    
    is_valid, parsed, errors = validator.validate_annotation(tweet_id, tweet_text, invalid_response)
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    
    # Test 3: Invalid confidence
    print("\nTest 3: Invalid confidence score")
    invalid_confidence = '''
    {
      "test_001": [
        {
          "label": "Opinion/Belief",
          "evidence_span": "Climate change is real",
          "confidence": 1.5,
          "rationale": "Testing invalid confidence."
        }
      ]
    }
    '''
    
    is_valid, parsed, errors = validator.validate_annotation(tweet_id, tweet_text, invalid_confidence)
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    
    # Test 4: Malformed JSON
    print("\nTest 4: Malformed JSON")
    malformed = "This is not JSON at all!"
    
    is_valid, parsed, errors = validator.validate_annotation(tweet_id, tweet_text, malformed)
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    
    # Print statistics
    validator.print_statistics()
    
    print("\n" + "="*70)
    print("VALIDATOR TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Setup logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_validator()
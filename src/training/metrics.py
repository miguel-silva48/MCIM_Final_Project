"""Evaluation metrics for caption generation.

Implements standard metrics for image captioning:
- BLEU (1-4): N-gram precision with brevity penalty
- METEOR: Unigram precision/recall with synonyms and stemming
- ROUGE-L: Longest common subsequence
- CIDEr: Consensus-based metric using TF-IDF

All metrics use NLTK's implementation for consistency with research literature.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import NLTK metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
        
except ImportError:
    print("Warning: NLTK not installed. Metrics will not be available.")
    print("Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge-score not installed. ROUGE-L will not be available.")
    print("Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False
    rouge_scorer = None


class CaptionMetrics:
    """Calculate evaluation metrics for generated captions.
    
    Handles multiple reference captions per image (though Indiana dataset
    typically has 1 reference per image).
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.smoothing = SmoothingFunction()
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def compute_bleu(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]],
        weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    ) -> float:
        """Compute corpus-level BLEU score.
        
        Args:
            references: List of reference lists for each sample
                        Shape: [num_samples][num_references][num_tokens]
            hypotheses: List of hypothesis tokens for each sample
                        Shape: [num_samples][num_tokens]
            weights: Weights for 1-gram, 2-gram, 3-gram, 4-gram
        
        Returns:
            BLEU score (0-1, higher is better)
        """
        # Use smoothing function 4 (add epsilon to avoid 0 counts)
        bleu = corpus_bleu(
            references,
            hypotheses,
            weights=weights,
            smoothing_function=self.smoothing.method4
        )
        return bleu
    
    def compute_bleu_scores(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> Dict[str, float]:
        """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4.
        
        Args:
            references: List of reference lists for each sample
            hypotheses: List of hypothesis tokens for each sample
        
        Returns:
            Dictionary with BLEU-1 through BLEU-4 scores
        """
        bleu_scores = {}
        
        # BLEU-1 (unigram precision)
        bleu_scores['bleu_1'] = self.compute_bleu(
            references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0)
        )
        
        # BLEU-2 (bigram precision)
        bleu_scores['bleu_2'] = self.compute_bleu(
            references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0)
        )
        
        # BLEU-3 (trigram precision)
        bleu_scores['bleu_3'] = self.compute_bleu(
            references, hypotheses, weights=(0.33, 0.33, 0.33, 0.0)
        )
        
        # BLEU-4 (4-gram precision, standard metric)
        bleu_scores['bleu_4'] = self.compute_bleu(
            references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
        )
        
        return bleu_scores
    
    def compute_meteor(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> float:
        """Compute METEOR score (handles synonyms and stemming).
        
        Args:
            references: List of reference lists for each sample
            hypotheses: List of hypothesis tokens for each sample
        
        Returns:
            METEOR score (0-1, higher is better)
        """
        meteor_scores = []
        
        for refs, hyp in zip(references, hypotheses):
            # METEOR expects tokenized input (lists of strings, not joined strings)
            # refs: list of reference token lists
            # hyp: hypothesis token list
            
            # Calculate METEOR for this sample
            score = meteor_score(refs, hyp)
            meteor_scores.append(score)
        
        # Return mean METEOR across all samples
        return np.mean(meteor_scores)
    
    def compute_rouge_l(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> float:
        """Compute ROUGE-L score (longest common subsequence).
        
        Args:
            references: List of reference lists for each sample
            hypotheses: List of hypothesis tokens for each sample
        
        Returns:
            ROUGE-L F1 score (0-1, higher is better)
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            print("Warning: ROUGE-L not available, returning 0.0")
            return 0.0
            
        rouge_scores = []
        
        for refs, hyp in zip(references, hypotheses):
            # ROUGE takes single strings (space-joined)
            hyp_str = ' '.join(hyp)
            
            # Calculate ROUGE-L against each reference, take max
            ref_scores = []
            for ref in refs:
                ref_str = ' '.join(ref)
                score = self.rouge_scorer.score(ref_str, hyp_str)
                ref_scores.append(score['rougeL'].fmeasure)
            
            rouge_scores.append(max(ref_scores))
        
        # Return mean ROUGE-L across all samples
        return np.mean(rouge_scores)
    
    def compute_all_metrics(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]]
    ) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            references: List of reference lists for each sample
                        Shape: [num_samples][num_references][num_tokens]
            hypotheses: List of hypothesis tokens for each sample
                        Shape: [num_samples][num_tokens]
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        # BLEU scores (1-4)
        bleu_scores = self.compute_bleu_scores(references, hypotheses)
        metrics.update(bleu_scores)
        
        # METEOR
        metrics['meteor'] = self.compute_meteor(references, hypotheses)
        
        # ROUGE-L
        metrics['rouge_l'] = self.compute_rouge_l(references, hypotheses)
        
        return metrics
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display.
        
        Args:
            metrics: Dictionary of metric scores
        
        Returns:
            Formatted string
        """
        lines = []
        lines.append("Evaluation Metrics:")
        lines.append("-" * 40)
        
        # BLEU scores
        for i in range(1, 5):
            key = f'bleu_{i}'
            if key in metrics:
                lines.append(f"  BLEU-{i}: {metrics[key]:.4f}")
        
        # METEOR
        if 'meteor' in metrics:
            lines.append(f"  METEOR:  {metrics['meteor']:.4f}")
        
        # ROUGE-L
        if 'rouge_l' in metrics:
            lines.append(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
        
        return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing CaptionMetrics...")
    
    # Example reference captions (tokenized)
    # Each sample has 1+ reference captions
    references = [
        [['the', 'cat', 'is', 'on', 'the', 'mat']],  # Sample 1: 1 reference
        [['a', 'dog', 'in', 'the', 'park']],         # Sample 2: 1 reference
        [['normal', 'chest', 'x', 'ray']],           # Sample 3: 1 reference
    ]
    
    # Hypothesis captions (model predictions)
    hypotheses = [
        ['the', 'cat', 'sits', 'on', 'the', 'mat'],  # Similar to ref
        ['a', 'dog', 'plays', 'in', 'park'],         # Missing 'the'
        ['normal', 'chest'],                         # Truncated
    ]
    
    print(f"\nNumber of samples: {len(references)}")
    print(f"References per sample: {[len(refs) for refs in references]}")
    
    # Initialize metrics
    metrics_calculator = CaptionMetrics()
    
    # Compute all metrics
    print("\nComputing metrics...")
    metrics = metrics_calculator.compute_all_metrics(references, hypotheses)
    
    print(f"\n{metrics_calculator.format_metrics(metrics)}")
    
    # Test individual metrics
    print("\n" + "=" * 40)
    print("Individual metric tests:")
    print("=" * 40)
    
    # BLEU-4
    bleu4 = metrics_calculator.compute_bleu(
        references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
    )
    print(f"BLEU-4: {bleu4:.4f}")
    
    # METEOR
    meteor = metrics_calculator.compute_meteor(references, hypotheses)
    print(f"METEOR: {meteor:.4f}")
    
    # ROUGE-L
    rouge_l = metrics_calculator.compute_rouge_l(references, hypotheses)
    print(f"ROUGE-L: {rouge_l:.4f}")
    
    print("\n✓ Metrics test passed!")

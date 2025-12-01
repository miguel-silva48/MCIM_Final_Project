"""
N-gram extraction and analysis functions.
"""

import pandas as pd
import nltk
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional
from .text_preprocessing import tokenize_text, download_nltk_data


def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: N-gram size (2 for bigrams, 3 for trigrams, etc.)
        
    Returns:
        List of n-gram tuples
    """
    download_nltk_data()
    
    if len(tokens) < n:
        return []
    
    return list(nltk.ngrams(tokens, n))


def get_ngram_frequencies(
    texts: List[str],
    n: int,
    top_k: Optional[int] = None,
    lowercase: bool = True,
    remove_punctuation: bool = False
) -> Counter:
    """
    Calculate n-gram frequencies across multiple texts.
    
    Args:
        texts: List of text strings
        n: N-gram size
        top_k: If specified, return only top k most common n-grams
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Counter object with n-gram frequencies
    """
    all_ngrams = []
    
    for text in texts:
        tokens = tokenize_text(text, lowercase=lowercase, remove_punctuation=remove_punctuation)
        ngrams = extract_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    
    ngram_counter = Counter(all_ngrams)
    
    if top_k is not None:
        # Return only top k
        return Counter(dict(ngram_counter.most_common(top_k)))
    
    return ngram_counter


def save_ngram_report(
    ngram_counter: Counter,
    output_path: Path,
    n: int
) -> None:
    """
    Save n-gram frequency report to CSV file.
    
    Args:
        ngram_counter: Counter object with n-gram frequencies
        output_path: Path to save CSV file
        n: N-gram size (for metadata)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total count and frequencies
    total_count = sum(ngram_counter.values())
    
    # Prepare data for DataFrame
    data = []
    for ngram, count in ngram_counter.most_common():
        ngram_str = ' '.join(ngram)
        frequency_pct = (count / total_count) * 100
        data.append({
            'ngram': ngram_str,
            'count': count,
            'frequency_pct': frequency_pct
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} {n}-grams to {output_path}")


if __name__ == '__main__':
    # Quick test
    sample_texts = [
        "No acute cardiopulmonary abnormality.",
        "No acute findings in the chest.",
        "Clear lungs, no pleural effusion."
    ]
    
    # Unigrams
    unigrams = get_ngram_frequencies(sample_texts, n=1, top_k=10)
    print("Top unigrams:", unigrams.most_common(5))
    
    # Bigrams
    bigrams = get_ngram_frequencies(sample_texts, n=2, top_k=10)
    print("Top bigrams:", bigrams.most_common(5))
    
    # Trigrams
    trigrams = get_ngram_frequencies(sample_texts, n=3, top_k=10)
    print("Top trigrams:", trigrams.most_common(5))

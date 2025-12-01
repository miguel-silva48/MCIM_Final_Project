"""
Text preprocessing and vocabulary building functions.
"""

import re
import pandas as pd
import nltk
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

# Download required NLTK data (will be called in notebook)
def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {package}: {e}")


def extract_report_text(
    row: pd.Series,
    use_findings: bool = True,
    use_impression: bool = True,
    separator: str = ' '
) -> str:
    """
    Extract report text from a DataFrame row.
    
    Args:
        row: DataFrame row containing report fields
        use_findings: Whether to include the 'findings' field
        use_impression: Whether to include the 'impression' field
        separator: String to join findings and impression
        
    Returns:
        Combined report text, or empty string if both fields are NaN
    """
    parts = []
    
    if use_findings and pd.notna(row.get('findings', None)):
        parts.append(str(row['findings']))
    
    if use_impression and pd.notna(row.get('impression', None)):
        parts.append(str(row['impression']))
    
    return separator.join(parts)


def calculate_censoring_ratio(text: str) -> float:
    """
    Calculate the ratio of censored tokens (XXXX) to total words.
    
    Args:
        text: Report text
        
    Returns:
        Ratio of XXXX tokens to total tokens (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0
    
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', str(text))
    
    if len(tokens) == 0:
        return 0.0
    
    xxxx_count = sum(1 for token in tokens if token == 'XXXX')
    
    return xxxx_count / len(tokens)


def filter_reports_by_censoring(
    df: pd.DataFrame,
    text_column: str,
    max_ratio: float = 0.3
) -> pd.DataFrame:
    """
    Filter out reports with excessive censoring (XXXX).
    
    Args:
        df: DataFrame containing report text
        text_column: Name of column with text to check
        max_ratio: Maximum allowed ratio of XXXX tokens (default 0.3 = 30%)
        
    Returns:
        Filtered DataFrame
    """
    censoring_ratios = df[text_column].apply(calculate_censoring_ratio)
    return df[censoring_ratios <= max_ratio].copy()


def clean_metadata_text(text: str) -> str:
    """
    Remove common metadata phrases that appear in reports.
    
    Args:
        text: Report text
        
    Returns:
        Cleaned text
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Patterns to remove (case-insensitive)
    patterns_to_remove = [
        r'XXXX-year-old\s+\w+',  # "XXXX-year-old female"
        r'\d+-year-old\s+\w+',   # "45-year-old female"
        r'PA and lateral',
        r'Xray Chest PA and Lateral',
        r'Chest, 2 views',
        r'frontal and lateral',
        r'dated XXXX',
        r'at XXXX hours',
        r'XXXX, XXXX',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> List[str]:
    """
    Tokenize text using NLTK word tokenizer.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation tokens
        
    Returns:
        List of tokens
    """
    download_nltk_data()
    
    if not text or pd.isna(text):
        return []
    
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    if remove_punctuation:
        # Keep only alphanumeric tokens
        tokens = [t for t in tokens if t.isalnum()]
    
    return tokens


def build_vocabulary(
    texts: List[str],
    min_freq: int = 5,
    lowercase: bool = True,
    remove_punctuation: bool = False
) -> Tuple[Counter, Dict[str, int]]:
    """
    Build vocabulary from a list of texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a token to be included
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Tuple of (token_counter, vocab_dict)
        - token_counter: Counter object with all token frequencies
        - vocab_dict: Dictionary mapping tokens to indices (only tokens >= min_freq)
    """
    # Collect all tokens
    all_tokens = []
    for text in texts:
        tokens = tokenize_text(text, lowercase=lowercase, remove_punctuation=remove_punctuation)
        all_tokens.extend(tokens)
    
    # Count frequencies
    token_counter = Counter(all_tokens)
    
    # Build vocabulary with special tokens
    vocab = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
    }
    
    # Add tokens that meet minimum frequency
    idx = len(vocab)
    for token, count in token_counter.most_common():
        if count >= min_freq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    
    return token_counter, vocab


def save_vocabulary(
    vocab: Dict[str, int],
    token_counter: Counter,
    output_path: Path
) -> None:
    """
    Save vocabulary to a text file with frequencies.
    
    Args:
        vocab: Vocabulary dictionary (token -> index)
        token_counter: Token frequency counter
        output_path: Path to save vocabulary file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("token\tindex\tcount\n")
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            count = token_counter.get(token, 0)
            f.write(f"{token}\t{idx}\t{count}\n")


if __name__ == '__main__':
    # Quick test
    sample_text = "The cardiac silhouette and mediastinum size are within normal limits. No pneumothorax."
    
    tokens = tokenize_text(sample_text)
    print(f"Tokens: {tokens}")
    
    ratio = calculate_censoring_ratio("This is XXXX test XXXX more XXXX")
    print(f"Censoring ratio: {ratio:.2f}")
    
    cleaned = clean_metadata_text("XXXX-year-old male with chest pain dated XXXX.")
    print(f"Cleaned: {cleaned}")

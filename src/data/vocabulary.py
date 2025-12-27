"""Vocabulary class for text tokenization and encoding."""

import os
from typing import List, Dict
from pathlib import Path


class Vocabulary:
    """Vocabulary for text to token conversion.
    
    This class manages the mapping between words and integer indices
    for the caption generation task.
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    UNK_TOKEN = '<UNK>'
    
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3
    
    def __init__(self, vocab_file: str):
        """Initialize vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary.txt file with format:
                        token\tindex\tcount
        """
        self.vocab_file = vocab_file
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.token_counts: Dict[str, int] = {}
        
        self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load vocabulary from file."""
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")
        
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    token, idx, count = parts[0], int(parts[1]), int(parts[2])
                    self.token_to_idx[token] = idx
                    self.idx_to_token[idx] = token
                    self.token_counts[token] = count
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to list of token indices.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add <START> and <END> tokens
        
        Returns:
            List of token indices
        """
        # Tokenize by splitting on whitespace
        tokens = text.lower().split()
        
        # Convert to indices
        indices = [self.token_to_idx.get(token, self.UNK_IDX) for token in tokens]
        
        # Add special tokens if requested
        if add_special_tokens:
            indices = [self.START_IDX] + indices + [self.END_IDX]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token indices to text.
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.UNK_TOKEN)
            
            # Skip special tokens if requested
            if skip_special_tokens and token in [
                self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN
            ]:
                continue
            
            # Stop at end token
            if token == self.END_TOKEN:
                break
            
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_idx)
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, file={self.vocab_file})"


if __name__ == '__main__':
    # Test vocabulary loading
    vocab_file = "data/processed/first_frontal_impression/vocabulary.txt"
    
    if os.path.exists(vocab_file):
        print("Testing Vocabulary class...")
        vocab = Vocabulary(vocab_file)
        
        print(f"\nVocabulary loaded:")
        print(f"  Size: {len(vocab)}")
        print(f"  Special tokens:")
        print(f"    PAD: {vocab.PAD_IDX}")
        print(f"    START: {vocab.START_IDX}")
        print(f"    END: {vocab.END_IDX}")
        print(f"    UNK: {vocab.UNK_IDX}")
        
        # Test encoding
        test_text = "no acute cardiopulmonary disease"
        encoded = vocab.encode(test_text)
        print(f"\nTest encoding:")
        print(f"  Input: '{test_text}'")
        print(f"  Encoded: {encoded}")
        
        # Test decoding
        decoded = vocab.decode(encoded)
        print(f"  Decoded: '{decoded}'")
        
        # Test with unknown token
        test_text_unk = "no acute xyzabc disease"
        encoded_unk = vocab.encode(test_text_unk)
        print(f"\nTest with unknown token:")
        print(f"  Input: '{test_text_unk}'")
        print(f"  Encoded: {encoded_unk}")
        print(f"  Decoded: '{vocab.decode(encoded_unk)}'")
        
        print("\n✓ Vocabulary test passed!")
    else:
        print(f"Vocabulary file not found: {vocab_file}")

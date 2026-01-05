"""
LSTM-based caption decoder with attention mechanism.

Generates captions word-by-word while attending to spatial image features.
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention import create_attention


class CaptionDecoder(nn.Module):
    """
    LSTM decoder with attention for generating image captions.
    
    At each time step:
    1. Embed current word
    2. Attend over image features
    3. Pass embedding + context through LSTM
    4. Project to vocabulary for next word prediction
    """
    
    def __init__(
        self,
        attention_type: str,
        attention_dim: int,
        embedding_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.5
    ):
        """
        Initialize caption decoder.
        
        Args:
            attention_type: Type of attention ("bahdanau" or "luong")
            attention_dim: Dimension of attention hidden layer
            embedding_dim: Dimension of word embeddings
            decoder_dim: Dimension of LSTM hidden state
            vocab_size: Size of vocabulary
            encoder_dim: Dimension of encoder features (default: 1024 for DenseNet)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CaptionDecoder, self).__init__()
        
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # Attention module
        self.attention = create_attention(
            attention_type=attention_type,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM cell
        # Input: concatenation of word embedding + context vector
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Initialize LSTM hidden state from encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Output layer: project LSTM hidden state to vocabulary
        self.fc_out = nn.Linear(decoder_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding and output layer weights."""
        # Initialize embeddings with uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        # Initialize output layer
        self.fc_out.bias.data.fill_(0)
        self.fc_out.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from encoder features.
        
        Args:
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
        
        Returns:
            h: Initial hidden state [num_layers, batch_size, decoder_dim]
            c: Initial cell state [num_layers, batch_size, decoder_dim]
        """
        # Average pool over spatial dimensions
        mean_encoder_out = encoder_out.mean(dim=1)  # [batch, encoder_dim]
        
        # Project to decoder dimension
        h = self.init_h(mean_encoder_out)  # [batch, decoder_dim]
        c = self.init_c(mean_encoder_out)  # [batch, decoder_dim]
        
        # Expand for multiple layers if needed
        if self.num_layers > 1:
            h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, decoder_dim]
            c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            h = h.unsqueeze(0)  # [1, batch, decoder_dim]
            c = c.unsqueeze(0)
        
        return h, c
    
    def forward_step(
        self,
        word: torch.Tensor,
        encoder_out: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            word: Current word indices [batch_size]
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
            hidden_state: Tuple of (h, c) LSTM states
        
        Returns:
            predictions: Vocabulary predictions [batch_size, vocab_size]
            hidden_state: Updated LSTM states
            attention_weights: Attention weights [batch_size, num_pixels]
        """
        h, c = hidden_state
        
        # Get decoder hidden state for attention (use last layer)
        decoder_hidden = h[-1]  # [batch, decoder_dim]
        
        # Apply attention
        context, attention_weights = self.attention(encoder_out, decoder_hidden)
        
        # Embed current word
        embeddings = self.embedding(word)  # [batch, embedding_dim]
        embeddings = self.dropout(embeddings)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embeddings, context], dim=1)  # [batch, embedding_dim + encoder_dim]
        lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, embedding_dim + encoder_dim]
        
        # LSTM forward
        lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
        lstm_out = lstm_out.squeeze(1)  # [batch, decoder_dim]
        
        # Project to vocabulary
        predictions = self.fc_out(lstm_out)  # [batch, vocab_size]
        
        return predictions, (h, c), attention_weights
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
            captions: Caption tokens [batch_size, max_length]
                     Includes <START> token but not <END> in input
            caption_lengths: Actual caption lengths [batch_size]
        
        Returns:
            predictions: Predictions for all time steps [batch_size, max_length-1, vocab_size]
            attention_weights: Attention weights [batch_size, max_length-1, num_pixels]
            sorted_captions: Captions sorted by length (for pack_padded_sequence)
            sorted_lengths: Lengths sorted in descending order
        """
        batch_size = encoder_out.size(0)
        max_length = captions.size(1)
        
        # Sort captions by length for pack_padded_sequence
        caption_lengths, sort_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]
        
        # Initialize hidden state
        h, c = self.init_hidden_state(encoder_out)

        # Decode for the length of targets
        # Input captions: [START, w1, ..., w(N-1)] has max_length tokens (padded)
        # We create predictions for max_length-1 to match sorted_captions[:, 1:]
        # But only iterate up to the longest actual caption to avoid empty tensors
        decode_length = max_length - 1  # Size of output tensor (matches target)
        decode_lengths = caption_lengths.tolist()  # Actual lengths (already excludes END)
        # Cap max_decode_length to not exceed predictions tensor size
        max_decode_length = min(max(decode_lengths), decode_length)  # Don't exceed predictions size

        # Create tensors to hold predictions and attention weights (padded size)
        predictions = torch.zeros(batch_size, decode_length, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, decode_length, encoder_out.size(1)).to(encoder_out.device)

        # Teacher forcing: feed ground truth at each step
        # Only iterate up to max actual length to avoid accessing out of bounds
        for t in range(max_decode_length):
            # Only process sequences that haven't finished yet
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Forward step
            preds, (h, c), attention_weights = self.forward_step(
                word=captions[:batch_size_t, t],
                encoder_out=encoder_out[:batch_size_t],
                hidden_state=(h[:, :batch_size_t, :].contiguous(), c[:, :batch_size_t, :].contiguous())
            )
            
            # Store predictions and attention
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = attention_weights
        
        return predictions, alphas, captions, caption_lengths


if __name__ == '__main__':
    # Test decoder
    print("Testing Caption Decoder:")
    
    batch_size = 2
    num_pixels = 49  # 7x7
    encoder_dim = 1024
    vocab_size = 1000
    max_length = 20
    
    # Config
    attention_type = "bahdanau"
    attention_dim = 512
    embedding_dim = 512
    decoder_dim = 1024
    num_layers = 1
    dropout = 0.5
    
    # Create decoder
    decoder = CaptionDecoder(
        attention_type=attention_type,
        attention_dim=attention_dim,
        embedding_dim=embedding_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    print(f"  Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,}")
    
    # Dummy inputs
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    captions = torch.randint(0, vocab_size, (batch_size, max_length))
    caption_lengths = torch.tensor([max_length, max_length - 5])
    
    # Forward pass
    predictions, alphas, sorted_captions, sorted_lengths = decoder(
        encoder_out, captions, caption_lengths
    )
    
    print(f"\nForward pass test:")
    print(f"  Encoder output shape: {encoder_out.shape}")
    print(f"  Input captions shape: {captions.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Attention weights shape: {alphas.shape}")
    print(f"  Attention weights sum (first timestep): {alphas[0, 0, :].sum():.4f}")  # Should be ~1.0

"""
Attention mechanism for image captioning.

Implements Bahdanau (additive) attention that allows the decoder to focus
on different spatial regions of the image at each decoding step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism (additive attention).
    
    Computes attention weights over spatial image features based on
    the decoder's hidden state.
    
    Reference:
        Bahdanau et al., "Neural Machine Translation by Jointly Learning to
        Align and Translate", ICLR 2015
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int
    ):
        """
        Initialize attention module.
        
        Args:
            encoder_dim: Dimension of encoder features (e.g., 1024 for DenseNet-121)
            decoder_dim: Dimension of decoder hidden state (e.g., 1024)
            attention_dim: Dimension of attention hidden layer (e.g., 512)
        """
        super(BahdanauAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Linear layers for computing attention scores
        # Transform encoder features to attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        
        # Transform decoder hidden state to attention space
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        
        # Final layer to compute scalar attention score
        self.full_att = nn.Linear(attention_dim, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute attention-weighted context vector.
        
        Args:
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
                        where num_pixels = spatial_h * spatial_w (e.g., 7*7=49)
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
        
        Returns:
            context: Attention-weighted context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, num_pixels]
        """
        # Transform encoder features
        att1 = self.encoder_att(encoder_out)  # [batch, num_pixels, attention_dim]
        
        # Transform decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # [batch, decoder_dim]
        
        # Add decoder state to each pixel (broadcasting)
        # att2 shape: [batch, attention_dim] -> [batch, 1, attention_dim]
        att2 = att2.unsqueeze(1)  # [batch, 1, attention_dim]
        
        # Additive attention: tanh(W1*encoder + W2*decoder)
        att_combined = self.relu(att1 + att2)  # [batch, num_pixels, attention_dim]
        
        # Compute attention scores
        att_scores = self.full_att(att_combined)  # [batch, num_pixels, 1]
        att_scores = att_scores.squeeze(2)  # [batch, num_pixels]
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(att_scores)  # [batch, num_pixels]
        
        # Compute context vector as weighted sum of encoder features
        # attention_weights: [batch, num_pixels] -> [batch, num_pixels, 1]
        attention_weights_expanded = attention_weights.unsqueeze(2)
        
        # Weighted sum: sum over num_pixels dimension
        context = (encoder_out * attention_weights_expanded).sum(dim=1)  # [batch, encoder_dim]
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong attention mechanism (multiplicative attention).
    
    Alternative attention mechanism - simpler but can work just as well.
    Included for potential future use.
    
    Reference:
        Luong et al., "Effective Approaches to Attention-based Neural Machine
        Translation", EMNLP 2015
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int
    ):
        """
        Initialize Luong attention.
        
        Args:
            encoder_dim: Dimension of encoder features
            decoder_dim: Dimension of decoder hidden state
        """
        super(LuongAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Linear layer to align dimensions if encoder_dim != decoder_dim
        if encoder_dim != decoder_dim:
            self.align = nn.Linear(decoder_dim, encoder_dim)
        else:
            self.align = None
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multiplicative attention.
        
        Args:
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
        
        Returns:
            context: Attention-weighted context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, num_pixels]
        """
        # Align decoder hidden to encoder dimension if needed
        if self.align is not None:
            decoder_hidden = self.align(decoder_hidden)  # [batch, encoder_dim]
        
        # Compute dot product attention scores
        # decoder_hidden: [batch, encoder_dim] -> [batch, encoder_dim, 1]
        decoder_hidden = decoder_hidden.unsqueeze(2)
        
        # Batch matrix multiplication: [batch, num_pixels, encoder_dim] @ [batch, encoder_dim, 1]
        att_scores = torch.bmm(encoder_out, decoder_hidden)  # [batch, num_pixels, 1]
        att_scores = att_scores.squeeze(2)  # [batch, num_pixels]
        
        # Apply softmax
        attention_weights = self.softmax(att_scores)  # [batch, num_pixels]
        
        # Compute context vector
        attention_weights_expanded = attention_weights.unsqueeze(2)  # [batch, num_pixels, 1]
        context = (encoder_out * attention_weights_expanded).sum(dim=1)  # [batch, encoder_dim]
        
        return context, attention_weights


def create_attention(
    attention_type: str,
    encoder_dim: int,
    decoder_dim: int,
    attention_dim: int = 512
) -> nn.Module:
    """
    Factory function to create attention module.
    
    Args:
        attention_type: Type of attention ("bahdanau" or "luong")
        encoder_dim: Dimension of encoder features
        decoder_dim: Dimension of decoder hidden state
        attention_dim: Dimension of attention hidden layer (only for Bahdanau)
    
    Returns:
        Attention module instance
    """
    if attention_type.lower() == "bahdanau":
        return BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
    elif attention_type.lower() == "luong":
        return LuongAttention(encoder_dim, decoder_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}. Choose 'bahdanau' or 'luong'.")


if __name__ == '__main__':
    # Test Bahdanau attention
    print("Testing Bahdanau Attention:")
    encoder_dim = 1024
    decoder_dim = 1024
    attention_dim = 512
    batch_size = 2
    num_pixels = 49  # 7x7
    
    attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
    
    # Dummy inputs
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    decoder_hidden = torch.randn(batch_size, decoder_dim)
    
    # Forward pass
    context, attention_weights = attention(encoder_out, decoder_hidden)
    
    print(f"  Encoder output shape: {encoder_out.shape}")
    print(f"  Decoder hidden shape: {decoder_hidden.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Attention weights sum: {attention_weights.sum(dim=1)}")  # Should be ~1.0
    
    # Test Luong attention
    print("\nTesting Luong Attention:")
    attention_luong = LuongAttention(encoder_dim, decoder_dim)
    context_l, attention_weights_l = attention_luong(encoder_out, decoder_hidden)
    
    print(f"  Context shape: {context_l.shape}")
    print(f"  Attention weights shape: {attention_weights_l.shape}")
    print(f"  Attention weights sum: {attention_weights_l.sum(dim=1)}")

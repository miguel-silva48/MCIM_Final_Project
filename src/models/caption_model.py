"""
Complete encoder-decoder model for image captioning.

Combines ImageEncoder and CaptionDecoder into a single model with
training and inference methods.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .encoder import ImageEncoder
from .decoder import CaptionDecoder


class EncoderDecoderModel(nn.Module):
    """
    Complete image captioning model.
    
    Combines encoder (DenseNet-121) and decoder (LSTM with attention)
    into a unified model for training and inference.
    """
    
    def __init__(
        self,
        # Encoder config
        encoder_architecture: str = "densenet121",
        encoder_pretrained: bool = True,
        encoder_freeze: bool = True,
        encoder_freeze_until: Optional[str] = None,
        encoder_feature_dim: int = 1024,
        # Decoder config
        attention_type: str = "bahdanau",
        attention_dim: int = 512,
        embedding_dim: int = 512,
        decoder_dim: int = 1024,
        num_decoder_layers: int = 1,
        dropout: float = 0.5,
        vocab_size: int = 1000,
        # Special tokens
        pad_idx: int = 0,
        start_idx: int = 1,
        end_idx: int = 2,
        unk_idx: int = 3
    ):
        """
        Initialize encoder-decoder model.
        
        Args:
            encoder_architecture: Encoder architecture name
            encoder_pretrained: Use pretrained encoder weights
            encoder_freeze: Freeze encoder weights initially
            encoder_freeze_until: Layer name to freeze up to
            encoder_feature_dim: Encoder output feature dimension
            attention_type: Type of attention mechanism
            attention_dim: Attention hidden dimension
            embedding_dim: Word embedding dimension
            decoder_dim: Decoder LSTM hidden dimension
            num_decoder_layers: Number of LSTM layers
            dropout: Dropout probability
            vocab_size: Vocabulary size
            pad_idx: Padding token index
            start_idx: Start-of-sequence token index
            end_idx: End-of-sequence token index
            unk_idx: Unknown token index
        """
        super(EncoderDecoderModel, self).__init__()
        
        # Store special token indices
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.unk_idx = unk_idx
        self.vocab_size = vocab_size
        
        # Create encoder
        self.encoder = ImageEncoder(
            architecture=encoder_architecture,
            pretrained=encoder_pretrained,
            freeze_backbone=encoder_freeze,
            freeze_until_layer=encoder_freeze_until,
            output_feature_dim=encoder_feature_dim
        )
        
        # Create decoder
        self.decoder = CaptionDecoder(
            attention_type=attention_type,
            attention_dim=attention_dim,
            embedding_dim=embedding_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_feature_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.encoder_feature_dim = encoder_feature_dim
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            captions: Target captions [batch_size, max_length]
            caption_lengths: Caption lengths [batch_size]
        
        Returns:
            predictions: Vocabulary predictions [batch_size, max_length-1, vocab_size]
            attention_weights: Attention weights [batch_size, max_length-1, num_pixels]
            sorted_captions: Captions sorted by length
            sorted_lengths: Lengths sorted in descending order
        """
        # Encode images
        encoder_out = self.encoder(images)  # [batch, encoder_dim, 7, 7]
        
        # Reshape encoder output for decoder
        # [batch, encoder_dim, H, W] -> [batch, H*W, encoder_dim]
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        encoder_out = encoder_out.reshape(batch_size, encoder_dim, -1)  # [batch, encoder_dim, H*W]
        encoder_out = encoder_out.permute(0, 2, 1)  # [batch, H*W, encoder_dim]
        
        # Decode captions
        predictions, attention_weights, sorted_captions, sorted_lengths = self.decoder(
            encoder_out, captions, caption_lengths
        )
        
        return predictions, attention_weights, sorted_captions, sorted_lengths
    
    def generate_caption(
        self,
        image: torch.Tensor,
        max_length: int = 50,
        beam_size: int = 3,
        length_penalty: float = 0.0
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generate caption for a single image using beam search.
        
        Args:
            image: Input image [1, 3, H, W] or [3, H, W]
            max_length: Maximum caption length
            beam_size: Beam size for beam search (1 = greedy)
            length_penalty: Length penalty factor
        
        Returns:
            caption: Generated caption token indices
            attention_weights: Attention weights for each timestep
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Encode image
        with torch.no_grad():
            encoder_out = self.encoder(image)  # [1, encoder_dim, 7, 7]
            
            # Reshape for decoder
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(1)
            encoder_out = encoder_out.reshape(batch_size, encoder_dim, -1)
            encoder_out = encoder_out.permute(0, 2, 1)  # [1, num_pixels, encoder_dim]
            
            if beam_size == 1:
                # Greedy decoding
                caption, attention_weights = self._greedy_decode(
                    encoder_out, max_length
                )
            else:
                # Beam search
                caption, attention_weights = self._beam_search(
                    encoder_out, max_length, beam_size, length_penalty
                )
        
        return caption, attention_weights
    
    def _greedy_decode(
        self,
        encoder_out: torch.Tensor,
        max_length: int
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Greedy decoding (fastest but may not be optimal).
        
        Args:
            encoder_out: Encoded image features [1, num_pixels, encoder_dim]
            max_length: Maximum caption length
        
        Returns:
            caption: Generated caption token indices
            attention_weights: Attention weights for each timestep
        """
        caption = []
        attention_weights = []
        
        # Initialize hidden state
        h, c = self.decoder.init_hidden_state(encoder_out)
        
        # Start with <START> token
        word = torch.tensor([self.start_idx]).to(encoder_out.device)
        
        for _ in range(max_length):
            # Decode one step
            predictions, (h, c), alpha = self.decoder.forward_step(
                word, encoder_out, (h, c)
            )
            
            # Get most likely word
            word = predictions.argmax(dim=1)
            word_idx = word.item()
            
            # Store
            caption.append(word_idx)
            attention_weights.append(alpha.squeeze(0).cpu())
            
            # Stop if <END> token generated
            if word_idx == self.end_idx:
                break
        
        return caption, attention_weights
    
    def _beam_search(
        self,
        encoder_out: torch.Tensor,
        max_length: int,
        beam_size: int,
        length_penalty: float
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Beam search decoding.
        
        Args:
            encoder_out: Encoded image features [1, num_pixels, encoder_dim]
            max_length: Maximum caption length
            beam_size: Number of beams
            length_penalty: Length penalty factor (0.0 = no penalty)
        
        Returns:
            caption: Best caption token indices
            attention_weights: Attention weights for best caption
        """
        # Initialize with <START> token
        k = beam_size
        vocab_size = self.vocab_size
        
        # Expand encoder output for beam search
        encoder_out = encoder_out.expand(k, -1, -1)  # [k, num_pixels, encoder_dim]
        
        # Initialize hidden states
        h, c = self.decoder.init_hidden_state(encoder_out)
        
        # Tensor to store top k sequences; now they're all <START>
        k_prev_words = torch.full((k, 1), self.start_idx, dtype=torch.long).to(encoder_out.device)
        
        # Tensor to store top k sequences' scores
        seqs_scores = torch.zeros(k, 1).to(encoder_out.device)
        
        # Lists to store completed sequences and their scores
        complete_seqs = []
        complete_seqs_scores = []
        complete_seqs_alpha = []
        
        # Start decoding
        step = 1
        
        # Store attention weights
        seqs_alpha = [[] for _ in range(k)]
        
        while True:
            # Get current word
            word = k_prev_words[:, -1]  # [k]
            
            # Decode one step
            predictions, (h, c), alpha = self.decoder.forward_step(
                word, encoder_out, (h, c)
            )
            
            scores = torch.log_softmax(predictions, dim=1)  # [k, vocab_size]
            
            # Add previous scores
            scores = seqs_scores.expand_as(scores) + scores  # [k, vocab_size]
            
            # For first step, all k beams are same
            if step == 1:
                top_scores, top_words = scores[0].topk(k, dim=0, largest=True, sorted=True)
            else:
                # Find top k scores across all beams
                top_scores, top_words = scores.reshape(-1).topk(k, dim=0, largest=True, sorted=True)
            
            # Convert to vocabulary indices
            prev_word_inds = top_words // vocab_size  # Which beam
            next_word_inds = top_words % vocab_size  # Which word
            
            # Build next step sequences
            seqs = torch.cat([k_prev_words[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Store attention weights
            for idx, beam_idx in enumerate(prev_word_inds):
                seqs_alpha[idx] = seqs_alpha[beam_idx.item()] + [alpha[beam_idx].cpu()]
            
            # Check for complete sequences
            incomplete_inds = []
            for ind, next_word in enumerate(next_word_inds):
                if next_word.item() == self.end_idx:
                    # Apply length penalty
                    seq_len = len(seqs[ind])
                    score = top_scores[ind].item()
                    if length_penalty != 0:
                        score = score / (seq_len ** length_penalty)
                    
                    complete_seqs.append(seqs[ind].tolist())
                    complete_seqs_scores.append(score)
                    complete_seqs_alpha.append(seqs_alpha[ind])
                else:
                    incomplete_inds.append(ind)
            
            # Stop if we have enough complete sequences
            if len(complete_seqs) >= k:
                break
            
            # Prepare for next iteration
            k_prev_words = seqs[incomplete_inds]
            seqs_scores = top_scores[incomplete_inds].unsqueeze(1)
            seqs_alpha = [seqs_alpha[i] for i in incomplete_inds]
            
            # Update hidden states for remaining beams
            h = h[:, prev_word_inds[incomplete_inds], :]
            c = c[:, prev_word_inds[incomplete_inds], :]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            
            step += 1
            
            # Stop if max length reached
            if step > max_length:
                break
        
        # If no complete sequences, use best incomplete
        if len(complete_seqs) == 0:
            complete_seqs = [k_prev_words[0].tolist()]
            complete_seqs_scores = [seqs_scores[0].item()]
            complete_seqs_alpha = [seqs_alpha[0]]
        
        # Select best sequence
        best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        caption = complete_seqs[best_idx]
        attention_weights = complete_seqs_alpha[best_idx]
        
        return caption, attention_weights
    
    def get_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def unfreeze_encoder(self, layer_name: Optional[str] = None):
        """
        Unfreeze encoder layers for fine-tuning.
        
        Args:
            layer_name: Layer to start unfreezing from (None = unfreeze all)
        """
        self.encoder.unfreeze_layers(layer_name)


if __name__ == '__main__':
    # Test complete model
    print("Testing Complete Encoder-Decoder Model:")
    
    # Create model
    model = EncoderDecoderModel(
        encoder_pretrained=False,  # Don't download weights for test
        vocab_size=1000,
        embedding_dim=512,
        decoder_dim=1024
    )
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {model.get_total_parameters():,}")
    print(f"  Trainable parameters: {model.get_trainable_parameters():,}")
    print(f"  Encoder frozen: {model.encoder.frozen_backbone}")
    
    # Test forward pass (training)
    print(f"\nTesting training forward pass:")
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, 1000, (batch_size, 20))
    caption_lengths = torch.tensor([20, 15])
    
    predictions, attention_weights, sorted_captions, sorted_lengths = model(
        images, captions, caption_lengths
    )
    
    print(f"  Input images shape: {images.shape}")
    print(f"  Input captions shape: {captions.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    
    # Test inference (greedy)
    print(f"\nTesting greedy inference:")
    test_image = torch.randn(1, 3, 224, 224)
    caption_greedy, alpha_greedy = model.generate_caption(
        test_image, max_length=20, beam_size=1
    )
    print(f"  Generated caption length: {len(caption_greedy)}")
    print(f"  Caption tokens: {caption_greedy[:10]}...")  # Show first 10
    
    # Test inference (beam search)
    print(f"\nTesting beam search inference:")
    caption_beam, alpha_beam = model.generate_caption(
        test_image, max_length=20, beam_size=3
    )
    print(f"  Generated caption length: {len(caption_beam)}")
    print(f"  Caption tokens: {caption_beam[:10]}...")

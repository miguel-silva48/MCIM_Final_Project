"""
Configuration loader with automatic type conversion.

Handles YAML files that may have numeric values incorrectly parsed as strings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration and convert string numbers to proper types.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary with proper types
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert known numeric fields from strings to proper types
    _sanitize_config(config)
    
    return config


def _sanitize_config(config: Dict[str, Any]) -> None:
    """
    In-place conversion of string numbers to proper numeric types.
    
    Args:
        config: Configuration dictionary to sanitize
    """
    # Training optimizer parameters
    if 'training' in config and 'optimizer' in config['training']:
        opt = config['training']['optimizer']
        if 'learning_rate' in opt:
            opt['learning_rate'] = float(opt['learning_rate'])
        if 'weight_decay' in opt:
            opt['weight_decay'] = float(opt['weight_decay'])
        if 'betas' in opt:
            if isinstance(opt['betas'], list):
                opt['betas'] = tuple(float(b) for b in opt['betas'])
    
    # Training parameters
    if 'training' in config:
        train = config['training']
        if 'num_epochs' in train:
            train['num_epochs'] = int(train['num_epochs'])
        if 'batch_size' in train:
            train['batch_size'] = int(train['batch_size'])
        if 'gradient_clip_norm' in train:
            train['gradient_clip_norm'] = float(train['gradient_clip_norm'])
        if 'label_smoothing' in train:
            train['label_smoothing'] = float(train['label_smoothing'])
    
    # Scheduler parameters
    if 'training' in config and 'scheduler' in config['training']:
        sched = config['training']['scheduler']
        if 'patience' in sched:
            sched['patience'] = int(sched['patience'])
        if 'factor' in sched:
            sched['factor'] = float(sched['factor'])
        if 'min_lr' in sched:
            sched['min_lr'] = float(sched['min_lr'])
    
    # Early stopping parameters
    if 'training' in config and 'early_stopping' in config['training']:
        early = config['training']['early_stopping']
        if 'patience' in early:
            early['patience'] = int(early['patience'])
    
    # Data parameters
    if 'data' in config:
        data = config['data']
        if 'image_size' in data:
            data['image_size'] = int(data['image_size'])
        if 'max_caption_length' in data:
            data['max_caption_length'] = int(data['max_caption_length'])
        if 'num_workers' in data:
            data['num_workers'] = int(data['num_workers'])
        if 'min_word_freq' in data:
            data['min_word_freq'] = int(data['min_word_freq'])
        
        # Normalization parameters
        if 'normalize' in data:
            norm = data['normalize']
            if 'mean' in norm and isinstance(norm['mean'], list):
                norm['mean'] = [float(v) for v in norm['mean']]
            if 'std' in norm and isinstance(norm['std'], list):
                norm['std'] = [float(v) for v in norm['std']]
        
        # Augmentation parameters
        if 'augmentation' in data:
            aug = data['augmentation']
            if 'random_rotation_degrees' in aug:
                aug['random_rotation_degrees'] = int(aug['random_rotation_degrees'])
    
    # Model parameters
    if 'model' in config:
        model = config['model']
        
        # Encoder parameters
        if 'encoder' in model:
            enc = model['encoder']
            if 'output_feature_dim' in enc:
                enc['output_feature_dim'] = int(enc['output_feature_dim'])
        
        # Decoder parameters
        if 'decoder' in model:
            dec = model['decoder']
            if 'embedding_dim' in dec:
                dec['embedding_dim'] = int(dec['embedding_dim'])
            if 'hidden_dim' in dec:
                dec['hidden_dim'] = int(dec['hidden_dim'])
            if 'num_layers' in dec:
                dec['num_layers'] = int(dec['num_layers'])
            if 'dropout' in dec:
                dec['dropout'] = float(dec['dropout'])
            
            # Attention parameters
            if 'attention' in dec:
                att = dec['attention']
                if 'attention_dim' in att:
                    att['attention_dim'] = int(att['attention_dim'])
    
    # Inference parameters
    if 'inference' in config and 'decoding' in config['inference']:
        decode = config['inference']['decoding']
        if 'max_length' in decode:
            decode['max_length'] = int(decode['max_length'])
        if 'beam_size' in decode:
            decode['beam_size'] = int(decode['beam_size'])
        if 'length_penalty' in decode:
            decode['length_penalty'] = float(decode['length_penalty'])


if __name__ == '__main__':
    # Test the config loader
    print("Testing config loader...")
    
    # Example: Create a test config
    test_config = {
        'training': {
            'num_epochs': '30',
            'batch_size': '32',
            'optimizer': {
                'learning_rate': '0.001',
                'weight_decay': '0.0001'
            }
        }
    }
    
    print("\nBefore sanitization:")
    print(f"  num_epochs type: {type(test_config['training']['num_epochs'])}")
    print(f"  learning_rate type: {type(test_config['training']['optimizer']['learning_rate'])}")
    
    _sanitize_config(test_config)
    
    print("\nAfter sanitization:")
    print(f"  num_epochs: {test_config['training']['num_epochs']} (type: {type(test_config['training']['num_epochs'])})")
    print(f"  learning_rate: {test_config['training']['optimizer']['learning_rate']} (type: {type(test_config['training']['optimizer']['learning_rate'])})")
    
    print("\n[OK] Config loader working correctly!")

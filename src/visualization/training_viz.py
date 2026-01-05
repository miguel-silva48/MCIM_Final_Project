"""Visualization functions for training metrics and results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def plot_training_metrics(
    metrics_csv: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """Plot training and validation metrics over epochs.
    
    Args:
        metrics_csv: Path to metrics CSV file
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if 'train_loss' in df.columns:
        ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    ax2 = axes[0, 1]
    if 'train_perplexity' in df.columns:
        ax2.plot(df['epoch'], df['train_perplexity'], 'o-', label='Train Perplexity', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Training Perplexity', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: BLEU Scores
    ax3 = axes[1, 0]
    bleu_cols = [col for col in df.columns if 'val_bleu' in col]
    for col in bleu_cols:
        bleu_num = col.split('_')[-1]  # Extract bleu number (1,2,3,4)
        ax3.plot(df['epoch'], df[col], 'o-', label=f'BLEU-{bleu_num}', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('BLEU Scores', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: METEOR & ROUGE-L
    ax4 = axes[1, 1]
    if 'val_meteor' in df.columns:
        ax4.plot(df['epoch'], df['val_meteor'], 'o-', label='METEOR', linewidth=2)
    if 'val_rouge_l' in df.columns:
        ax4.plot(df['epoch'], df['val_rouge_l'], 's-', label='ROUGE-L', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('METEOR & ROUGE-L Scores', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    
    return fig


def plot_learning_rate(
    metrics_csv: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot learning rate schedule over epochs.
    
    Args:
        metrics_csv: Path to metrics CSV file
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    if 'learning_rate' not in df.columns:
        print("No learning rate data found in metrics")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['epoch'], df['learning_rate'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale for LR
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to: {save_path}")
    
    return fig


def plot_sample_predictions(
    samples: List[Dict[str, str]],
    max_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot sample predictions vs ground truth captions.
    
    Args:
        samples: List of dicts with 'uid', 'reference', 'generated' keys
        max_samples: Maximum number of samples to display
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    samples = samples[:max_samples]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Create table data
    table_data = []
    for i, sample in enumerate(samples, 1):
        table_data.append([
            f"Sample {i}\n(UID: {sample['uid']})",
            sample['reference'],
            sample['generated']
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Sample', 'Ground Truth', 'Generated'],
        cellLoc='left',
        loc='center',
        colWidths=[0.15, 0.425, 0.425]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Sample Predictions vs Ground Truth', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to: {save_path}")
    
    return fig


def plot_metrics_comparison(
    metrics_csv: str,
    baseline_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Compare final metrics with baseline (if provided).
    
    Args:
        metrics_csv: Path to metrics CSV file
        baseline_metrics: Optional dict with baseline metric values
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    # Get final epoch metrics
    final_metrics = df.iloc[-1]
    
    # Extract validation metrics
    metric_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L']
    metric_cols = ['val_bleu_1', 'val_bleu_2', 'val_bleu_3', 'val_bleu_4', 'val_meteor', 'val_rouge_l']
    
    our_values = [final_metrics[col] for col in metric_cols if col in final_metrics]
    available_names = [name for name, col in zip(metric_names, metric_cols) if col in final_metrics]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(available_names))
    width = 0.35
    
    bars1 = ax.bar(x, our_values, width, label='Our Model', color='#4CAF50')
    
    if baseline_metrics:
        baseline_values = [baseline_metrics.get(name.lower().replace('-', '_'), 0) 
                          for name in available_names]
        bars2 = ax.bar(x + width, baseline_values, width, label='Baseline', color='#FFA726')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2 if baseline_metrics else x)
    ax.set_xticklabels(available_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1] + ([bars2] if baseline_metrics else []):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {save_path}")
    
    return fig


def plot_epoch_time(
    metrics_csv: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot epoch training time.
    
    Args:
        metrics_csv: Path to metrics CSV file
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    if 'epoch_time_seconds' not in df.columns:
        print("No epoch time data found in metrics")
        return None
    
    # Convert to minutes
    df['epoch_time_minutes'] = df['epoch_time_seconds'] / 60
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(df['epoch'], df['epoch_time_minutes'], color='#2196F3', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_time = df['epoch_time_minutes'].mean()
    ax.axhline(y=avg_time, color='r', linestyle='--', linewidth=2, 
               label=f'Average: {avg_time:.1f} min')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Epoch time plot saved to: {save_path}")
    
    return fig


if __name__ == '__main__':
    print("Training visualization module")
    print("\nAvailable functions:")
    print("  - plot_training_metrics()")
    print("  - plot_learning_rate()")
    print("  - plot_sample_predictions()")
    print("  - plot_metrics_comparison()")
    print("  - plot_epoch_time()")

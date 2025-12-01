"""
Visualization functions for exploratory data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, Optional
from PIL import Image

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_patient_image_distribution(
    patient_counts_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot distribution of images per patient and frontal/lateral breakdown.
    
    Args:
        patient_counts_df: DataFrame with columns uid, n_frontal, n_lateral, total_images
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total images per patient
    axes[0].hist(patient_counts_df['total_images'], bins=range(1, patient_counts_df['total_images'].max() + 2),
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Images per Patient')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title('Distribution of Images per Patient')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Frontal images per patient
    axes[1].hist(patient_counts_df['n_frontal'], bins=range(0, patient_counts_df['n_frontal'].max() + 2),
                 edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Number of Frontal Images')
    axes[1].set_ylabel('Number of Patients')
    axes[1].set_title('Frontal Images per Patient')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Lateral images per patient
    axes[2].hist(patient_counts_df['n_lateral'], bins=range(0, patient_counts_df['n_lateral'].max() + 2),
                 edgecolor='black', alpha=0.7, color='coral')
    axes[2].set_xlabel('Number of Lateral Images')
    axes[2].set_ylabel('Number of Patients')
    axes[2].set_title('Lateral Images per Patient')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_projection_breakdown(
    stats: Dict[str, int],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot pie chart breakdown of patient projection patterns.
    
    Args:
        stats: Dictionary from get_projection_statistics
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = [
        f"Ideal Pairs (1F+1L)\n{stats['ideal_pairs']}",
        f"Frontal Only\n{stats['frontal_only']}",
        f"Lateral Only\n{stats['lateral_only']}",
        f"Multiple Frontals\n{stats['multiple_frontals']}",
        f"Multiple Laterals\n{stats['multiple_laterals']}"
    ]
    
    sizes = [
        stats['ideal_pairs'],
        stats['frontal_only'],
        stats['lateral_only'],
        stats['multiple_frontals'],
        stats['multiple_laterals']
    ]
    
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    explode = (0.05, 0, 0, 0, 0)  # Emphasize ideal pairs
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title(f'Patient Projection Patterns (Total: {stats["total_patients"]} patients)', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_text_length_distributions(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot text length distributions for findings, impression, and combined text.
    
    Args:
        df: DataFrame with 'findings' and 'impression' columns
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calculate lengths (in words)
    findings_lengths = df['findings'].dropna().apply(lambda x: len(str(x).split()))
    impression_lengths = df['impression'].dropna().apply(lambda x: len(str(x).split()))
    combined_lengths = df.apply(
        lambda row: len(str(row['findings']).split()) + len(str(row['impression']).split())
        if pd.notna(row['findings']) and pd.notna(row['impression'])
        else len(str(row['findings']).split()) if pd.notna(row['findings'])
        else len(str(row['impression']).split()) if pd.notna(row['impression'])
        else 0,
        axis=1
    )
    
    # Findings
    axes[0].hist(findings_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Findings Length Distribution\n(Mean: {findings_lengths.mean():.1f} words)')
    axes[0].axvline(findings_lengths.median(), color='red', linestyle='--', label=f'Median: {findings_lengths.median():.1f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Impression
    axes[1].hist(impression_lengths, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Impression Length Distribution\n(Mean: {impression_lengths.mean():.1f} words)')
    axes[1].axvline(impression_lengths.median(), color='red', linestyle='--', label=f'Median: {impression_lengths.median():.1f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Combined
    axes[2].hist(combined_lengths, bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[2].set_xlabel('Number of Words')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Combined Text Length Distribution\n(Mean: {combined_lengths.mean():.1f} words)')
    axes[2].axvline(combined_lengths.median(), color='red', linestyle='--', label=f'Median: {combined_lengths.median():.1f}')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_censoring_distribution(
    censoring_ratios: pd.Series,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot distribution of censoring ratios (XXXX prevalence).
    
    Args:
        censoring_ratios: Series of censoring ratios (0.0 to 1.0)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(censoring_ratios, bins=50, edgecolor='black', alpha=0.7, color='darkred')
    ax.set_xlabel('Censoring Ratio (proportion of XXXX tokens)')
    ax.set_ylabel('Number of Reports')
    ax.set_title(f'Distribution of Report Censoring\n(Mean: {censoring_ratios.mean():.3f}, Median: {censoring_ratios.median():.3f})')
    ax.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='30% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_ngram_frequencies(
    ngram_counter: Counter,
    n: int,
    top_k: int = 30,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal bar chart of most frequent n-grams.
    
    Args:
        ngram_counter: Counter object with n-gram frequencies
        n: N-gram size (for title)
        top_k: Number of top n-grams to display
        save_path: Optional path to save figure
    """
    # Get top k n-grams
    top_ngrams = ngram_counter.most_common(top_k)
    
    # Prepare data
    ngrams = [' '.join(ng) for ng, _ in top_ngrams]
    counts = [count for _, count in top_ngrams]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.3)))
    
    y_pos = np.arange(len(ngrams))
    ax.barh(y_pos, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngrams, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_k} Most Frequent {n}-grams', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_sample_xrays(
    df: pd.DataFrame,
    data_paths: Dict[str, Path],
    n_samples: int = 6,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize sample X-ray images with their associated reports.
    
    Args:
        df: Merged DataFrame with image filenames and reports
        data_paths: Dictionary with path to images directory
        n_samples: Number of samples to display
        save_path: Optional path to save figure
    """
    # Sample random patients with both frontal and lateral views
    patient_samples = df.groupby('uid').filter(
        lambda x: (x['projection'] == 'Frontal').any() and (x['projection'] == 'Lateral').any()
    ).groupby('uid').first().sample(min(n_samples, 6))
    
    n_rows = min(n_samples, len(patient_samples))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    images_dir = data_paths['images_dir']
    
    for idx, (uid, row) in enumerate(patient_samples.iterrows()):
        if idx >= n_rows:
            break
        
        # Get frontal and lateral images for this patient
        patient_images = df[df['uid'] == uid]
        frontal_img = patient_images[patient_images['projection'] == 'Frontal'].iloc[0] if (patient_images['projection'] == 'Frontal').any() else None
        lateral_img = patient_images[patient_images['projection'] == 'Lateral'].iloc[0] if (patient_images['projection'] == 'Lateral').any() else None
        
        # Plot frontal
        if frontal_img is not None:
            img_path = images_dir / frontal_img['filename']
            if img_path.exists():
                img = Image.open(img_path)
                axes[idx, 0].imshow(img, cmap='gray')
                axes[idx, 0].set_title(f'Patient {uid} - Frontal', fontweight='bold')
                axes[idx, 0].axis('off')
        
        # Plot lateral
        if lateral_img is not None:
            img_path = images_dir / lateral_img['filename']
            if img_path.exists():
                img = Image.open(img_path)
                axes[idx, 1].imshow(img, cmap='gray')
                axes[idx, 1].set_title(f'Patient {uid} - Lateral', fontweight='bold')
                axes[idx, 1].axis('off')
        
        # Add report text below (using first row for this patient)
        impression = row.get('impression', 'N/A')
        if pd.notna(impression):
            impression_text = str(impression)[:150] + '...' if len(str(impression)) > 150 else str(impression)
            fig.text(0.5, 1 - (idx + 0.9) / n_rows, f"Impression: {impression_text}",
                    ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Visualization module loaded. Use functions in Jupyter notebook for EDA.")

#!/usr/bin/env python3
"""
Visualize channels_beam from train_normalized_2.mat or test_normalized_2.mat
Displays magnitude of 5 randomly selected channels_beam (4x16 complex matrices)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import argparse
import os


def load_channels_beam(datadir, is_train=True, num_samples=5):
    """Load random channels_beam samples from .mat file"""
    if is_train:
        mat_path = os.path.join(datadir, 'train_normalized.mat')
        dataset_name = "Train"
    else:
        mat_path = os.path.join(datadir, 'test_normalized.mat')
        dataset_name = "Test"
    
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")
    
    mat_data = scipy.io.loadmat(mat_path)
    channels_beam = mat_data['channels_beam']  # (N, 4, 16) complex64
    
    # Use specific indices
    selected_indices = [11081, 11082, 11083, 11084, 11085]
    
    # Randomly select samples (commented out)
    # n_samples = channels_beam.shape[0]
    # num_samples = min(num_samples, n_samples)
    # selected_indices = np.random.choice(n_samples, num_samples, replace=False)
    
    selected_channels = []
    for idx in selected_indices:
        selected_channels.append(channels_beam[idx])  # (4, 16) complex64
    
    return selected_channels, selected_indices, dataset_name


def visualize_channels_beam(datadir, is_train=True, num_samples=5, save_path=None):
    """Visualize magnitude of channels_beam"""
    # Load data
    selected_channels, selected_indices, dataset_name = load_channels_beam(datadir, is_train, num_samples)
    
    # Create figure with 5x1 subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    
    # Handle single subplot case
    if num_samples == 1:
        axes = [axes]
    
    for i, (channels_beam, idx) in enumerate(zip(selected_channels, selected_indices)):
        # Compute magnitude
        magnitude = np.abs(channels_beam)  # (4, 16)
        
        # Plot magnitude
        ax = axes[i]
        im = ax.imshow(magnitude, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title(f'{dataset_name} Data - User {i+1} (Index {idx})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nt (Transmit Antennas)', fontsize=10)
        ax.set_ylabel('Nr (Receive Antennas)', fontsize=10)
        
        # Set ticks
        ax.set_xticks(range(16))
        ax.set_yticks(range(4))
        ax.set_xticklabels(range(16))
        ax.set_yticklabels(range(4))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Magnitude')
    
    plt.tight_layout()
    
    # Default save path: utils/visualized_channels.png
    if save_path is None:
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(utils_dir, 'visualized_channels.png')
    
    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.close()
    
    # Print channels_beam complex values
    print(f"\n=== {dataset_name} Dataset Channels_Beam (4x16 Complex) ===")
    print(f"Selected {len(selected_indices)} samples")
    print(f"Indices: {selected_indices}")
    for i, (channels_beam, idx) in enumerate(zip(selected_channels, selected_indices)):
        print(f"\nUser {i+1} (Index {idx}):")
        print(channels_beam)


def main():
    parser = argparse.ArgumentParser(description='Visualize channels_beam magnitude from .mat file')
    parser.add_argument('--datadir', type=str, default='./dataset/asu_campus_3p5',
                       help='Path to dataset directory')
    parser.add_argument('--is_train', action='store_true', default=True,
                       help='Use train_normalized_2.mat (default: True)')
    parser.add_argument('--is_test', action='store_true', default=False,
                       help='Use test_normalized_2.mat (overrides --is_train)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random samples to visualize (default: 5)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the visualization (default: utils/visualized_channels.png)')
    
    args = parser.parse_args()
    
    # Determine which dataset to use
    is_train = not args.is_test if args.is_test else args.is_train
    
    visualize_channels_beam(args.datadir, is_train, args.num_samples, args.save)


if __name__ == "__main__":
    main()


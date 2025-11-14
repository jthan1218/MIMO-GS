#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
import scipy.io
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    Note: For MIMO dataset, train and test are already separated in .mat files,
    so this function is not used for MIMO dataset.
    """
    if dataset_type == "mimo":
        # MIMO dataset already has train_normalized.mat and test_normalized.mat separated
        # No need to split
        print("MIMO dataset: train and test are already separated in .mat files")
        return
    elif dataset_type == "rfid":
        # Fallback to old method for legacy datasets
        spectrum_dir = os.path.join(datadir, 'spectrum')
        if os.path.exists(spectrum_dir):
            spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
            index = [x.split('.')[0] for x in spt_names]
            random.shuffle(index)

            train_len = int(len(index) * ratio)
            train_index = np.array(index[:train_len])
            test_index = np.array(index[train_len:])

            np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
            np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class MIMO_dataset(Dataset):
    """MIMO dataset class for loading channels_beam and positions from .mat files."""
    
    def __init__(self, datadir, is_train=True) -> None:
        super().__init__()
        self.datadir = datadir
        self.is_train = is_train
        
        # Load .mat file
        if is_train:
            mat_path = os.path.join(datadir, 'train_normalized.mat')
        else:
            mat_path = os.path.join(datadir, 'test_normalized.mat')
        
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Dataset file not found: {mat_path}")
        
        mat_data = scipy.io.loadmat(mat_path)
        self.positions = mat_data['positions']  # (N, 3)
        self.channels_beam = mat_data['channels_beam']  # (N, 4, 16) complex64
        
        # Use all samples from the .mat file (train and test are already separated)
        self.n_samples = self.positions.shape[0]
        
        print(f"Loaded {self.n_samples} samples from {mat_path}")
        print(f"  Positions shape: {self.positions.shape}")
        print(f"  Channels_beam shape: {self.channels_beam.shape}")

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        # Get Rx position (3D coordinates)
        rx_pos = torch.tensor(self.positions[index], dtype=torch.float32)
        
        # Get channels_beam (4, 16) complex array
        channels_beam = self.channels_beam[index]  # (4, 16) complex64
        
        # Convert to torch tensor (split into real and imaginary parts)
        # channels_beam_real: (4, 16), channels_beam_imag: (4, 16)
        channels_beam_real = torch.tensor(np.real(channels_beam), dtype=torch.float32)
        channels_beam_imag = torch.tensor(np.imag(channels_beam), dtype=torch.float32)
        
        # Stack real and imag to create (2, 4, 16) tensor
        channels_beam_tensor = torch.stack([channels_beam_real, channels_beam_imag], dim=0)
        
        return channels_beam_tensor, rx_pos  


dataset_dict = {"mimo": MIMO_dataset}

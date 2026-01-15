#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io as sio


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class Spectrum_dataset(Dataset):
    """Spectrum dataset class."""
    
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrum')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        self.tx_pos = pd.read_csv(self.tx_pos_dir).values  
        self.n_samples = len(self.dataset_index)  

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        tx_pos_i = torch.tensor(self.tx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)

        return spectrum, tx_pos_i  


# class MatMagnitudeDataset(Dataset):
#     """MIMO magnitude dataset from .mat files."""

#     def __init__(self, mat_path: str) -> None:
#         super().__init__()
#         mat_data = sio.loadmat(mat_path)
#         self.positions = torch.tensor(mat_data["positions"], dtype=torch.float32)
#         self.magnitude = torch.tensor(mat_data["magnitude"], dtype=torch.float32)
#         self.n_samples = self.positions.shape[0]

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, index):
#         return self.magnitude[index], self.positions[index]

class MatMagnitudeDataset(Dataset):
    """MIMO magnitude dataset from .mat files with Auto-Normalization."""

    def __init__(self, mat_path: str, normalize: bool = True) -> None:
        super().__init__()
        mat_data = sio.loadmat(mat_path)
        
        # 1. 원본 데이터 로드
        # (N, 3) 또는 (N, 2) 형태의 실제 좌표 (예: -10m ~ 10m)
        self.positions = torch.tensor(mat_data["positions"], dtype=torch.float32)
        self.magnitude = torch.tensor(mat_data["magnitude"], dtype=torch.float32)
        self.n_samples = self.positions.shape[0]

        # 2. [핵심 수정] 스케일 정규화 (Normalization)
        self.normalize = normalize
        self.scale_factor = 1.0  # 나중에 복원할 때 필요할 수 있으니 저장

        if self.normalize:
            # 좌표 중 절대값이 가장 큰 값을 찾습니다. (예: 12.5)
            # 이 값으로 나누면 모든 좌표가 -1.0 ~ 1.0 안으로 들어옵니다.
            max_val = self.positions.abs().max()
            
            # 안전하게 나누기 위해 0.0001 더해줌 (0으로 나누기 방지)
            self.scale_factor = float(max_val) + 1e-6

            print(f"[Dataset] Auto-normalizing positions...")
            print(f"   - Max coordinate found: {max_val:.4f}")
            print(f"   - Scale factor applied: {self.scale_factor:.4f}")
            
            # 모든 좌표를 -1 ~ 1 범위로 압축
            self.positions = self.positions / self.scale_factor

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # 이미 __init__에서 정규화된 positions를 반환합니다.
        return self.magnitude[index], self.positions[index]


dataset_dict = {"rfid": Spectrum_dataset, "mimo": MatMagnitudeDataset}

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np
import json

import torch.nn.functional as F

import logging

import tqdm
log = logging.getLogger(__name__)

from numba import njit, prange

@njit(fastmath=True)
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            # 대각선, 위, 왼쪽 중 최소값 선택
            last_min = min(dtw_matrix[i - 1, j],  # insertion
                           dtw_matrix[i, j - 1],  # deletion
                           dtw_matrix[i - 1, j - 1])  # match
            dtw_matrix[i, j] = cost + last_min

    return np.sqrt(dtw_matrix[n, m])

def make_graph(grid, grid_size):
    T, X, Y = grid.shape
    num_nodes = X * Y
    grid = grid.reshape(T, num_nodes)
    edges = []
    for i in tqdm.tqdm(range(num_nodes), desc="Constructing graph edges"):
        for j in range(i + 1, num_nodes):
            dist = dtw_distance(grid[:, i], grid[:, j])
            edges.append({'u': i, 'v': j, 'w': dist})
            edges.append({'u': j, 'v': i, 'w': dist})
    root = Path('./data/processed')
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(edges)
    df.to_csv(root / f'dmvst_graph_edges_{grid_size}.csv', index=False)
            

# T, (X, Y) 4000, 13, 13
class DMVSTDataset(Dataset):
    def __init__(self, time_step, patch_size=7, grid_size=9500, target_columns=['강수량(mm)', '기온(°C)', '습도(%)', '적설(cm)']):
        root_path = Path('./data/raw')
        demands = np.load(root_path / f'grid({grid_size}).npy')  # (T, X, Y)

        if not (Path(f"data/processed/dmvst_graph_edges_{grid_size}.csv")).exists():
            make_graph(demands, grid_size)

        demands = torch.from_numpy(demands) # (T, X, Y)
        self.T, self.X, self.Y = demands.shape
        padding = (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2)
        self.patch_size = patch_size
        self.demands = F.pad(demands, padding, mode='constant', value=0)
        self.demands = self.demands.to(torch.float32)
        # print(f"Loaded demands with shape: {self.demands.shape}")
        self.time_step = time_step

        df = pd.read_csv(root_path / 'meteorological_data.csv', encoding='cp949')
        df_filled = df.fillna(0)
        self.temporal_features = torch.tensor(df_filled[target_columns].values, dtype=torch.float32)  # (T, num_features)
        log.info(f'total data length: {self.__len__()}')
        
        expected_nodes = self.X * self.Y
        log.info(f"Dataset Info: X={self.X}, Y={self.Y}, Total Nodes(Max ID)={expected_nodes}")
        
        #유효 격자 체크
        self.valid_mask = torch.zeros(self.X, self.Y, dtype=torch.bool)
        
        with open(root_path / 'graph_data.json', 'r') as f:
            graph_data = json.load(f)
        meta_data = graph_data['nodes']
        for node in meta_data:
            #print(f'keys: {node.keys()}')
            cells = node.get('cells', [])
            for cell in cells:
                x, y = cell
                if 0 <= x < self.X and 0 <= y < self.Y:
                    self.valid_mask[x, y] = True  # 유효한 격자 위치는 True로 표시
        # print(f'valid grid: {self.valid_mask.to(torch.int)}')
        # print(f'sample shape: {self.__getitem__(0)['demands'].shape}')
        # print(f'sample: {self.__getitem__(0)['demands']}')
        # print(f'sample: {self.__getitem__(0)['labels']}')
            
            
    def __len__(self):
        return (self.T - self.time_step) * self.X * self.Y

    def __getitem__(self, idx:int):
        t_idx = idx // (self.X * self.Y)
        xy_idx = idx % (self.X * self.Y)
        x_idx = xy_idx // self.Y
        y_idx = xy_idx % self.Y

        demand_seq = self.demands[t_idx:t_idx + self.time_step, x_idx:x_idx + self.patch_size, y_idx:y_idx + self.patch_size]
        label = self.demands[t_idx + self.time_step, x_idx + self.patch_size // 2, y_idx + self.patch_size // 2]

        return {
            'demands': demand_seq,  # (time_step, 7, 7)
            'labels': label,  # scalar,
            'temporal_features': self.temporal_features[t_idx:t_idx + self.time_step],  # (time_step, num_features)
            'valid': self.valid_mask[x_idx, y_idx],  # 해당 격자가 유효한지 여부
            'node_id': x_idx * self.Y + y_idx  # unique node id
        }
        
def collate_fn(batch):
    demands = torch.stack([item['demands'] for item in batch], dim=0)  # (B, 1, time_step, 7, 7)
    labels = torch.stack([item['labels'] for item in batch], dim=0)  # (B,)
    temporal_features = torch.stack([item['temporal_features'] for item in batch], dim=0)  # (B, time_step, num_features)
    valid_mask = torch.tensor([item['valid'] for item in batch], dtype=torch.bool)  # (B,)
    node_ids = torch.tensor([item['node_id'] for item in batch], dtype=torch.long)  # (B,)

    return {
        'demands': demands,
        'labels': labels,
        'temporal_features': temporal_features,
        'valid_mask': valid_mask,
        'node_ids': node_ids
    }
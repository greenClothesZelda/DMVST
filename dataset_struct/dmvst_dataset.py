import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np

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

def make_graph(grid, output_path):
    T, X, Y = grid.shape
    num_nodes = X * Y
    grid = grid.reshape(T, num_nodes)
    edges = []
    for i in tqdm.tqdm(range(num_nodes), desc="Constructing graph edges"):
        for j in range(i + 1, num_nodes):
            dist = dtw_distance(grid[:, i], grid[:, j])
            edges.append({'u': i, 'v': j, 'w': dist})
            edges.append({'u': j, 'v': i, 'w': dist})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(edges)
    df.to_csv(output_path, index=False)
            

# T, (X, Y) 4000, 13, 13
class DMVSTDataset(Dataset):
    def __init__(self, time_step, patch_size=7, grid_size=9500, target_columns=['강수량(mm)', '기온(°C)', '습도(%)', '적설(cm)']):
        root_path = Path('./data/raw')
        demands = np.load(root_path / f'grid({grid_size}).npy')  # (T, X, Y)
        demands = torch.from_numpy(demands).to(torch.float32) # (T, X, Y)
        self.raw_demands = demands
        self.T, self.X, self.Y = demands.shape
        self.grid_size = grid_size
        self.num_nodes = self.X * self.Y
        padding = (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2)
        self.patch_size = patch_size
        self.demands = F.pad(self.raw_demands, padding, mode='constant', value=0)
        self.time_step = time_step

        df = pd.read_csv(root_path / 'meteorological_data.csv', encoding='cp949')
        df_filled = df.fillna(0)
        self.temporal_features = torch.tensor(df_filled[target_columns].values, dtype=torch.float32)  # (T, num_features)
        log.info(f'total data length: {self.__len__()}')
        
        expected_nodes = self.num_nodes
        log.info(f"Dataset Info: X={self.X}, Y={self.Y}, Total Nodes(Max ID)={expected_nodes}")
        


    def __len__(self):
        return (self.T - self.time_step) * self.num_nodes

    def get_train_graph_path(self, num_samples: int):
        if num_samples <= 0 or num_samples > len(self):
            raise ValueError(f"num_samples must be in [1, {len(self)}], got {num_samples}")

        train_label_steps = (num_samples + self.num_nodes - 1) // self.num_nodes
        raw_steps = min(self.T, train_label_steps + self.time_step)
        graph_path = Path('data/processed') / f'dmvst_graph_edges_{self.grid_size}_train_{raw_steps}.csv'

        if not graph_path.exists():
            make_graph(self.raw_demands[:raw_steps].numpy(), graph_path)

        return graph_path

    def __getitem__(self, idx:int):
        t_idx = idx // self.num_nodes
        xy_idx = idx % self.num_nodes
        x_idx = xy_idx // self.Y
        y_idx = xy_idx % self.Y

        demand_seq = self.demands[t_idx:t_idx + self.time_step, x_idx:x_idx + self.patch_size, y_idx:y_idx + self.patch_size]
        label = self.raw_demands[t_idx + self.time_step, x_idx, y_idx]

        return {
            'demands': demand_seq,  # (time_step, 7, 7)
            'labels': label,  # scalar,
            'temporal_features': self.temporal_features[t_idx:t_idx + self.time_step],  # (time_step, num_features)
            'node_id': x_idx * self.Y + y_idx  # unique node id
        }
        
def collate_fn(batch):
    demands = torch.stack([item['demands'] for item in batch], dim=0)  # (B, 1, time_step, 7, 7)
    labels = torch.stack([item['labels'] for item in batch], dim=0)  # (B,)
    temporal_features = torch.stack([item['temporal_features'] for item in batch], dim=0)  # (B, time_step, num_features)
    node_ids = torch.tensor([item['node_id'] for item in batch], dtype=torch.long)  # (B,)

    return {
        'demands': demands,
        'labels': labels,
        'temporal_features': temporal_features,
        'node_ids': node_ids
    }

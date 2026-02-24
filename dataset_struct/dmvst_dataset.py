from numba import njit, prange
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


@njit(fastmath=True, parallel=True)
def calculate_all_distances(valid_grid_data, num_nodes):
    # 결과를 저장할 행렬 (u, v, weight)
    # 데이터 양이 많으므로 리스트 대신 넘파이 배열 활용이 효율적입니다.
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # prange를 사용하여 병렬 루프 실행
    for i in prange(num_nodes):
        for j in range(i + 1, num_nodes):
            s1 = valid_grid_data[i]
            s2 = valid_grid_data[j]
            
            # 내부 DTW 로직 (인라인화 되어 실행됨)
            n, m = len(s1), len(s2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            for ii in range(1, n + 1):
                for jj in range(1, m + 1):
                    cost = (s1[ii - 1] - s2[jj - 1]) ** 2
                    last_min = min(dtw_matrix[ii - 1, jj], 
                                   dtw_matrix[ii, jj - 1], 
                                   dtw_matrix[ii - 1, jj - 1])
                    dtw_matrix[ii, jj] = cost + last_min
            
            dist = np.sqrt(dtw_matrix[n, m])
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist
            
    return adj_matrix

def make_graph(grid, grid_size, valid_grids):
    num_nodes = len(valid_grids)
    
    # [핵심] 루프 밖에서 미리 모든 유효 데이터를 넘파이 배열로 추출
    # shape: (num_nodes, T)
    print("Preparing data...")
    grid_np = grid.detach().cpu().numpy()
    valid_grid_data = np.zeros((num_nodes, grid_np.shape[0]))
    for i in range(num_nodes):
        valid_grid_data[i] = grid_np[:, valid_grids[i][0], valid_grids[i][1]]

    print(f"Constructing graph edges (Parallel)...")
    # Numba 병렬 함수 호출
    adj_matrix = calculate_all_distances(valid_grid_data, num_nodes)
    
    # 결과 저장 로직
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = adj_matrix[i, j]
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

        demands = torch.from_numpy(demands)  # (T, X, Y)
        self.T, self.X, self.Y = demands.shape
        padding = (patch_size // 2, patch_size // 2,
                   patch_size // 2, patch_size // 2)
        self.patch_size = patch_size
        self.demands = F.pad(demands, padding, mode='constant', value=0)
        self.demands = self.demands.to(torch.float32)
        # print(f"Loaded demands with shape: {self.demands.shape}")
        self.time_step = time_step

        df = pd.read_csv(
            root_path / 'meteorological_data.csv', encoding='cp949')
        df_filled = df.fillna(0)
        self.temporal_features = torch.tensor(
            # (T, num_features)
            df_filled[target_columns].values, dtype=torch.float32)
        # log.info(f'total data length: {self.__len__()}')

        expected_nodes = self.X * self.Y
        log.info(
            f"Dataset Info: X={self.X}, Y={self.Y}, Total Nodes(Max ID)={expected_nodes}")

        # 유효 격자 체크
        self.valid_mask = torch.zeros(self.X, self.Y, dtype=torch.bool)
        self.valid_grids = []

        with open(root_path / 'graph_data.json', 'r') as f:
            graph_data = json.load(f)
        meta_data = graph_data['nodes']
        for node in meta_data:
            # print(f'keys: {node.keys()}')
            cells = node.get('cells', [])
            for cell in cells:
                x, y = cell
                if 0 <= x < self.X and 0 <= y < self.Y:
                    self.valid_mask[x, y] = True  
                    self.valid_grids.append((x, y))

        self.valid_indices = []
        for t_idx in range(self.T - self.time_step):
            node_id = 0 
            for x_idx in range(self.X):
                for y_idx in range(self.Y):
                    if self.valid_mask[x_idx, y_idx]:
                        node_id += 1
                        self.valid_indices.append((t_idx, x_idx, y_idx, node_id))
                        
        if not (Path(f"data/processed/dmvst_graph_edges_{grid_size}.csv")).exists():
            make_graph(demands, grid_size, self.valid_grids)
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        t_idx, x_idx, y_idx, node_id = self.valid_indices[idx]
        demand_seq = self.demands[t_idx:t_idx + self.time_step,
                                  x_idx:x_idx + self.patch_size, y_idx:y_idx + self.patch_size]
        label = self.demands[t_idx + self.time_step, x_idx +
                             self.patch_size // 2, y_idx + self.patch_size // 2]

        return {
            'demands': demand_seq,  # (time_step, 7, 7)
            'labels': label,  # scalar,
            # (time_step, num_features)
            'temporal_features': self.temporal_features[t_idx:t_idx + self.time_step],
            'valid': self.valid_mask[x_idx, y_idx],  # 해당 격자가 유효한지 여부
            'node_id': node_id  # unique node id
        }


def collate_fn(batch):
    demands = torch.stack([item['demands']
                          for item in batch], dim=0)  # (B, 1, time_step, 7, 7)
    labels = torch.stack([item['labels'] for item in batch], dim=0)  # (B,)
    # (B, time_step, num_features)
    temporal_features = torch.stack(
        [item['temporal_features'] for item in batch], dim=0)
    valid_mask = torch.tensor([item['valid']
                              for item in batch], dtype=torch.bool)  # (B,)
    node_ids = torch.tensor([item['node_id']
                            for item in batch], dtype=torch.long)  # (B,)

    return {
        'demands': demands,
        'labels': labels,
        'temporal_features': temporal_features,
        'valid_mask': valid_mask,
        'node_ids': node_ids
    }

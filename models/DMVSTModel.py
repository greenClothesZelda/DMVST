import numpy as np
from regex import F
import torch
import torch.nn as nn
from loss_fn import DMVSTLoss

from cogdl.data import Graph
from cogdl.models.emb.line import LINE
import pandas as pd
from pathlib import Path
import logging
log = logging.getLogger(__name__)


if not hasattr(np, 'int'):
    np.int = int


def line(grid_size, dimension=64, walk_length=40, walk_num=10, negative=5, batch_size=100, alpha=0.025, order=2):
    root = Path('./data/processed')
    log.info(
        f"Loading LINE embeddings from graph CSV for grid size: {grid_size}")
    df = pd.read_csv(root / f'dmvst_graph_edges_{grid_size}.csv')
    edges = df[['u', 'v', 'w']].values
    nodes = set()
    for u, v, w in edges:
        nodes.add(u)
        nodes.add(v)

    node_to_id = {node: i for i, node in enumerate(sorted(nodes))}
    num_nodes = len(nodes)

    src_list = []
    dst_list = []
    edge_weights = []

    for src, dst, w in edges:
        src_list.append(node_to_id[src])
        dst_list.append(node_to_id[dst])
        edge_weights.append(w)
    edge_index = torch.LongTensor([src_list, dst_list])
    edge_weight = torch.FloatTensor(edge_weights)
    data = Graph(edge_index=edge_index,
                 edge_weight=edge_weight, num_nodes=num_nodes)

    model = LINE(
        dimension=dimension,
        walk_length=walk_length,
        walk_num=walk_num,
        negative=negative,
        batch_size=batch_size,
        alpha=alpha,
        order=order
    )
    embeddings = model(data)
    log.info(f"Generated LINE embeddings with shape: {embeddings.shape}")
    return embeddings


class LocalCNN(nn.Module):
    def __init__(self, num_filters, num_cnn_layers, kernel_size, neighborhood_size, embedding_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        channels = 1
        padding = kernel_size // 2
        for _ in range(num_cnn_layers):
            out_channels = num_filters * channels
            self.convs.append(nn.Conv2d(channels, out_channels,
                              kernel_size=kernel_size, padding=padding))
            # self.convs.append(nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels))
            self.convs.append(nn.ReLU())
            channels = out_channels
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(
            neighborhood_size * neighborhood_size * channels, embedding_dim)

    def forward(self, x):
        B, T, X, Y = x.size()
        x = x.view(B * T, 1, X, Y)
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.embedding_layer(x)
        x = x.view(B, T, -1)
        return x


class DMVST(nn.Module):
    def __init__(self, demand_embedding_dim, temporal_embedding_dim, context_embedding_dim, num_temporal_features, Local_cnn, LSTM, Line, grid_size, loss_fn=None):
        super().__init__()

        self.demand_embedding_dim = demand_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.context_embedding_dim = context_embedding_dim

        self.temporal_layer = nn.Linear(
            num_temporal_features, temporal_embedding_dim)
        self.local_cnn = LocalCNN(
            **Local_cnn, embedding_dim=demand_embedding_dim)
        line_embeddings = line(grid_size, **Line)
        if isinstance(line_embeddings, np.ndarray):
            line_embeddings = torch.from_numpy(line_embeddings).float()
        else:
            # 만약 이미 텐서라면 float형으로만 보장
            line_embeddings = line_embeddings.float()
        self.register_buffer('line_embeddings', line_embeddings)
        self.context_embedding_layer = nn.Linear(
            Line['dimension'], context_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=demand_embedding_dim + temporal_embedding_dim,
            batch_first=True,
            bidirectional=False,
            proj_size=0,
            **LSTM
        )

        self.final_fc = nn.Linear(
            LSTM['hidden_size'] + context_embedding_dim, 1)

        self.loss_fn = loss_fn if loss_fn is not None else DMVSTLoss()

    def predict(self, demands, temporal_features=None, node_ids=None):
        # Dummy implementation for prediction
        # print(demands.shape)
        B, T, X, Y = demands.size()
        demands_features = self.local_cnn(demands)  # (B, T, D1)

        if temporal_features is not None:
            temporal_emb = self.temporal_layer(temporal_features)  # (B, T, D2)
        else:
            temporal_emb = torch.zeros(
                B, T, self.temporal_embedding_dim).to(demands.device)

        if node_ids is not None:
            # print(f'type(node_ids): {type(node_ids)}, shape: {node_ids.shape}')
            context_emb = self.context_embedding_layer(
                self.line_embeddings[node_ids].to(demands.device))  # (B, D3)
        else:
            context_emb = torch.zeros(
                B, self.context_embedding_dim).to(demands.device)

        lstm_input = torch.cat(
            [demands_features, temporal_emb], dim=-1)  # (B, T, D1 + D2)
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, hidden_size)

        final_features = torch.cat(
            [lstm_out[:, -1, :], context_emb], dim=-1)  # (B, hidden_size + D3)
        output = self.final_fc(final_features)  # (B, 1)
        return output.squeeze(-1)

    def forward(self, demands, valid_mask, temporal_features=None, node_ids=None, labels=None):
        predictions = self.predict(demands, temporal_features, node_ids)
        outputs = {'predictions': predictions}
        if labels is not None:
            labels = labels * valid_mask.float()
            predictions = predictions * valid_mask.float()
            loss = self.loss_fn(predictions, labels)
            outputs['loss'] = loss.unsqueeze(0)
        return outputs

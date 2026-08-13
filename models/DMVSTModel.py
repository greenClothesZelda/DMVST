import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cogdl.data import Graph
from cogdl.models.emb.line import LINE

from loss_fn import DMVSTLoss

log = logging.getLogger(__name__)


if not hasattr(np, 'int'):
    np.int = int


class IRModule(nn.Module):
    def __init__(self, dataset, device, k, storage_dtype=torch.float16):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be greater than 0")

        source_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

        self.device = device
        self.k = k
        self.num_nodes = source_dataset.num_nodes
        self.num_time_samples = source_dataset.T - source_dataset.time_step
        self.time_step = source_dataset.time_step
        self.patch_size = source_dataset.patch_size
        self.grid_width = source_dataset.Y
        self.storage_dtype = storage_dtype

        flat_dim = self.time_step * self.patch_size * self.patch_size
        self.db_keys = torch.empty(
            self.num_nodes,
            self.num_time_samples,
            flat_dim,
            dtype=storage_dtype
        )
        self.db_norms = torch.empty(
            self.num_nodes,
            self.num_time_samples,
            dtype=torch.float32
        )
        self.db_values = source_dataset.raw_demands[source_dataset.time_step:].reshape(
            self.num_time_samples,
            self.num_nodes
        ).transpose(0, 1).contiguous().to(torch.float32)

        for node_id in range(self.num_nodes):
            x_idx = node_id // source_dataset.Y
            y_idx = node_id % source_dataset.Y
            patch_series = source_dataset.demands[
                :,
                x_idx:x_idx + self.patch_size,
                y_idx:y_idx + self.patch_size
            ]
            patch_windows = patch_series.unfold(0, self.time_step, 1)[:self.num_time_samples]
            patch_windows = patch_windows.permute(0, 3, 1, 2).contiguous().reshape(self.num_time_samples, -1)
            patch_windows = patch_windows.to(torch.float32)
            self.db_keys[node_id] = patch_windows.to(storage_dtype)
            self.db_norms[node_id] = torch.norm(patch_windows, dim=1).clamp_min(1e-8)

        log.info(
            "IRModule initialized: Nodes=%s, TimeSamples=%s, PatchDim=%s, TopK=%s",
            self.num_nodes,
            self.num_time_samples,
            flat_dim,
            self.k
        )

    def forward(self, query_demands, node_ids, sample_idx):
        queries = query_demands.to(torch.float32).flatten(start_dim=1)
        query_device = queries.device
        node_ids = node_ids.to(device=query_device, dtype=torch.long)
        time_indices = (sample_idx.to(dtype=torch.long) // self.num_nodes).to(query_device)

        batch_size = queries.size(0)
        aggregated = torch.zeros(batch_size, device=query_device, dtype=torch.float32)
        retrieved_indices = torch.full((batch_size, self.k), -1, device=query_device, dtype=torch.long)

        for node_id in torch.unique(node_ids):
            batch_positions = torch.nonzero(node_ids == node_id, as_tuple=False).squeeze(-1)
            node_queries = queries[batch_positions]
            node_time_indices = time_indices[batch_positions]
            max_candidate = int(node_time_indices.max().item())
            if max_candidate <= 0:
                continue

            prefix_keys = self.db_keys[node_id.item(), :max_candidate].to(device=query_device, dtype=torch.float32)
            prefix_values = self.db_values[node_id.item(), :max_candidate].to(device=query_device, dtype=torch.float32)
            prefix_norms = self.db_norms[node_id.item(), :max_candidate].to(device=query_device, dtype=torch.float32)

            query_norms = torch.norm(node_queries, dim=1, keepdim=True).clamp_min(1e-8)
            cosine_similarities = torch.matmul(node_queries, prefix_keys.transpose(0, 1))
            cosine_similarities = cosine_similarities / (query_norms * prefix_norms.unsqueeze(0))

            valid_candidates = torch.arange(max_candidate, device=query_device).unsqueeze(0) < node_time_indices.unsqueeze(1)
            cosine_similarities = cosine_similarities.masked_fill(~valid_candidates, float('-inf'))

            top_k = min(self.k, max_candidate)
            values, indices = torch.topk(cosine_similarities, k=top_k, dim=1)
            valid_topk = torch.isfinite(values)

            masked_values = values.masked_fill(~valid_topk, -1e9)
            weights = torch.softmax(masked_values, dim=1)
            weights = weights * valid_topk
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

            retrieved_values = prefix_values[indices]
            aggregated[batch_positions] = torch.sum(retrieved_values * weights, dim=1)
            retrieved_indices[batch_positions, :top_k] = indices.masked_fill(~valid_topk, -1)

        return aggregated, retrieved_indices


def line(graph_path, dimension=64, walk_length=40, walk_num=10, negative=5, batch_size=100, alpha=0.025, order=2):
    graph_path = Path(graph_path)
    log.info(f"Loading LINE embeddings from graph CSV: {graph_path}")
    df = pd.read_csv(graph_path)
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
    data = Graph(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

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
        self.neighborhood_size = neighborhood_size
        self.mid_point = neighborhood_size // 2
        print(f'neighborhood_size: {neighborhood_size}, mid_point: {self.mid_point}')
        channels = 1
        padding = kernel_size // 2
        for _ in range(num_cnn_layers):
            out_channels = num_filters * channels
            self.convs.append(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding))
            self.convs.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.ReLU())
            channels = out_channels
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(neighborhood_size * neighborhood_size * channels, embedding_dim)

    def forward(self, x):
        batch_size, time_step, width, height = x.size()
        x = x.view(batch_size * time_step, 1, width, height)
        for conv in self.convs:
            x = conv(x)
        #print(f"LocalCNN output shape before flattening: {x.shape}")
        #x = x[:, :, self.mid_point:self.mid_point+1, self.mid_point:self.mid_point+1]
        #print(f"LocalCNN output shape after neighborhood pooling: {x.shape}")
        x = self.flatten(x)
        #print(f"LocalCNN output shape before embedding: {x.shape}")
        x = self.embedding_layer(x)
        x = x.view(batch_size, time_step, -1)
        return x


class DMVST(nn.Module):
    def __init__(
        self,
        demand_embedding_dim,
        temporal_embedding_dim,
        context_embedding_dim,
        num_temporal_features,
        Local_cnn,
        LSTM,
        Line,
        grid_size,
        loss_fn=None,
        line_graph_path=None,
        ir_module=None
    ):
        super().__init__()

        self.demand_embedding_dim = demand_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.ir_module = ir_module

        self.temporal_layer = nn.Linear(num_temporal_features, temporal_embedding_dim)
        self.local_cnn = LocalCNN(**Local_cnn, embedding_dim=demand_embedding_dim)

        graph_path = line_graph_path or Path('./data/processed') / f'dmvst_graph_edges_{grid_size}.csv'
        line_embeddings = line(graph_path, **Line)
        if isinstance(line_embeddings, np.ndarray):
            line_embeddings = torch.from_numpy(line_embeddings).float()
        else:
            line_embeddings = line_embeddings.float()
        self.register_buffer('line_embeddings', line_embeddings)
        self.context_embedding_layer = nn.Linear(Line['dimension'], context_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=demand_embedding_dim + temporal_embedding_dim,
            batch_first=True,
            bidirectional=False,
            proj_size=0,
            **LSTM
        )

        self.fusion_feature_dim = LSTM['hidden_size'] + context_embedding_dim
        self.final_fc = nn.Linear(self.fusion_feature_dim, 1)
        self.lambda_layer = nn.Linear(self.fusion_feature_dim + 1, 1) if ir_module is not None else None

        self.loss_fn = loss_fn if loss_fn is not None else DMVSTLoss()

    def predict(self, demands, temporal_features=None, node_ids=None, sample_idx=None):
        batch_size, time_step, _, _ = demands.size()
        demands_features = self.local_cnn(demands)

        if temporal_features is not None:
            temporal_emb = self.temporal_layer(temporal_features)
        else:
            temporal_emb = torch.zeros(batch_size, time_step, self.temporal_embedding_dim, device=demands.device)

        if node_ids is not None:
            #context_emb = self.context_embedding_layer(self.line_embeddings[node_ids].to(demands.device))
            context_emb = torch.zeros(batch_size, self.context_embedding_dim, device=demands.device)
        else:
            context_emb = torch.zeros(batch_size, self.context_embedding_dim, device=demands.device)

        lstm_input = torch.cat([demands_features, temporal_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)

        final_features = torch.cat([lstm_out[:, -1, :], context_emb], dim=-1)
        output = self.final_fc(final_features).squeeze(-1)

        if self.ir_module is not None:
            if node_ids is None or sample_idx is None:
                raise ValueError("node_ids and sample_idx are required when IRModule is enabled")
            ir_out, _ = self.ir_module(demands, node_ids, sample_idx)
            lambda_input = torch.cat([final_features, ir_out.unsqueeze(-1)], dim=-1)
            lambda_weight = torch.sigmoid(self.lambda_layer(lambda_input)).squeeze(-1)
            output = lambda_weight * output + (1.0 - lambda_weight) * ir_out

        return output

    def forward(self, demands, temporal_features=None, node_ids=None, sample_idx=None, labels=None):
        predictions = self.predict(demands, temporal_features, node_ids, sample_idx)
        outputs = {'predictions': predictions}
        if labels is not None:
            loss = self.loss_fn(predictions, labels)
            outputs['loss'] = loss.unsqueeze(0)
        return outputs

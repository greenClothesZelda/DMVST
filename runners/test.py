
from cProfile import label
from dataset_struct.dmvst_dataset import collate_fn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def test_loop(model, test_dataset, output_dir, device, k):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn)
    all_predictions = []
    all_labels = []

    for batch in test_loader:
        demands = batch['demands'].to(device)
        temporal_features = batch['temporal_features'].to(device)
        node_ids = batch['node_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model.predict(demands, temporal_features, node_ids)
        outputs = torch.clamp(outputs, min=0.0)
        all_predictions.append(outputs.cpu())
        all_labels.append(labels.cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    result = {
        'labels': all_labels.numpy().tolist(),
        'predictions': all_predictions.numpy().tolist()
    }

    mae = F.l1_loss(all_predictions, all_labels).item()
    mape = torch.mean(torch.abs((all_labels - all_predictions) / (all_labels + 1))).item() * 100

    result_df = pd.DataFrame(result)
    result_df.to_csv(f"{output_dir}/test_results.csv", index=False)
    visualize_predictions(f"{output_dir}/test_results.csv", num_nodes=test_dataset.dataset.X * test_dataset.dataset.Y, output_dir=output_dir)

    topk_mae, topk_mape = topk_node_loss(all_predictions, all_labels, total_nodes=test_dataset.dataset.X * test_dataset.dataset.Y, k=k)

    return {
        'MAE': mae,
        'MAPE': mape,
        f'Top{k}_MAE': topk_mae,
        f'Top{k}_MAPE': topk_mape
    }

def topk_node_loss(pred, label, total_nodes, k=20):
    pred = pred.reshape(-1, total_nodes)
    label = label.reshape(-1, total_nodes)
    label_sum = torch.sum(label, dim=0)  #[num_nodes]
    topk_values, topk_indices = torch.topk(label_sum, k=k)
    topk_pred = pred[:, topk_indices]  #(num_samples, k)
    topk_label = label[:, topk_indices]  #(num_samples, k)
    mae = F.l1_loss(topk_pred, topk_label).item()
    mape = torch.mean(torch.abs((topk_label - topk_pred) / (topk_label + 1))).item() * 100
    return mae, mape
    
def visualize_predictions(csv_path, num_nodes, output_dir):
    df = pd.read_csv(csv_path)
    pred = np.array(df['predictions'].values)
    labels = np.array(df['labels'].values)

    pred = pred.reshape(-1, num_nodes)
    labels = labels.reshape(-1, num_nodes)
    
    diff = np.abs(labels - pred)

    demand_sum = np.sum(labels, axis=1) / num_nodes
    #demand_max = np.max(labels, axis=1) / 10
    #std = np.std(pred, axis=1)
    mean = np.mean(diff, axis=1)

    plt.figure(figsize=(24, 16))
    plt.plot(demand_sum, label='Average Demand', color='blue')
    #plt.plot(demand_max, label='Max Demand (scaled by 10)', color='orange')
    #plt.plot(std, label='Std of Absolute Error', color='red')
    plt.plot(mean, label='Mean of Absolute Error', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Demand and Prediction Error Over Time')
    plt.legend()
    plt.savefig(f"{output_dir}/demand_error_analysis.png")
    plt.close()

    sum_demands = np.sum(labels, axis=0) #[num_nodes]
    max_demand_node = np.argmax(sum_demands)
    min_demand_node = np.argmin(sum_demands)
    mid_demand_node = np.argsort(sum_demands)[num_nodes // 2]

    visualize_sample(pred[:, max_demand_node], labels[:, max_demand_node], output_dir, name='max_demand_node')
    visualize_sample(pred[:, min_demand_node], labels[:, min_demand_node], output_dir, name='min_demand_node')
    visualize_sample(pred[:, mid_demand_node], labels[:, mid_demand_node], output_dir, name='mid_demand_node')

def visualize_sample(pred, labels, output_dir, name):
    pred = np.array(pred)
    labels = np.array(labels)

    time_steps = pred.shape[0]

    plt.figure(figsize=(24, 16))
    plt.plot(labels, label='Labels', color='blue')
    plt.plot(pred, label='Predictions', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.title(f'Predictions vs Labels for Sample Node: {name}')
    plt.legend()
    plt.savefig(f"{output_dir}/predictions_{name}.png")
    plt.close()
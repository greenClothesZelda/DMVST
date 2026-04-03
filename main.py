import torch
import torch.nn.functional as F
from torch.utils.data import Subset

import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from omegaconf import OmegaConf

from dataset_struct.dmvst_dataset import DMVSTDataset, collate_fn
from loss_fn.dmvst_loss import DMVSTLoss
from models.DMVSTModel import DMVST, IRModule
from runners.test import test_loop


log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

BEST_MODEL_METRIC_ALIASES = {
    'rsme': 'rmse',
}

BEST_MODEL_METRIC_DIRECTIONS = {
    'loss': False,
    'mae': False,
    'mape': False,
    'rmse': False,
    'evaluater': False,
}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    import numpy as np

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    mae = np.mean(np.abs(predictions - labels))
    mape = np.mean(np.abs(predictions - labels) / (labels + 1.0)) * 100.0
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    evaluater = mape + 50 * rmse

    return {
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'evaluater': float(evaluater)
    }


def resolve_best_model_metric(train_config):
    metric_name = str(train_config.metric_for_best_model).strip().lower()
    metric_name = BEST_MODEL_METRIC_ALIASES.get(metric_name, metric_name)

    if metric_name.startswith('eval_'):
        metric_key = metric_name[5:]
    else:
        metric_key = metric_name

    if metric_key not in BEST_MODEL_METRIC_DIRECTIONS:
        valid_metrics = ', '.join(sorted(BEST_MODEL_METRIC_DIRECTIONS))
        raise ValueError(
            f"Unsupported train.metric_for_best_model='{train_config.metric_for_best_model}'. "
            f"Use one of: {valid_metrics}"
        )

    train_config.metric_for_best_model = metric_key
    train_config.greater_is_better = BEST_MODEL_METRIC_DIRECTIONS[metric_key]
    return metric_key

def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    # 결정론적 연산을 위한 설정 (필요 시)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

@hydra.main(config_path="configs", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    output_dir = HydraConfig.get().runtime.output_dir

    # 데이터셋 및 데이터로더 설정
    dataset = DMVSTDataset(**config.dataset)
    dataset_size = len(dataset)
    train_end = (int(dataset_size * config.train_split) // dataset.num_nodes) * dataset.num_nodes

    ir_config = config.model.get('IRModule')
    warmup_steps = ir_config.k * dataset.num_nodes if ir_config is not None else 0

    if train_end <= warmup_steps:
        raise ValueError(f"train_end ({train_end}) must be greater than warmup_steps ({warmup_steps}).")
    if train_end >= dataset_size:
        raise ValueError(f"train_end ({train_end}) must be smaller than dataset length ({dataset_size}).")

    line_graph_path = dataset.get_train_graph_path(train_end)

    criterion = DMVSTLoss(
        **config.criterion
    )

    ir_module = None
    if ir_config is not None:
        ir_module = IRModule(dataset=dataset, device=device, k=ir_config.k)

    model_config = OmegaConf.to_container(config.model, resolve=True)
    model_config.pop('IRModule', None)

    model = DMVST(
        **model_config,
        loss_fn=criterion,
        line_graph_path=line_graph_path,
        ir_module=ir_module
    )

    train_indices = range(warmup_steps, train_end)
    eval_indices = range(train_end, dataset_size)

    log.info(
        "Dataset sizes - RetrievalOnly: %s, Train: %s, Eval/Test: %s",
        warmup_steps,
        len(train_indices),
        len(eval_indices)
    )

    args = TrainingArguments(
        **config['train'],
        output_dir=output_dir,
        report_to=[],
        log_level='info'
    )

    best_model_metric = resolve_best_model_metric(args)
    log.info(
        "Best model selection metric: %s (greater_is_better=%s)",
        best_model_metric,
        args.greater_is_better
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Subset(dataset, train_indices),
        eval_dataset=Subset(dataset, eval_indices),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(**config.callbacks.early_stopping)]
    )
    trainer.train()
    
    test_results = test_loop(model, Subset(dataset, eval_indices), output_dir, device, **config.test)
    results.append(test_results)

if __name__ == "__main__":
    run()
    log.info(results)

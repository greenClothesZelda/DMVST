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
from models.DMVSTModel import DMVST
from runners.test import test_loop


log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

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
    train_size = int(dataset_size * config.train_split)

    criterion = DMVSTLoss(
        **config.criterion
    )

    model = DMVST(
        **config.model,
        loss_fn=criterion
    )

    args = TrainingArguments(
        **config['train'],
        output_dir=output_dir,
        report_to=[],
        log_level='info'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Subset(dataset, range(train_size)),
        eval_dataset=Subset(dataset, range(train_size, dataset_size)),
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(**config.callbacks.early_stopping)]
    )
    trainer.train()
    
    test_results = test_loop(model, Subset(dataset, range(train_size, dataset_size)), output_dir, device, **config.test)
    results.append(test_results)

if __name__ == "__main__":
    run()
    log.info(results)

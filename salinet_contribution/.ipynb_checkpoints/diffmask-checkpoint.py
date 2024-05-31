import torch
import lightning.pytorch as pl

# from .diffmask import DiffMaskConfig 

from transformers import get_constant_schedule_with_warmup, get_constant_schedule

import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class MaskConfig:
    model: str
    attn_heads: int = 0
    mlps: int = 0

@dataclass
class DataConfig:
    seed: int
    name: str
    path: str
    workers: int
    val_size: float

@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    checkpoint_path: str
    results_path: str

@dataclass
class DiffMaskConfig:
    seed: int
    mask: MaskConfig
    data: DataConfig
    trainer: TrainerConfig



class DiffMask(pl.LightningModule):
    def __init__(self, config: DiffMaskConfig) -> None:
        super().__init__()
        self.config = config
        self.lambda1 = torch.nn.Parameter(torch.ones((1,)), requires_grad=True)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                params=[self.location],
                lr=self.config.trainer.lr,
            ),
            torch.optim.Adam(
                params=[self.lambda1],
                lr=self.config.trainer.lr,
            ),
        ]

        schedulers = [
            get_constant_schedule(optimizers[0]),
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers


    def optimizer_step(
        self,
        optimizer,
        optimizer_idx,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            self.lambda1.grad *= -1
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            self.lambda1.data = torch.where(
                self.lambda1.data < 0,
                torch.full_like(self.lambda1.data, 0),
                self.lambda1.data,
            )
            self.lambda1.data = torch.where(
                self.lambda1.data > 200,
                torch.full_like(self.lambda1.data, 200),
                self.lambda1.data,
            )

@torch.distributions.kl.register_kl(
    torch.distributions.Bernoulli, torch.distributions.Bernoulli
)
def kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (torch.log(p.probs + 1e-5) - torch.log(q.probs + 1e-5))
    t2 = (1 - p.probs) * (torch.log1p(-p.probs + 1e-5) - torch.log1p(-q.probs + 1e-5))
    return t1 + t2

if __name__ == "__main__":
    # 读取 YAML 配置文件
    with open('diffmask.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    
    # 初始化配置对象
    config_data['trainer']['lr'] = float(config_data['trainer']['lr'])
    config_data['trainer']['momentum'] = float(config_data['trainer']['momentum'])
    config_data['trainer']['weight_decay'] = float(config_data['trainer']['weight_decay'])
    config_data['trainer']['batch_size'] = int(config_data['trainer']['batch_size'])
    config_data['trainer']['epochs'] = int(config_data['trainer']['epochs'])
    
    mask_config = MaskConfig(**config_data['mask'])
    data_config = DataConfig(**config_data['data'])
    trainer_config = TrainerConfig(**config_data['trainer'])
    
    config = DiffMaskConfig(
        seed=config_data['seed'],
        mask=mask_config,
        data=data_config,
        trainer=trainer_config
    )
       
    diffmask = DiffMask(config, device="cuda:0")
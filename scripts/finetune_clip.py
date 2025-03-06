import torch
from torch.utils.data import DataLoader
from models.clip_vln import CLIPVLN
from data_loader.robothor_loader import RoboTHORDataset
import yaml
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 加载数据集
    dataset = RoboTHORDataset(args.config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )

    # 初始化模型
    model = CLIPVLN(args.config)
    model = model.to('cuda')

    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 训练循环
    for epoch in range(config['training']['epochs']):
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to('cuda')
            texts = texts.to('cuda')

            # 前向传播
            outputs = model(images, texts)
            loss = torch.nn.functional.cross_entropy(outputs, torch.randint(0, 4, (outputs.size(0),)).to('cuda'))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % config['training']['log_interval'] == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item()}")

if __name__ == "__main__":
    train()
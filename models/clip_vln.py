import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor
import yaml

class CLIPVLN(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 加载CLIP预训练模型
        self.clip = CLIPModel.from_pretrained(self.config['model']['base_model'])
        self.processor = CLIPProcessor.from_pretrained(self.config['model']['base_model'])
        
        # 冻结图像编码器
        for param in self.clip.visual.parameters():
            param.requires_grad = False
        
        # 自定义路径预测头
        self.trajectory_head = nn.Linear(
            self.config['model']['trajectory_head']['input_dim'],
            self.config['model']['trajectory_head']['output_dim']
        )

    def forward(self, images, texts):
        # 多模态编码
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip(**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        
        # 跨模态特征融合
        fused_features = torch.cat([text_features, image_features], dim=1)
        trajectory_probs = self.trajectory_head(fused_features)
        
        return trajectory_probs

if __name__ == "__main__":
    model = CLIPVLN('configs/config.yaml')
    print(model)
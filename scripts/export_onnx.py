import torch
from models.clip_vln import CLIPVLN
import yaml
import argparse

def export_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='clip_vln.pth')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 加载模型
    model = CLIPVLN(args.config)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # 创建示例输入
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = ["Sample text"]

    # 导出ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        "clip_vln.onnx",
        input_names=["image", "text"],
        output_names=["trajectory"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "text": {0: "batch_size"},
            "trajectory": {0: "batch_size"}
        }
    )

if __name__ == "__main__":
    export_onnx()
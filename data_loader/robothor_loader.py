import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import yaml

class RoboTHORDataset(Dataset):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['dataset']['path']
        self.image_files = [f for f in os.listdir(os.path.join(self.data_dir, 'images')) if f.endswith('.png')]
        self.text_files = [f for f in os.listdir(os.path.join(self.data_dir, 'texts')) if f.endswith('.txt')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, 'images', self.image_files[idx])
        text_path = os.path.join(self.data_dir, 'texts', self.text_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        with open(text_path, 'r') as f:
            text = f.read().strip()
        
        return image, text

if __name__ == "__main__":
    dataset = RoboTHORDataset('configs/config.yaml')
    print(f"Dataset size: {len(dataset)}")
    image, text = dataset[0]
    print(f"First sample: Image shape {image.size}, Text: {text}")
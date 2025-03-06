import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from models.tensorrt_utils import TensorRTInference

class JetsonDeploy:
    def __init__(self, engine_path):
        self.trt_model = TensorRTInference(engine_path)
    
    def process_input(self, image, text):
        # 预处理逻辑
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)
        text_tensor = torch.tensor(np.array([ord(c) for c in text])).unsqueeze(0)
        
        return image_tensor, text_tensor
    
    def infer(self, image, text):
        image_tensor, text_tensor = self.process_input(image, text)
        output = self.trt_model.infer([image_tensor, text_tensor])
        return output

if __name__ == "__main__":
    deployer = JetsonDeploy('clip_vln.engine')
    dummy_image = np.random.rand(224, 224, 3).astype(np.uint8)
    dummy_text = "Sample text"
    output = deployer.infer(dummy_image, dummy_text)
    print(f"Output: {output}")
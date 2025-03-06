import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def infer(self, inputs):
        bindings = [None] * self.engine.num_bindings
        for i, input_data in enumerate(inputs):
            bindings[i] = input_data.data_ptr()
        
        self.context.execute_async(
            bindings=bindings,
            stream_handle=self.stream.handle
        )
        cuda.Context.synchronize()
        
        output = torch.tensor(np.array(self.engine.get_binding_shape(1))).cuda()
        return output

if __name__ == "__main__":
    trt_model = TensorRTInference('clip_vln.engine')
    dummy_input = [torch.randn(1, 3, 224, 224).cuda(), ["Sample text"]]
    output = trt_model.infer(dummy_input)
    print(f"Output shape: {output.shape}")
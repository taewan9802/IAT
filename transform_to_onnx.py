import torch
import torch.onnx
import os
import argparse
import onnx
from model.video_model import IAT

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weight', type=str, default='best_cropmix.pth')
    parser.add_argument('--output_onnx', type=str, default='1107.onnx')
    config = parser.parse_args()
    print(config)
    
    #torch.set_default_tensor_type('torch.FloatTensor')
    model = IAT().eval()
    model.load_state_dict(torch.load(config.pretrained_weight))
    dummy_torch_tensor = torch.rand((1, 3, 270, 240), dtype=torch.float32, device='cpu')

    torch.onnx.export(model=model,
                    args=dummy_torch_tensor,
                    f=config.output_onnx,
                    verbose=True,
                    opset_version=13)
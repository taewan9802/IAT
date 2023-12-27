import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.onnx
import os
import argparse
import tqdm
from model.video_model import IAT
from data_loaders.lol_v1_whole import lowlight_loader_new
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib, quant_modules, enable_onnx_export
from pytorch_quantization.tensor_quant import QuantDescriptor

def collect_stats(model, data_loader, num_batches):
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i, image in tqdm.tqdm(enumerate(data_loader), total=num_batches):
        model(image[0].cuda())
        if i >= num_batches:
            break
    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()

def quantization():
    quant_modules.initialize()
    quant_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose3d.set_default_quant_desc_input(quant_desc_input)

    quant_desc_weight = QuantDescriptor(num_bits=8, calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose3d.set_default_quant_desc_weight(quant_desc_weight)

    val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config.val_batch_size,
                                            shuffle=True,
                                            num_workers=config.num_worker,
                                            pin_memory=True)

    model_quantization = IAT().cuda().eval()
    model_quantization.load_state_dict(torch.load(config.pretrained_weight))

    with torch.no_grad():
        collect_stats(model=model_quantization, data_loader=val_loader, num_batches=config.val_batch_size)
        compute_amax(model=model_quantization, method="percentile", percentile=99.99) 

        total_loss=0
        num_iteration=0
        for iteration, imgs in enumerate(tqdm.tqdm(val_loader)):
            num_iteration = num_iteration+1
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            enhance_img = model_quantization(low_img)
            loss = F.huber_loss(enhance_img, high_img, reduction='mean',delta=1.0)
            total_loss = total_loss + loss
        print("Loss = %.6f" %(total_loss/num_iteration))

    torch.save(model_quantization, 'PTQ.pt')
    model_quantization.cuda().eval()
    #model_quantization.eval()
    #model_quantization.to('cpu')
    dummy_torch_tensor = torch.randn(1, 3, 270, 240, dtype=torch.float32, device='cuda')

    with enable_onnx_export():
        torch.onnx.export(model=model_quantization,
                        args=dummy_torch_tensor,
                        f=config.output_onnx,
                        verbose=True,
                        opset_version=13)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_val_path', type=str, default='dataset/newdataset/crop256/test/low/')
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_worker', type=int, default=6)
    parser.add_argument('--pretrained_weight', type=str, default='best_cropmix.pth')
    parser.add_argument('--output_onnx', type=str, default='PTQ.onnx')
    config = parser.parse_args()
    print(config)
    quantization()


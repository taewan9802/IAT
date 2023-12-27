import os
import torch
import cv2
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary
from thop import profile, clever_format
from model.IAT_main import IAT
from torchvision.transforms import Normalize
from PIL import Image
import glob
import time

def lowlight(file_path):
    image = (np.asarray(Image.open(config.file_name))/ 255.0)
    image = torch.from_numpy(image).cuda().float().permute(2,0,1).unsqueeze(0)
    _, _, enhanced_img = model(input)
    result_path = file_path.replace('data/demo_image/','data/result/')
    torchvision.utils.save_image(enhanced_img, result_path)

if __name__ == '__main__':
    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_name', type=str, default='demo_imgs/temp.jpg')
        parser.add_argument('--pretrained_weight', type=str, default='best_cropmix.pth')
        config = parser.parse_args()
        print(config)

        model = IAT().cuda().eval()
        model.load_state_dict(torch.load(config.pretrained_weight))

        '''
        dummy_torch_tensor = torch.zeros(1, 3, 540, 960).cuda()
        print(pytorch_model_summary.summary(model, dummy_torch_tensor, show_input=True))
        macs, params = profile(model, inputs=(dummy_torch_tensor, ))
        macs, params = clever_format([macs, params], "%.3f")
        # FLOPs(FLoating point OPerations) vs FLOPS(FLoating point OPerations per Second,flop/s)
        FLOPs = macs.rstrip('G') 
        FLOPs = str(float(FLOPs)*2)+'G'
        print(f"Macs : {macs}")
        print(f"FLOPs : {FLOPs}")
        print(f"Params : {params}")
        '''

        filePath = 'data/demo_image/'
        file_list = os.listdir(filePath)

        for _ in file_list:
            test_list = glob.glob(filePath+"*.jpg")
            for image in test_list: 
                present_time = time.time()
                lowlight(image)
                process_time = (time.time() - present_time)
                print('%.6fms' %(process_time*1000))
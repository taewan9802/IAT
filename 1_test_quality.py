import os
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary
from thop import profile, clever_format

from model.new_temp import IAT
from torchvision.transforms import Normalize
from PIL import Image

if __name__ == '__main__':
    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_name', type=str, default='demo_imgs/low2.jpg')
        parser.add_argument('--pretrained_weight', type=str, default='best_cropmix.pth')
        config = parser.parse_args()
        print(config)

        model = IAT().cuda().eval()
        model.load_state_dict(torch.load(config.pretrained_weight))

        image = (np.asarray(Image.open(config.file_name))/ 255.0)
        image = torch.from_numpy(image).cuda().float().permute(2,0,1).unsqueeze(0)
        _, _, enhanced_image = model(image)
        result_path = config.file_name.replace('demo_imgs/','result/')
        torchvision.utils.save_image(enhanced_image, result_path)


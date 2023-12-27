import os
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary
import glob
from thop import profile, clever_format
from utils import PSNR, validation, LossNetwork
#from model.IAT_main import IAT
#from model.IlluminationAdaptiveTransformer import IAT
from model.new_temp import IAT
from torchvision.transforms import Normalize, ToTensor
from PIL import Image

def transform(path, i, index):
    result_path = path.replace('part2/{}/'.format(i), 'result/')
    result_path = result_path.replace('result/', 'result/{}'.format(index))
    print(path)
    print(result_path)
    os.rename(path, result_path)

if __name__ == '__main__':
    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        parser = argparse.ArgumentParser()
        parser.add_argument('--low_data_path', default='data/frame/part2/', type=str)
        config = parser.parse_args()
        print(config)
        
        index = 0

        for i in range(1,228):
            filePath = config.low_data_path
            filePath = filePath.replace('part2/','part2/{}/'.format(i) )
            image_list = glob.glob(filePath+"*.JPG")
            for image in image_list:
                transform(image, i, index)
                index = index+1



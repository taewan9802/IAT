import torch
import os
import argparse
from utils import PSNR
from IQA_pytorch import SSIM, utils
from PIL import Image
import cv2
import numpy as np
import colour

parser = argparse.ArgumentParser()
parser.add_argument('--ori_img_path', type=str, default='demo_imgs/temp.jpg')
#parser.add_argument('--enhanced_img_path', type=str, default='result/1_2.jpg')
#parser.add_argument('--enhanced_img_path', type=str, default='equ_YCrCb.jpg')
parser.add_argument('--enhanced_img_path', type=str, default='result/temp.jpg')
config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES']='0'

ori_img = utils.prepare_image(Image.open(config.ori_img_path).convert("RGB"))
enhanced_img = utils.prepare_image(Image.open(config.enhanced_img_path).convert("RGB"))

psnr = PSNR()
psnr_score = psnr(enhanced_img, ori_img).item()

ssim = SSIM(channels=3)
ssim_score = ssim(enhanced_img, ori_img, as_loss=False)

image1_rgb = cv2.imread(config.ori_img_path)
image2_rgb = cv2.imread(config.enhanced_img_path)
image1_lab = cv2.cvtColor(image1_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
image2_lab = cv2.cvtColor(image2_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
Delta_E = colour.delta_E(image1_lab, image2_lab)

print('The PSNR Value is : %.3f' %(psnr_score))
print('The SSIM Value is : %.3f' %(ssim_score.item()))
print('The Delta-E Value is : %.3f' %(np.mean(Delta_E)))

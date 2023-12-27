import os
import time
import time
import cv2
import argparse
import glob
import numpy as np
from PIL import Image

def make_file_list(file_path):
    filePath = file_path
    file_list = os.listdir(filePath)
    result_list = []
    for _ in file_list:
        test_list = glob.glob(filePath+"*.jpg")
        for image in test_list:
            result_list.append(image)
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--input_image_path', type=str, default="data/frame/")
    parser.add_argument('--output_image_path', type=str, default="data/frame/")
    parser.add_argument('--input_video', type=str, default="data/test.mp4")
    parser.add_argument('--output_video', type=str, default="data/result.mp4")
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=59.94)
    config = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(config.output_video, fourcc, config.fps, (config.width,config.height), isColor=True)
    start = time.time()

    folder_path = "data/1/"
    file_count = len(glob.glob(folder_path + '/*'))
    print(file_count)

    file_list_1 = make_file_list('data/1/')
    file_list_2 = make_file_list('data/2/')
    file_list_3 = make_file_list('data/3/')
    file_list_4 = make_file_list('data/4/')

    for i in range(200):
        image1 = Image.open(file_list_1[i])
        image2 = Image.open(file_list_2[i])
        image3 = Image.open(file_list_3[i])
        image4 = Image.open(file_list_4[i])
        image1 = np.asarray(image1)
        image2 = np.asarray(image2)
        image3 = np.asarray(image3)
        image4 = np.asarray(image4)
        temp1 = np.hstack((image1, image2))
        temp2 = np.hstack((image3, image4))
        result_image = np.vstack((temp1, temp2))
        result.write(result_image)

    end_time = (time.time() - start)
    print('%.6fs' %(end_time))


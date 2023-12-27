import glob
import numpy as np
import cv2
import argparse

def crop(image, config, crop_size):
    low_image_path = image
    high_image_path = image.replace('low/','high/')
    low_result_path = low_image_path.replace('newdataset/', 'newdataset/crop{}/'.format(crop_size))
    high_result_path = high_image_path.replace('newdataset/', 'newdataset/crop{}/'.format(crop_size))
    low_image = cv2.imread(low_image_path)
    high_image = cv2.imread(high_image_path)
    H = low_image.shape[0]
    W = low_image.shape[1]
    for i in range(config.num_patches):
        rr = np.random.randint(0, H - crop_size)
        cc = np.random.randint(0, W - crop_size)
        low_crop = low_image[rr:rr + crop_size, cc:cc + crop_size, :]
        high_crop = high_image[rr:rr + crop_size, cc:cc + crop_size, :]
        low_result_path = low_result_path.replace('.png', '_{}.png'.format(i))
        high_result_path = high_result_path.replace('.png', '_{}.png'.format(i))
        cv2.imwrite(low_result_path, low_crop)
        cv2.imwrite(high_result_path, high_crop)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
    parser.add_argument('--train_data_path', default='dataset/newdataset/train/low/', type=str)
    parser.add_argument('--test_data_path', default='dataset/newdataset/test/low/', type=str)
    parser.add_argument('--num_patches', default=5, type=int, help='Number of patches per image')
    config = parser.parse_args()

    NUM_PATCHES = config.num_patches

    # make train_dataset #
    filePath = config.train_data_path
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 64)
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 128)
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 256)

    # make test_dataset #
    filePath = config.test_data_path
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 64)
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 128)
    image_list = glob.glob(filePath+"*.png")
    for image in image_list:
        crop(image, config, 256)


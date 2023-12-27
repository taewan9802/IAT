import torch
import os
import time
import time
import cv2
import argparse
import glob

#filePath = 'data/test_data/'
#file_list = os.listdir(filePath)
#sum_time = 0
#for file_name in file_list:
#    test_list = glob.glob(filePath+file_name+"/*") 
#    for image in test_list: 
#        print(image)
#        present_time = time.time()
#        process_time = (time.time() - present_time)
#        print('%.6fms' %(process_time*1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    #parser.add_argument('--input_image_path', type=str, default="data/frame/")
    parser.add_argument('--output_image_path', type=str, default="data/video/")
    parser.add_argument('--input_video', type=str, default="data/demo_video/14.mp4")
    #parser.add_argument('--output_video', type=str, default="data/result.mp4")
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    config = parser.parse_args()

    video = cv2.VideoCapture(config.input_video)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    present_frame = 0
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #result = cv2.VideoWriter(config.output_video, fourcc, float(fps), (config.width,config.height), isColor=True)
    start = time.time()
    aaaaa = 2341
    bbbbb = 2
    #ccccc = 3
    #ddddd = 4
    temp = 0

    while True:
        ret, image = video.read()
        if temp == 180:
            end_time = (time.time() - start)
            print('%.6fs' %(end_time))
            break
        if ret:
            temp+=1
            number = '%.4d'%(aaaaa)
            xxx = config.output_image_path + str(number) + '.jpg'
            image = cv2.resize(image,(config.width,config.height))
            cv2.imwrite(xxx,image)
            print(aaaaa)
            aaaaa += 1
            

    video.release()
    #result.release()

    '''
    while True:
        ret, image = video.read()
        if ret:
            if (temp>0) and (temp<=1700):
                number = '%.4d'%(aaaaa)
                xxx = config.output_image_path + str(number) + '.jpg'
                image = cv2.resize(image,(config.width,config.height))
                cv2.imwrite(xxx,image)
                print(aaaaa)
                aaaaa += 4
                temp+=1
            if (temp>1700) and (temp<=3400):
                number = '%.4d'%(bbbbb)
                xxx = config.output_image_path + str(number) + '.jpg'
                image = cv2.resize(image,(config.width,config.height))
                cv2.imwrite(xxx,image)
                print(bbbbb)
                bbbbb += 4
                temp+=1
            if (temp>3400) and (temp<=5100):
                number = '%.4d'%(ccccc)
                xxx = config.output_image_path + str(number) + '.jpg'
                image = cv2.resize(image,(config.width,config.height))
                cv2.imwrite(xxx,image)
                print(ccccc)
                ccccc += 4
                temp+=1
            if (temp>5100) and (temp<=6800):
                number = '%.4d'%(ddddd)
                xxx = config.output_image_path + str(number) + '.jpg'
                image = cv2.resize(image,(config.width,config.height))
                cv2.imwrite(xxx,image)
                print(ddddd)
                ddddd += 4
                temp+=1
            if (temp>6800):
                print('empty')
        else:
            end_time = (time.time() - start)
            print('%.6fs' %(end_time))
            break
    video.release()
    result.release()
    '''

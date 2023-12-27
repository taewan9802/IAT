import os
import torch
import argparse
import glob
import time
#import cv2
#import torchvision
#import numpy as np
#import matplotlib.pyplot as plt
#from torchvision.transforms import Normalize
#from torchvision.io import write_video
from torchvision.io import read_image
from skvideo.io import FFmpegWriter
from model.video_model import IAT

'''
torch BGR image (B,C,H,W)
'''

if __name__ == '__main__':
    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_path', type=str, default='data/video/')
        parser.add_argument('--pretrained_weight', type=str, default='best_cropmix.pth')
        config = parser.parse_args()
        print(config)

        model = IAT().cuda().eval()
        model.load_state_dict(torch.load(config.pretrained_weight))

        '''
        ### model summary ###
        from pytorch_model_summary import summary
        from thop import profile, clever_format
        dummy_torch_tensor = torch.zeros(1, 3, 540, 960).cuda()
        print(summary(model, dummy_torch_tensor, show_input=True))
        macs, params = profile(model, inputs=(dummy_torch_tensor, ))
        macs, params = clever_format([macs, params], "%.3f")
        # FLOPs(FLoating point OPerations) vs FLOPS(FLoating point OPerations per Second,flop/s)
        FLOPs = macs.rstrip('G')
        FLOPs = str(float(FLOPs)*2)+'G'
        print(f"Macs : {macs}")
        print(f"FLOPs : {FLOPs}")
        print(f"Params : {params}")
        '''

        ### skvideo.io.FFmpegWriter ###
        fps = str(60.0)
        bitrate = str(1300*1000)
        result_video = FFmpegWriter("3333.mp4", outputdict={'-r':fps, '-b':bitrate, '-vcodec':'libx264', '-pix_fmt':'rgb24'})
        frame_torch_tensor = torch.zeros(270, 480, 3, dtype=torch.uint8, device='cuda')
        
        filePath = config.file_path
        file_list = os.listdir(filePath)

        test_list = glob.glob(filePath+"*.jpg")
        test_list.sort(reverse=False)
        
        frame=1 # control frame count #
        #print(test_list)

        for image_name in test_list:
            image = (((read_image(path=image_name)).cuda())/255.0).unsqueeze(0)

            frame_torch_tensor[0:270, 0:480, 0:3].add_(model(image))
            result_video.writeFrame(frame_torch_tensor.detach().cpu().numpy())
            frame_torch_tensor.sub_(frame_torch_tensor)
            print(frame)
            frame+=1
            if frame==20000:
                break
        result_video.close()


        '''
        ### skvideo.io.FFmpegWriter ###
        fps = str(60.0)
        bitrate = str(1300*1000)
        result_video = FFmpegWriter("3333.mp4", outputdict={'-r':fps, '-b':bitrate, '-vcodec':'libx264', '-pix_fmt':'rgb24'})
        frame_torch_tensor = torch.zeros(1080, 1920, 3, dtype=torch.uint8, device='cuda')
        
        filePath = config.file_path
        file_list = os.listdir(filePath)
        image_number = 0

        test_list = glob.glob(filePath+"*.jpg")
        test_list.sort(reverse=False)
        
        frame=1 # control frame count #
        #print(test_list)

        for image_name in test_list:
            image = (((read_image(path=image_name)).cuda())/255.0).unsqueeze(0)
            #print(image.shape)
            image_number+=1
            if (image_number==1):
                #present_time = time.time()
                frame_torch_tensor[0:540, 0:960, 0:3].add_(model(image))
                #process_time = (time.time() - present_time)
                #print('%.6fms' %(process_time*1000))

                continue
            if (image_number==2):
                frame_torch_tensor[0:540, 960:1920, 0:3].add_(model(image))
                continue
            if (image_number==3):
                frame_torch_tensor[540:1080, 0:960, 0:3].add_(model(image))
                continue
            if (image_number==4):
                frame_torch_tensor[540:1080, 960:1920, 0:3].add_(model(image))
                #print(frame_torch_tensor.shape)
                result_video.writeFrame(frame_torch_tensor.detach().cpu().numpy())
                frame_torch_tensor.sub_(frame_torch_tensor)
                image_number=0
                
                
                frame+=1
                if frame==50:
                    break
        result_video.close()
        '''

        '''
        ### add version ###
        for image_name in test_list:
            #print(image_name)
            image = (((read_image(path=image_name)).cuda())/255.0).unsqueeze(0)
            enhanced_img = model(image)

            if image_number==2:
                temp2=torch.cat((temp1,enhanced_img),dim=1)
            if image_number==4:
                temp3=torch.cat((temp1,enhanced_img),dim=1)
                result_img=(torch.cat((temp2,temp3),dim=0)).detach().cpu().numpy()
                result_video.writeFrame(result_img)
                image_number=1
                
                process_time = (time.time() - present_time)
                print('%.6fms' %(process_time*1000))
                present_time = time.time()

                frame+=1
                if frame==30:
                    break

                continue
            if (image_number==1) or (image_number==3):
                temp1=enhanced_img
            image_number+=1
        result_video.close()
        '''



        '''
        ### torchvision.io.write_video ###
        filePath = config.file_path
        file_list = os.listdir(filePath)
        image_number = 1
        test_list = glob.glob(filePath+"*.jpg")
        test_list.sort(reverse=False)

        frame_torch_tensor = torch.zeros(1, 540*2, 960*2, 3).cpu()
        frame=1
        #print(test_list)
        present_time = time.time()
        for image_name in test_list:
            print(image_name)
            image = (((read_image(path=image_name)).cuda())/255.0).unsqueeze(0)
            enhanced_img = model(image)
            if image_number==2:
                temp2=torch.cat((temp1,enhanced_img),dim=2)
            if image_number==4:
                temp3=torch.cat((temp1,enhanced_img),dim=2)
                result_img=(torch.cat((temp2,temp3),dim=1)).cpu()
                #print(result_img)
                #print(result_img.shape)
                frame_torch_tensor = torch.cat((frame_torch_tensor,result_img),dim=0)
                process_time = (time.time() - present_time)
                print('%.6fms' %(process_time*1000))
                present_time = time.time()
                #print(frame_torch_tensor.shape)
                image_number=1


                frame+=1
                if frame==50:
                    break


                continue
            if (image_number==1) or (image_number==3):
                temp1=enhanced_img
            image_number+=1
            #result_path = image_name.replace('data/video/','result/video/')
            #torchvision.utils.save_image(enhanced_img, result_path)
            #process_time = (time.time() - present_time)
            #print('%.6fms' %(process_time*1000))
        write_video(filename='1111.mp4',video_array=frame_torch_tensor,fps=10.0)
        process_time = (time.time() - present_time)
        print('%.6fms' %(process_time*1000))
        '''

        '''
        ### cv2.VideoWriter ###
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result_video = cv2.VideoWriter('2222.mp4', fourcc, 10.0, (1920,1080))

        filePath = config.file_path
        file_list = os.listdir(filePath)
        image_number = 1
        test_list = glob.glob(filePath+"*.jpg")
        test_list.sort(reverse=False)

        frame=1
        #print(test_list)
        present_time = time.time()
        for image_name in test_list:
            print(image_name)
            image = (((read_image(path=image_name)).cuda())/255.0).unsqueeze(0)
            enhanced_img = model(image)
            if image_number==2:
                temp2=torch.cat((temp1,enhanced_img),dim=1)
            if image_number==4:
                temp3=torch.cat((temp1,enhanced_img),dim=1)
                # change BGR -> RGB #
                result_img=((torch.cat((temp2,temp3),dim=0))[:,:,[2,1,0]]).detach().cpu().numpy()
                result_video.write(result_img)
                image_number=1

                process_time = (time.time() - present_time)
                print('%.6fms' %(process_time*1000))
                present_time = time.time()

                frame+=1
                if frame==30:
                    break

                continue
            if (image_number==1) or (image_number==3):
                temp1=enhanced_img
            image_number+=1
        result_video.release()
        '''



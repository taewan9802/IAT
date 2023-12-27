import numpy as np
import os
import time
import glob
import argparse
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torchvision.io import read_image
from PIL import Image
import cv2

def load_engine(engine_file):
    print('Reading engine file')
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as engine:
        return engine.deserialize_cuda_engine(f.read())

def context_initialize(engine):
    context = engine.create_execution_context()
    context.set_binding_shape(engine.get_binding_index('inputs.1'), (1, 3, IMAGE_HEIGHT, int(IMAGE_WIDTH/2)))
    return context

def memory_initialize(engine):
    bindings = []
    dummy_numpy_tensor = np.zeros((1, 3, IMAGE_HEIGHT, int(IMAGE_WIDTH/2)), dtype=np.float32)
    stream = cuda.Stream()
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(dummy_numpy_tensor)
            host_memory = cuda.mem_alloc(dummy_numpy_tensor.nbytes)
            bindings.append(int(host_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            device_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(device_memory))
    return input_buffer, output_buffer, host_memory, device_memory, bindings, stream

def image_preprocess(image_name):
    image = (read_image(path=image_name)/255.0).unsqueeze(0).numpy()
    image = image[0:1, 0:3, 0:IMAGE_HEIGHT, int(IMAGE_WIDTH/2):IMAGE_WIDTH]
    return np.ascontiguousarray(image) # input_buffer


def inference(input_buffer, output_buffer, host_memory, device_memory, bindings, stream):
    cuda.memcpy_htod_async(host_memory, input_buffer, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_buffer, device_memory, stream)
    stream.synchronize()
    return output_buffer

def image_postprocess(output_buffer, image_name):
    image_name = image_name.replace(config.file_path, config.result_path)
    result = np.reshape(output_buffer, (1, 3, IMAGE_HEIGHT, int(IMAGE_WIDTH/2))).squeeze()
    result = np.moveaxis(result, 0, 2)
    mn = result.min()
    mx = result.max()
    result = (((result-mn)/(mx-mn))*255.0)
    result = np.round(result).astype(np.uint8)
    result = np.asarray(Image.fromarray(result, 'RGB'))
    #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    cuda.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='data/video/')
    #parser.add_argument('--file_path', type=str, default='data/1103_high/')
    #parser.add_argument('--engine_file', type=str, default='2_temp2.trt')
    #parser.add_argument('--engine_file', type=str, default='6_temp6.trt')
    #parser.add_argument('--engine_file', type=str, default='7_temp7.trt')
    #parser.add_argument('--engine_file', type=str, default='1107_1.trt')
    #parser.add_argument('--engine_file', type=str, default='float32_1.trt')
    parser.add_argument('--engine_file', type=str, default='quant_int8.trt')
    parser.add_argument('--result_path', type=str, default='result/')
    parser.add_argument('--image_height', type=int, default=270)
    parser.add_argument('--image_width', type=int, default=480)
    config = parser.parse_args()
    #print(config)

    IMAGE_HEIGHT = int(config.image_height)
    IMAGE_WIDTH = int(config.image_width)
    TRT_LOGGER = trt.Logger()
    #TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    engine = load_engine(config.engine_file)
    context = context_initialize(engine)
    input_buffer, output_buffer, host_memory, device_memory, bindings, stream = memory_initialize(engine)
    file_list = glob.glob(config.file_path+"*.jpg")
    file_list.sort(reverse=False)
    
    frame_number = 1

    print("Running TensorRT inference for IAT")
    for image_name in file_list:
        frame_number = frame_number + 1
        if frame_number % 3 == 0:
            numpy_frame = np.uint8(cv2.imread(image_name))
            present_time = time.time()
            input_buffer = image_preprocess(image_name) #image->input_buffer
            output_buffer = inference(input_buffer, output_buffer, host_memory, device_memory, bindings, stream)
            result_image = image_postprocess(output_buffer, image_name) #output_buffer->image
            process_time = (time.time() - present_time)
            numpy_frame[0:IMAGE_HEIGHT, int(IMAGE_WIDTH/2):IMAGE_WIDTH, [2,1,0]] = result_image
            #numpy_frame = cv2.resize(numpy_frame, (IMAGE_WIDTH*2, IMAGE_HEIGHT*2), interpolation= cv2.INTER_LINEAR)
            numpy_frame = cv2.resize(numpy_frame, (IMAGE_WIDTH*2, IMAGE_HEIGHT*2), interpolation= cv2.INTER_CUBIC)
            fps = 'FPS: {:.2f}'.format(1.0/process_time)
            cv2.putText(numpy_frame, fps, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 1.0, (250, 250, 250), thickness=1)
            cv2.imshow('Low Light Enhancement', numpy_frame)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    host_memory.free()
    device_memory.free()




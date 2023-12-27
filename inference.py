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

def load_engine(engine_file):
    print("Reading engine from file {}".format(engine_file))
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as engine:
        return engine.deserialize_cuda_engine(f.read())

def context_initialize(engine):
    context = engine.create_execution_context()
    context.set_binding_shape(engine.get_binding_index('inputs.1'), (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
    return context

def memory_initialize(engine):
    bindings = []
    dummy_numpy_tensor = np.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
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
    #print(image.dtype)
    #print(image.shape)
    #print(image)
    return np.ascontiguousarray(image) # input_buffer

def inference(input_buffer, output_buffer, host_memory, device_memory, bindings, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(host_memory, input_buffer, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buffer, device_memory, stream)
    # Synchronize the stream
    stream.synchronize()
    return output_buffer

def image_postprocess(output_buffer, image_name):
    image_name = image_name.replace('data/video/','result/')
    #print(output_buffer)
    #print(type(output_buffer))
    #print(output_buffer.dtype)
    #print(output_buffer.shape)
    result = np.reshape(output_buffer, (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)).squeeze()
    result = np.moveaxis(result, 0, 2)
    mn = result.min()
    mx = result.max()
    result = (((result-mn)/(mx-mn))*255.0).astype(np.uint8)

    #result = np.round(result).astype(np.uint8)
    result = Image.fromarray(result, 'RGB')
    result.save(image_name)
    return

def count_fps(time_time):
    fps = time_time
    return fps

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    cuda.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='data/')
    parser.add_argument('--engine_file', type=str, default='7_temp7.trt')
    #parser.add_argument('--engine_file', type=str, default='IAT_quantization_simple.trt')
    parser.add_argument('--result_path', type=str, default='result/')
    parser.add_argument('--image_height', type=int, default=270)
    parser.add_argument('--image_width', type=int, default=480)
    config = parser.parse_args()
    print(config)

    IMAGE_HEIGHT = config.image_height
    IMAGE_WIDTH = config.image_width
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    engine = load_engine(config.engine_file)
    context = context_initialize(engine)
    input_buffer, output_buffer, host_memory, device_memory, bindings, stream = memory_initialize(engine)
    file_list = glob.glob(config.file_path+"*.jpg")
    file_list.sort(reverse=False)
    fps = 0.0
    count = 1

    print("Running TensorRT inference for IAT")
    for image_name in file_list:
        input_buffer = image_preprocess(image_name) #image->input_buffer

        present_time = time.time()
        output_buffer = inference(input_buffer, output_buffer, host_memory, device_memory, bindings, stream)
        process_time = (time.time() - present_time)
        print('%.6fms' %(process_time*1000))

        image_postprocess(output_buffer, image_name) #output_buffer->image
        
        if count == 10:
            break
        count += 1

    host_memory.free()
    device_memory.free()







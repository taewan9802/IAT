import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
import cv2
from PIL import Image
import time

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "IAT_v3.trt"
input_file  = "temp.jpg"
output_file = "result_image.jpg"

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_file, output_file):
    print("Reading input image from file {}".format(input_file))
    image = (read_image(path=input_file)/255.0).unsqueeze(0)
    image = image.numpy().astype(np.float32)

    '''
    image = cv2.imread(input_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) # BGR -> RGB
    image = np.expand_dims(image, axis=0)
    print(image.dtype)
    print(image.shape)
    print(image)
    '''

    '''
    with Image.open(input_file) as image:
        #mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        #stddev = np.array([0.229, 0.224, 0.225]).astype(np.float32)
        #image = (np.asarray(image).astype(np.float32)/float(255.0)-mean)/stddev
        #image = np.moveaxis(image, 2, 0) # HWC -> CHW
        #image = np.expand_dims(image, axis=0)

        print(image)

        mn = image.min()
        mx = image.max()

        print(image.min())
        print(image.max())
    '''

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index('input.1'), (1, 3, 540, 960))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(image)
                input_memory = cuda.mem_alloc(image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()

        present_time = time.time()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        process_time = (time.time() - present_time)
        print('%.6fms' %(process_time*1000))

        result = output_buffer

        input_memory.free()
        output_memory.free()

        result = np.reshape(result, (540, 960, 3))

        mn = result.min()
        mx = result.max()

        result = (((result-mn)/(mx-mn))*255.0)

        result = np.round(result).astype(np.uint8)
        #result = np.clip(result, 0, 255)

        #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(output_file, result)

        result=Image.fromarray(result)
        result.save(output_file)
        #result = torch.from_numpy(result).type(torch.uint8).permute(2,0,1)
        #save_image(result, output_file)

print("Running TensorRT inference for IAT")
with load_engine(engine_file) as engine:
    infer(engine, input_file, output_file)
    


import os
import numpy as np
import torch, torchvision
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as v1
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

def load_engine(engine_file_path):

    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine_deserialized = runtime.deserialize_cuda_engine(f.read())
    return engine_deserialized


path_to_original_imgs = f'resnet18/images/'
labels_path = f'resnet18/imagenet-classes.txt'
original_img_list = os.listdir(path_to_original_imgs)


with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f]

img_list = []
img_path = []


logger = trt.Logger(trt.Logger.WARNING)
engine_file_path = '/workdir/resnet18/resnet.engine'
engine = load_engine(engine_file_path)
context = engine.create_execution_context()

for img in original_img_list:

    if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):

        full_path = os.path.join(path_to_original_imgs, img)
        image = Image.open(full_path)
        image = image.resize((244,244), Image.NEAREST)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = img_np.transpose(2,0,1)
        img_np = np.expand_dims(img_np, axis=0)
        img_list.append(img_np)
        img_path.append(full_path)

        input = np.ascontiguousarray(img_np)
        output = np.zeros((1, 1000))

        d_input = cuda.mem_alloc(input.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        bindings = [d_input, d_output]

        cuda.memcpy_htod(d_input, input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)

        label = np.argmax(output)
        print(f'Actual: {img} --- Prediction: {labels[label]}   ')

        d_input.free()
        d_output.free()


        # img_cpy = np.squeeze(img_np, axis=0)
        # # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert channels to BGR order
        # img_cpy = img_cpy * 255
        # img_cpy = img_cpy.astype(np.uint8)    
        # plt.imshow(img_cpy)
        # plt.savefig('output_image.png')
        # break
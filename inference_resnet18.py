

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as v1
import torch


from PIL import Image

np.bool = np.bool_

class TRTInference:

    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file) -> None:
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_file_path = engine_file_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.class_labels_file = class_labels_file

        self.engine = self.load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

        with open(class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]


    def load_engine(self, engine_file_path):

        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
        return engine_deserialized

    def preprocess_img(self, original_images_path):
        img_list = []
        img_path = []

        for img_original in os.listdir(original_images_path):
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('.jpeg'):

                img_full_path = os.path.join(original_images_path, img_original)
                image = Image.open(img_full_path)

                img = image.resize((244,244), Image.NEAREST)
                img_np = np.array(img).astype(np.float32) / 255.0
                img_np = img_np.transpose((2,0,1)) 
                img_np = np.expand_dims(img_np, axis=0)
                self.save_image_to_plot(self, img_np, img_original)
                img_list.append(img_np)
                img_path.append(img_full_path)

        return img_list, img_path
    
    def save_image_to_plot(self, img_np, img_original):

        img_to_plot = img_np.squeeze().transpose((1, 2, 0))  # Remove extra dimensions and rearrange to (height, width, channels)
        plt.imshow(img_to_plot)
        plt.show()
        plt.imshow(img_to_plot)
        plt.savefig(f'/workdir/resnet18/images_plot/{img_original}.png')
    
    def post_process_img(self, outputs):

        class_indices = []

        for output in outputs:
            class_idx = output.argmax()
            print(f'Class Detected:', self.class_labels[class_idx])
            class_indices.append(class_idx)
        
        return class_indices
    
    def inference_detection(self, image_path):

        input_list, full_img_paths = self.preprocess_img(image_path)
        results = []

        for inputs, full_img_path in zip(input_list, full_img_paths):
            inputs = np.ascontiguousarray(inputs)
            outputs = np.empty(self.output_shape)

            d_inputs = cuda.mem_alloc(inputs.nbytes)
            d_outputs = cuda.mem_alloc(outputs.nbytes)
            bindings = [d_inputs, d_outputs]
            cuda.memcpy_htod(d_inputs, inputs)
            self.context.execute_v2(bindings)
            cuda.memcpy_dtoh(outputs, d_outputs)

            result = self.post_process_img(outputs)

            d_inputs.free()
            d_outputs.free()

            results.append(result)
            self.display_recognized_images(full_img_path, result)

        return results      
    
    def display_recognized_images(self, image_path, class_labels): # NOT WORKING TO FIX

        image = Image.open(image_path)

        for class_idx in class_labels:

            path_to_detected_imgs = '/workdir/resnet18/images_detected'

            if not os.path.exists(path_to_detected_imgs):
                os.mkdir(path_to_detected_imgs)

            plt.imshow(image)
            plt.title(f'Recognized Image : {self.class_labels[class_idx]}')
            plt.axis('off')

            save_img = os.path.join(path_to_detected_imgs, f'{self.class_labels[class_idx]}.jpg')
            plt.savefig(save_img)
            plt.close()

            return image, save_img
        
# engine_file_path = '/workdir/resnet18/trt_engines/resnet.engine'        
# engine_file_path = '/workdir/resnet18/trt_engines/resnettrt.engine'        
engine_file_path = '/workdir/resnet18/trt_engines/torch_model_trt.engine'

input_shape = (1,3,224,224)
output_shape = (1,1000)
path_to_original_imgs = 'resnet18/images'
class_labels = '/workdir/resnet18/imagenet-classes.txt'


inference = TRTInference(engine_file_path=engine_file_path, 
                         input_shape=input_shape, 
                         output_shape=output_shape,
                         class_labels_file=class_labels
                         )

inference.inference_detection(path_to_original_imgs)


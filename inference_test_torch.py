import os
import numpy as np
import torch, torchvision
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as v1
import matplotlib.pyplot as plt

path_to_original_imgs = f'resnet18/images/'
labels_path = f'resnet18/imagenet-classes.txt'
original_img_list = os.listdir(path_to_original_imgs)

with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f]

img_list = []
img_path = []

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.eval()

for img in original_img_list:

    if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):

        full_path = os.path.join(path_to_original_imgs, img)
        image = Image.open(full_path)
        img = image.resize((244,244), Image.NEAREST)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, axis=0)
        img_list.append(img_np)
        img_path.append(full_path)

for image_path in img_path:

    image = read_image(image_path) / 255
    transforms = torch.nn.Sequential(
        v1.Resize((224,224)),
        v1.RandomHorizontalFlip(p=0.1)
    )

    scripted_transforms = torch.jit.script(transforms)
    image = scripted_transforms(image).float().unsqueeze(dim=0)
    output = model(image)
    class_output = torch.argmax(output)
    print(f'{image_path} -------- {labels[class_output]}')

        
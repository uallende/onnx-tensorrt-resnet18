import torch, torchvision
from torch2trt import torch2trt

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model = model.cuda()
model.eval()  # Set the model to evaluation mode
x = torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()  # Adjust the shape as necessary

model_trt = torch2trt(model, [x])


with open('torch_model_trt.engine', 'wb') as f:
    f.write(model_trt.engine.serialize())

y = model_trt(x)
print(y)
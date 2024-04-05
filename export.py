import torch, torchvision

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.eval()
dummy_img = torch.rand(1,3,224,224)
torch.onnx.export(model, dummy_img, 'resnet18/resnet18.onnx')
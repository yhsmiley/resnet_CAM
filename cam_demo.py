import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms, datasets
from torchvision.utils import save_image
import os

from cam import CAM, GradCAM, GradCAMpp
from visualize import visualize, reverse_normalize

image = Image.open('/home/yinghui/Projects/AICCC/testing_images/train_all/PalmTree/135230075.jpg')
train_data_dir = '/home/yinghui/Projects/AICCC/testing_images/train'
model_dir = '/home/yinghui/Projects/AICCC/resnet_CAM/models/model_2.pth'

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['model'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        model.class_to_idx = chpt['class_to_idx']
        model.fc = chpt['classifier']
    else:
        print("Sorry base architecture not recognized")
        exit()
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

# preprocessing. mean and std from ImageNet
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# convert image to tensor
tensor = preprocess(image)
tensor = tensor.to(device)

# reshape 4D tensor (N, C, H, W)
tensor = tensor.unsqueeze(0)

# datasets = datasets.ImageFolder(train_data_dir)
# class_names = datasets.classes
# num_classes = len(class_names)

model = load_model(model_dir)
model = model.to(device)
model.eval()

# the layer you want to visualize
target_layer = model.layer4[2].conv3

# wrapper for class activation mapping. Choose one of the following.
# wrapped_model = CAM(model, target_layer)
# wrapped_model = GradCAM(model, target_layer)
wrapped_model = GradCAMpp(model, target_layer)

cam = wrapped_model(tensor)
cam = cam.cpu()

# reverse normalization for display
img = reverse_normalize(tensor)
heatmap = visualize(img, cam)

# save image
# save_image(heatmap, './img/cam.png')
# save_image(heatmap, './img/gradcam.png')
save_image(heatmap, './img/gradcampp.png')
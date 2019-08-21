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
model_dir = '/home/yinghui/Projects/AICCC/resnet_CAM/models/less_pics/model_10.pth'

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

datasets = datasets.ImageFolder(train_data_dir)
class_names = datasets.classes
num_classes = len(class_names)

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)

model.load_state_dict(torch.load(model_dir))
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
from torchvision import models
import torch.nn as nn
from collections import OrderedDict

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(num_ftrs, 2))
]))
model.fc = classifier
print(model)


for name, child in model.named_children():
    print(name)

# print(model)
from torchvision import models
import torch.nn as nn

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

for name, child in model.named_children():
    print(name)

# print(model)
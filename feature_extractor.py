### Based on PyTorch's transfer learning tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html ###

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import logging
import sys
from collections import OrderedDict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

matplotlib.use("TkAgg")

def main():

    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([mean_nums])
        std = np.array([std_nums])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def train_model(model, criterion, optimizer, scheduler, num_epochs=10, checkpoint_period=10):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        logger.info("Start training")
        for epoch in range(num_epochs):
            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                current_loss = 0.0
                current_corrects = 0

                # Here's where the training happens
                # print('Iterating through data...')
                logger.info('Iterating through data...')

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #  We need to zero the gradients, don't forget it
                    optimizer.zero_grad()

                    # Time to carry out the forward training poss
                    # We only need to log the loss stats if we are in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # We want variables to hold the loss statistics
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += torch.sum(preds == labels.data)

                epoch_loss = current_loss / dataset_sizes[phase]
                epoch_acc = current_corrects.double() / dataset_sizes[phase]

                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # Make a copy of the model if the accuracy on the validation set has improved
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            if (epoch+1) % checkpoint_period == 0:
                output_path = os.path.join(output_dir, 'model_{}.pth'.format(epoch))
                checkpoint = {'model': 'resnet50',
                              'classifier': model.fc,
                              'state_dict': model.state_dict(),
                              'class_to_idx': model.class_to_idx
                            }
                torch.save(checkpoint, output_path)
                # print('Saved model_{}.pth'.format(epoch+1))
                logger.info('Saved model_{}.pth'.format(epoch))

            # print()
            logger.info('')

        time_since = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
        logger.info('Best val Acc: {:4f}'.format(best_acc))

        # Now we'll load in the best model weights and return it
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_handeled = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(dataloaders['val'])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_handeled += 1
                    ax = plt.subplot(num_images//2, 2, images_handeled)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_handeled == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

        fig.savefig('visualize_model.jpg')

    def make_logger(name, save_dir, save_filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, save_filename+".txt"), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    # Make transforms and use data loaders
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    chosen_transforms = {'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ])
    , 'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
    ]),
    }

    # Set the directory for the data
    data_dir = '/home/yinghui/Projects/AICCC/testing_images'
    output_dir = '/home/yinghui/Projects/AICCC/resnet_CAM/models'

    logger = make_logger("Resnet_CAM", output_dir,'log')
    # logger.info("Loaded configuration file {}".format(config_file))
    # logger.info("Running with config:\n{}".format(cfg))

    # Use the image folder function to create datasets.
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              chosen_transforms[x])
                      for x in ['train', 'val']}


    # Make iterables with the dataloaders.
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=64,
                                                 shuffle=True, num_workers=2)
                  for x in ['train', 'val']}


    dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val']}
    class_names = chosen_datasets['train'].classes
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # visualize some images
    # Going to grab some of the training data to visualize
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])

    # Setting up the model
    # We need to load in pretrained and reset final fully connected
    res_mod = models.resnet50(pretrained=True)
    for param in res_mod.parameters():
        param.requires_grad = False

    # the parameters of imported models are set to requires_grad=True by default
    num_ftrs = res_mod.fc.in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, num_classes))
    ]))
    res_mod.fc = classifier

    res_mod.class_to_idx = chosen_datasets['train'].class_to_idx

    res_mod = res_mod.to(device)
    criterion = nn.CrossEntropyLoss()

    # Here's the main change, instead of all paramters being optimized
    # Only the params of the final layers are being optmized
    optimizer_ft = optim.SGD(res_mod.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # How you can selectively unfreeze layers...
    # in order to selectively unfreeze layers, need to specify the layers that require grad
    for name, child in res_mod.named_children():
       if name in ['layer3', 'layer4']:
           # print(name + ' has been unfrozen.')
           logger.info(name + ' has been unfrozen.')
           for param in child.parameters():
               param.requires_grad = True
       else:
           for param in child.parameters():
               param.requires_grad = False

    # also need to update optimization function
    # only optimize those that require grad
    optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, res_mod.parameters()), lr=0.001, momentum=0.9)

    ## if only reset fully connected layer
    # base_model = train_model(res_mod, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    ## if selectively unfreeze layers
    base_model = train_model(res_mod, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=10, checkpoint_period=1)
    visualize_model(base_model)

if __name__ == '__main__':
    main()

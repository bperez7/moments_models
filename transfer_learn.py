# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy

from utils import extract_frames, load_frames, render_frames
from os import listdir
from os.path import isfile, join
import models
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#plt.ion()   # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {"train":8, "val":0}
def data_prep():
    num_segments = 8
    videos_path = "./training_videos"

    print('Extracting frames using ffmpeg...')

    transform = models.load_transform()

    training_x_list = []
    training_y_list = []

    video_files = [f for f in listdir(videos_path) if isfile(join(videos_path, f))]
    video_files = filter(lambda x: x[-3:] in {"mp4","mov","MOV"}, video_files)

    for video_file in video_files:
        print(videos_path+"/"+video_file)

        frames = extract_frames(videos_path + "/" + video_file, num_segments)
        video_input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
        training_x_list.append(video_input)
        training_y_list.append(torch.tensor([305]))

    x_stack = torch.stack(training_x_list)
    y_stack = torch.stack(training_y_list)
    #print(np_x_list[0])
   # tensor_training_x_list = torch.tensor(x_stack)
   # tensor_training_y_list = torch.tensor(np_y_list)

    return x_stack, y_stack



def train_model(model, criterion, optimizer, scheduler, data, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

        # Iterate over data.
        # for inputs, labels in dataloaders[phase]:
        #for inputs, labels in enumerate(data,0):
            inputs = data[0]
            labels=data[1]
            for i in range(len(labels)):
                input = inputs[i]
                label = labels[i]
                print('starting')
                input = input.to(device)
                label = label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                   # "success in output"
                    _, pred = torch.max(output, 1)
                    print(pred)
                    loss = criterion(output, label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
              #  running_loss += loss.item() * inputs.size(0)
              #  running_corrects += torch.sum(pred == labels.data)
                if phase == 'train':
                    scheduler.step()

            #epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

           # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           #     phase, epoch_loss, epoch_acc))

            # deep copy the model
        #    if phase == 'val' and epoch_acc > best_acc:
        #        best_acc = epoch_acc
        #        best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__=="__main__":
    training_data = data_prep()
    model = models.load_model("resnet3d50")
    expansion = 4
    model.fc = nn.Linear(512 * expansion, 306)
    model.last_linear = nn.Linear(in_features=512 * expansion, out_features=306, bias=True)
    print(model)
    #top_module = nn.Sequential(nn.Linear(n_inputs,))
   # model = nn.Sequential(model,)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    train_model(model,criterion, optimizer_ft, exp_lr_scheduler,training_data)

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt

#augmentors
import vidaug.augmentors as va

import os
import pandas as pd
#from torchvision.io import read_image
import models
from utils import extract_frames, load_frames, render_frames

class CustomImageTrainAugmentedDataset(Dataset):
    def __init__(self, annotations_file, vid_dir, transform=models.load_transform(), target_transform=None):
        self.vid_labels = pd.read_csv(annotations_file)
        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.vid_labels)

    def __getitem__(self, idx):
        vid_path = os.path.join(self.vid_dir, self.vid_labels.iloc[idx, 0])

        #image = read_image(vid_path)
        num_segments = 8 #may need to adjust

        frames = extract_frames(vid_path, num_segments)

        sometimes = lambda aug: va.Sometimes(.2, aug)  # Used to apply augmentor with 20% probability
        seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
            sometimes(va.HorizontalFlip()),  # horizontally flip the video with 100% probability
            sometimes(va.GaussianBlur(sigma=1)),
            sometimes(va.ElasticTransformation()),
            sometimes(va.Salt()),
         #   sometimes(va.PiecewiseAffineTransform()),
         #   sometimes(va.Superpixel()),
            sometimes(va.Pepper()),
            sometimes(va.Add()),
            sometimes(va.Multiply())
        ])
        video_aug = seq(frames)
        #frames = extract_frames("videos/label_videos/excavating/excavating_2.mp4")
        #video = torch.stack([self.transform(frame) for frame in frames], 1).unsqueeze(0)
        video = torch.stack([self.transform(frame) for frame in video_aug], 1)

        label = self.vid_labels.iloc[idx, 1]

        #if self.transform:
        #    image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            #hotfix
            label = label.to(dtype=torch.long)
        return video, label

class CustomImageTrainDataset(Dataset):
    def __init__(self, annotations_file, vid_dir, transform=models.load_transform(), target_transform=None):
        self.vid_labels = pd.read_csv(annotations_file)
        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.vid_labels)

    def __getitem__(self, idx):
        vid_path = os.path.join(self.vid_dir, self.vid_labels.iloc[idx, 0])

        #image = read_image(vid_path)
        num_segments = 8 #may need to adjust

        frames = extract_frames(vid_path, num_segments)


        #frames = extract_frames("videos/label_videos/excavating/excavating_2.mp4")
        #video = torch.stack([self.transform(frame) for frame in frames], 1).unsqueeze(0)
        video = torch.stack([self.transform(frame) for frame in frames], 1)

        label = self.vid_labels.iloc[idx, 1]

        #if self.transform:
        #    image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            #hotfix
            label = label.to(dtype=torch.long)
        return video, label

class CustomImageValDataset(Dataset):
    def __init__(self, annotations_file, vid_dir, transform=models.load_transform(), target_transform=None):
        self.vid_labels = pd.read_csv(annotations_file)
        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.vid_labels)

    def __getitem__(self, idx):
        vid_path = os.path.join(self.vid_dir, self.vid_labels.iloc[idx, 0])

        #image = read_image(vid_path)
        num_segments = 8 #may need to adjust
        frames = extract_frames(vid_path, num_segments)
        #video = torch.stack([self.transform(frame) for frame in frames], 1).unsqueeze(0)
        video = torch.stack([self.transform(frame) for frame in frames], 1)
        label = self.vid_labels.iloc[idx, 1]
        #if self.transform:
        #    image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            # hotfix
            label = label.to(dtype=torch.long)
        return video, label





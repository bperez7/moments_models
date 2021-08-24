import torch
from torch.utils.data import Dataset

from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
from torchvision.transforms import ToTensor

#import matplotlib.pyplot as plt

#augmentors
import vidaug.augmentors as va

import os
import numpy as np
import pandas as pd
#from torchvision.io import read_image
import models
from utils import extract_frames, load_frames, render_frames

class MachineTotalDataset(Dataset):
    def __init__(self, annotations_file, vid_dir, transform=models.load_transform(), target_transform=None):
        self.vid_labels = pd.read_csv(annotations_file)
        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.vid_labels)

    def __getitem__(self, vid_id):
        print('getting item')
        print(vid_id)


       # vid_path = "videos/label_videos/bulldozing/bulldozing_2.mp4"
        #vid_path = os.path.join(self.vid_dir, self.vid_labels.iloc[idx, 0])
        vid_path = os.path.join(self.vid_dir, vid_id)
        self.vid_labels
        label = np.array(self.vid_labels[self.vid_labels['image_id']] == vid_id['label'].values[0])

        #image = read_image(vid_path)
        num_segments = 8 #may need to adjust

        frames = extract_frames(vid_path, num_segments)


        #frames = extract_frames("videos/label_videos/excavating/excavating_2.mp4")
        #video = torch.stack([self.transform(frame) for frame in frames], 1).unsqueeze(0)
        video = torch.stack([self.transform(frame) for frame in frames], 1)

       # label = self.vid_labels.iloc[idx, 1]

        #if self.transform:
        #    image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            #hotfix
            label = label.to(dtype=torch.long)
        return video, label

class VideoSampler(Sampler):
    def __init__(self,
                 sample_idx,
                 data_source='dataset/all_labels.csv'):
        super().__init__(data_source)
        self.sample_idx = sample_idx
        self.df_images = pd.read_csv(data_source)

    def __iter__(self):
        vid_ids = self.df_images['video_id'].loc[self.sample_idx]
        #ids = self.df_images.iloc[:,0].loc[self.sample_idx]
        return iter(vid_ids)

    def __len__(self):
        return len(self.sample_idx)


class VideoBatchSampler(BatchSampler):
    def __init__(self,
                 sampler,
                 aug_count=5,
                 batch_size=30,
                 drop_last=True):
        super().__init__(sampler, batch_size, drop_last)
        self.aug_count = aug_count
        assert self.batch_size % self.aug_count == 0, 'Batch size must be an integer multiple of the aug_count.'

    def __iter__(self):
        batch = []

        for image_id in self.sampler:
            print(image_id)
            for i in range(self.aug_count):
                batch.append(image_id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def create_split_loaders(dataset, split, aug_count, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = VideoSampler(train_folds_idx)
    valid_sampler = VideoSampler(valid_folds_idx)
    train_batch_sampler = VideoBatchSampler(train_sampler,
                                            aug_count,
                                            batch_size)
    valid_batch_sampler = VideoBatchSampler(valid_sampler,
                                            aug_count=1,
                                            batch_size=batch_size,
                                            drop_last=False)
    train_loader = DataLoader(dataset,batch_sampler=train_batch_sampler)
    valid_loader = DataLoader(dataset, batch_sampler=valid_batch_sampler)
    return (train_loader, valid_loader)


def get_all_split_loaders(dataset, cv_splits, aug_count=5, batch_size=30):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to
                 be used in each fold for each split.
    aug_count -- Number of variations for each sample in dataset.
    batch_size -- batch size.

    """
    split_samplers = []

    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset,
                                 cv_splits[i],
                                 aug_count,
                                 batch_size)
        )
    return split_samplers

# splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
#
#
# df_train = pd.read_csv('dataset/all_labels.csv')
#
# splits = []
# for train_idx, test_idx in splitter.split(df_train['video_id'], df_train['label']):
#     splits.append((train_idx, test_idx))
# print(splits)
#
# dataset = MachineTotalDataset('dataset/all_labels.csv', vid_dir='videos/label_videos')
#
#
# dataloaders = get_all_split_loaders(dataset, splits, aug_count=1, batch_size=10)



class MachineTotalAugmentedDataset(Dataset):
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





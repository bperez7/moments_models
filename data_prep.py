import models
from utils import extract_frames, load_frames, render_frames
from os import listdir
from os.path import isfile, join
import torch
import numpy as np


num_segments = 8
videos_path = "./training_videos"

print('Extracting frames using ffmpeg...')

transform = models.load_transform()

training_x_list = []
training_y_list = []

video_files = [f for f in listdir(videos_path) if isfile(join(videos_path, f))]
for video_file in video_files:
    frames = extract_frames(videos_path+"/"+video_file, num_segments)
    video_input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
    training_x_list.append(video_input)
    training_y_list.append(306)



















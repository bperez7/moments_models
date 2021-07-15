import cv2
import os
import time
import subprocess
import pytest

from localization_tool import VideoCropTool


def test_create_crop_tool_1():
    video_file_path = 'videos/IMG_4884.MOV'
    video_start_time = 0  # in secs
    fps = 30

    video_start_frame =video_start_time*fps
    cap = cv2.VideoCapture(video_file_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)
    #should not raise any exception



def test_crop_predict_video_4887():
    #draw the bounding box and indicate start and end times. Check the following
    #1. The trimmed video file is saved in the indicated output folder
    #2. Top 5 predictions are given in the terminal

    video_file_path = 'videos/IMG_4887.MOV'
    output_file = "demo_clip"
    output_folder = "trimmed_videos"
    output_label = "tossing"
    result_text = ""
    video_start_time = 0 # in secs
    fps = 30
    video_start_frame = video_start_time*fps

    cap = cv2.VideoCapture(video_file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    my_crop_tool = VideoCropTool(video_file_path, output_file, output_folder, 0, cap, output_label)
    my_crop_tool.crop_and_predict()


def test_crop_label_video_4884():
    # 1. The trimmed video file is saved in the indicated output folder (which is named after the label
    # 2. Top 5 predictions are given in the terminal

    video_file_path = 'videos/IMG_4884.MOV'
    output_file = "random_1"
    output_folder = "label_videos"
    output_label = "random"
    result_text = ""
    video_start_time = 0 # in secs
    fps = 30
    video_start_frame = video_start_time*fps

    cap = cv2.VideoCapture(video_file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    my_crop_tool = VideoCropTool(video_file_path, output_file, output_folder, 0, cap, output_label)
    my_crop_tool.crop_and_label()




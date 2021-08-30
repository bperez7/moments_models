import cv2
import os
import time
import subprocess
import json
import math

import torchvision.models.detection.faster_rcnn

from utils import extract_frames
#from matplotlib import pyplot as plt
import numpy as np
import torch
import models
from torchvision import transforms
from torch import nn
#from tsm_model import TemporalShift

from matplotlib import cm
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detecto.core import Model as DetectoModel
from detecto import utils as detecto_utils
from detecto import visualize as detecto_visualize

from model_zoo.model_builder import build_model_no_args

from helpers import biggest_overlapping_bbox



#from test_video import get_predictions_results
#cam_capture = cv2.VideoCapture(0)
#cv2.destroyAllWindows()
from torch.nn import functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# TODO:
#   1. Directory hierarchy
#   2. FFMPEG error with labelled videos
#   3. YOLO (try on gpu)
#   4. config file for models, etc
#   5. R2_1D categories are different
#





frame_time = 10

frame_count = 0
global_trim_time = None
crop_started = False

#from imageai.Detection import ObjectDetection
import os
import cv2
from moviepy.editor import *

execution_path = os.getcwd()



proximity_threshold = 300



all_results = []



def proximity_check(threshold, x1, x2, y1,y2):
    if abs(x1-x2)> threshold:
        return False
    if abs(y1-y2)>threshold:
        return False
    return True



class VideoCropTool:


    def __init__(self, output_file, output_folder, video_start_time,
                    capture, multi_label=False,trn_mode=False,tsm_mode=False,
                 subtract_background=False,
                 time_window=3,fps=30,obj_detection_model="fast_rcnn",
                 max_number_of_objects=4, allow_overlapping_bboxes=False,
                 action_recognition_model="resnet3d", bbox_expansion_factor = .3):
        """

        Args:
            video_path:
            output_file:
            output_folder:
            video_start_time:
            capture:
            output_label:
            multi_label:
            time_window_on:
            time_window:
        """
        self.output_folder = output_folder
        self.output_file = output_file
        self.video_start_time = video_start_time
        self.cap = capture
        self.fps = fps


    #    self.video_start_frame = video_start_frame

        self.recording = False


        #result
        self.result_text = ""

        #frame properties
        self.frame_width = 0
        self.frame_height = 0

        self.current_frame_count = 0
        self.frame_buffer = [None for i in range(90)]
        self.subtracted_buffer = [None for i in range(90)]
        self.sampled_buffer = []


        #detected object arrays/limits
        self.detected_objects = [None for i in range(50)]
        self.max_num_of_objects = max_number_of_objects

        #transforms
        self.det_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        #models
        self.action_recognition_model_type = action_recognition_model
        self.obj_detector_model_type = obj_detection_model

        #object detection model
        if self.obj_detector_model_type=="detecto":
            self.detecto_model = DetectoModel()
        elif self.obj_detector_model_type=="fast-rcnn":
            self.fast_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
            self.fast_mobile = None
            self.fast_rcnn.eval()

        #to determine whether bboxes will overlap or not
        self.allow_overlapping_bboxes = allow_overlapping_bboxes

        #decide model and categories
        if self.action_recognition_model_type=="resnet3d":
            self.resnet_3d_model = models.load_model('resnet3d50')
            self.action_recognition_model = self.resnet_3d_model
            self.categories = models.load_categories('category_momentsv2.txt')
        elif self.action_recognition_model_type=="r2_1d":
            pretrained_r2_1d = torch.load('pretrained_models/model_best_r2_1d.pth.tar',map_location=torch.device('cpu') )
            r2_1d_dict = pretrained_r2_1d['state_dict']
            #print(r2_1d_dict)
            # model.load_state_dict(new_dict, strict=False)
            self.r2_1d_model = build_model_no_args(test_mode=True)
            self.r2_1d_model.load_state_dict(r2_1d_dict, strict=False)
            self.action_recognition_model=self.r2_1d_model
            self.categories = models.load_categories('category_momentsv2.txt')
        elif self.action_recognition_model_type=="fine-tune-resnet3d":
            self.fine_tuned_model = torch.load('trained_models/model_aug_08-18-1.pth', map_location=torch.device('cpu'))
            self.fine_tuned_model = nn.DataParallel(self.fine_tuned_model)
            self.action_recognition_model = self.fine_tuned_model
            self.categories = models.load_categories('dataset/machine_categories.txt')

#TODO: fix yolo model(v5, may not be compatible with pytorch 1.6.0)
       # self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
       # self.yolo_model.eval()


        #model type
        self.multi_label=multi_label
        self.trn = trn_mode
        # self.tsm = tsm_mode
        # self.tsm_model = TemporalShift(torch.nn.Sequential(), n_segment=8, n_div=8, inplace=False)
        # self.tsm_model.load_state_dict(torch.load("pretrained_models/"+"TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth",
        #                               map_location=torch.device('cpu')),strict=False)

        # self.tsm_model = torch.load("pretrained_models/"+"TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth",
        #                             map_location=torch.device('cpu'))


        #subtract background
        self.subtract_background = subtract_background
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)


        #initialization of input
        self.recorded_frames = []
        self.cropped_frames = []


        #object detection
        #yolo v3 not compatible with pytorch? only tensorflow
        # self.detector = ObjectDetection()
        # #self.detector.setModelTypeAsTinyYOLOv3()
        # self.detector.setModelTypeAsYOLOv3()
        # self.detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
        # self.detector.loadModel(detection_speed="fast")
        self.detected_bboxes = [None for i in range(20)]
        self.bbox_expansion_factor = bbox_expansion_factor

        self.num_objects = None


        #video


    def detect_from_image(self, frame, obj_output_folder, frame_name):
        out_path = os.path.join(execution_path + "/" + obj_output_folder,frame_name+ "_result.jpg")
        detections, objects_path = self.detector.detectObjectsFromImage(input_type="array",
            input_image= frame,
            output_image_path=out_path,
           # output_image_path=os.path.join(execution_path + "/" + output_folder, name_file[:-4] + "_result.jpg"),
            minimum_percentage_probability=30, extract_detected_objects=True)
        obj_name, obj_prob, obj_points = None, None, None
        frame_results = []
        object_count = 0
        for eachObject, eachObjectPath in zip(detections, objects_path):
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
            print("Object's image saved in " + eachObjectPath)
            print("--------------------------------")
            obj_name = eachObject["name"]
            obj_prob = eachObject["percentage_probability"]
            obj_points = eachObject["box_points"]
            results = obj_name, obj_prob, obj_points, object_count, out_path
            frame_results.append(results)
            object_count += 1

        return frame_results



    def detect_and_predict(self):

        """
        - Plays back the selected video in an opencv frame and allows for cropping/time selection
        - Runs the moments in time model and gives the top 5 predictions for the selected segment in the terminal



        Returns: None


        """


        #video
     #   size = tuple(frames[0].shape)


        frame_count = 0
        while (self.cap.isOpened()):



            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if ret == True:

                # check if subtracting background
              #  if self.subtract_background == True:

                # increment frame count
                self.current_frame_count += 1
                if self.current_frame_count==1:
                    # get vcap property (height and width)
                    self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
                    self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

                    # for generating the background subtracted video
                    mask_array = []
                    mask_size = (self.frame_width, self.frame_height)
                    mask_frames = []

                    video_frame_size = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

                    out = cv2.VideoWriter(self.output_folder + "/" + self.output_file+ '.mp4',
                                          fourcc, float(self.fps), (frame.shape[1], frame.shape[0]))
                elif self.current_frame_count>1:
                    mask = self.subtractor.apply(frame)
                    mask_array.append(mask)
                    extended_mask = np.expand_dims(mask, axis=2)
                    subtracted_frame = np.where(extended_mask == 255, frame, 0)  # filter by mask
                    self.subtracted_buffer = self.subtracted_buffer[1:] + [subtracted_frame]

                self.frame_buffer = self.frame_buffer[1:]+[frame]
                #for background subtracted buffer

                # every 90 frames apply object detection (need sliding frame buffer)

                #correct by -1 since starts at index 1

                """Continue HERE!!!!"""
                if ((self.current_frame_count-1) % 90 == 0) and self.current_frame_count!=1:

                  #  print('buffer shift time: ' + str(buffer_shift_diff))

                    #adjust for background subtraction
                    if self.subtract_background:
                        obj_detection_time_start = time.time()
                        obj_pil_frame = Image.fromarray(subtracted_frame, 'RGB')
                    else:
                        obj_detection_time_start = time.time()
                        obj_pil_frame = Image.fromarray(frame, 'RGB')


                  #  obj_detect_result = self.detect_from_image(frame=frame, obj_output_folder="obj_detect_out",frame_name="obj_test")

                   # frame_tensor = torch.from_numpy(frame)
                    #print(frame_tensor)
                    transformed_frame = self.det_transform(obj_pil_frame)
                    #print(transformed_frame.size())
                    if self.obj_detector_model_type=="detecto":
                       # obj_detect_result = self.detecto_model(frame)
                        obj_detect_result = self.detecto_model.predict(frame)
                    else:
                        obj_detect_result = self.fast_rcnn([self.det_transform(obj_pil_frame)])
                    #obj_detect_result = self.yolo_model([self.det_transform(obj_pil_frame)])
                  #  print(obj_detect_result)
                    obj_detection_time_end = time.time()
                    obj_detection_diff = obj_detection_time_start-obj_detection_time_end
                    print('obj detection time: ' + str(obj_detection_diff))

                    obj_count = 0
                    #obj_length = obj_detect_result[0]['boxes']
                    #print(obj_length)
                    if self.obj_detector_model_type=="detecto":
                        bboxes = obj_detect_result[1]
                    else:
                        bboxes = obj_detect_result[0]['bboxes']


                  #  print(len(bboxes))
                    self.num_objects = min(len(bboxes), self.max_num_of_objects)
                    self.detected_objects[:self.num_objects]=bboxes[:self.num_objects]
                  #  print(self.detected_objects)
                    if not self.allow_overlapping_bboxes:
                        dominant_box_indices = biggest_overlapping_bbox(self.detected_objects[:self.num_objects])

                        dominant_boxes = []

                        for i in range(self.num_objects):
                            if i in dominant_box_indices:
                                dominant_boxes.append(bboxes[i])

                        self.detected_objects[:len(dominant_boxes)] = dominant_boxes
                        self.num_objects = len(dominant_boxes)


                    for i in range(self.num_objects):
                        obj_count+=1
                       # bbox = result[2]
                        if self.obj_detector_model_type=="detecto":
                            #bbox = bboxes[i]
                            bbox = self.detected_objects[i] #dominanat boxes
                        else:
                            bbox = obj_detect_result[0]['boxes'][i]


                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        detected_obj_array = frame[x1:x2,y1:y2]

                        #30% default area expansion factor (multiply length of sides by 1.14, account for edges
                        length_expansion_factor = math.sqrt(1+self.bbox_expansion_factor)
                        x_length = int((x2-x1)*(length_expansion_factor-1)) #(for example 1.14-1 = .14)
                        y_length =int((y2-y1)*(length_expansion_factor-1))

                        x1 = x1-x_length//2
                        y1 = y1-y_length//2
                        x2 = x2+x_length//2
                        y2 = y2+y_length//2

                        #boundary cases

                        if x1 < 0:
                            x1 = 0
                        if x2 > video_frame_size[1]: #greater than max x coord
                            x2 = video_frame_size[1]-1
                        if y1 < 0:
                            y1=0
                        if y2 > video_frame_size[0]: #greater than max y coord
                            y2 = video_frame_size[0]-1




                        # evenly samples frames in buffer (8 frames)
                        #TODO: how to skip frames

                        self.sampled_buffer = [self.frame_buffer[i] for i in range(90) if i%10==0]

                        #check subtract background
                        if self.subtract_background: #run model on subtracted frames
                            detected_obj_array_sequence = [f[int(y1):int(y2), int(x1):int(x2), :] for f in
                                                           self.subtracted_buffer]
                        else:
                            detected_obj_array_sequence = [f[int(y1):int(y2),int(x1):int(x2),:] for f in self.sampled_buffer]

                        obj_pil_frames = [Image.fromarray(frame, 'RGB') for frame in detected_obj_array_sequence]

                        transform = models.load_transform()
                        res_obj_input = torch.stack([transform(frame) for frame in obj_pil_frames], 1).unsqueeze(0)
                        # res_input = torch.stack([transform(frame) for frame in self.cropped_frames], 1).unsqueeze(0)

                        with torch.no_grad():
                            print('start prediction')

                            logits = self.action_recognition_model(res_obj_input.to(device="cpu"))

                            print('end prediction')
                            h_x = F.softmax(logits, 1).mean(dim=0)
                            probs, idx = h_x.sort(0, True)
                        for i in range(0, min(len(self.categories),5)): #top 5 predictions or just top k predictions
                            if i == 0:
                                # print(round(float(probs[i]),3))
                                # print(self.categories)
                                # print(self.categories[idx[i]])
                                bbox_label = str(round(float(probs[i]),3)) + ", " + str(self.categories[idx[i]])
                                self.detected_bboxes[obj_count-1] = ([x1,y1,x2,y2], bbox_label)
                            print('{:.3f} -> {}'.format(probs[i], self.categories[idx[i]]))
                            next_result = '{:.3f} -> {}'.format(probs[i], self.categories[idx[i]])
                    self.detected_bboxes[obj_count:] = [None for i in range(obj_count,20)]







                #draw bboxes
                for bounding_box in self.detected_bboxes[:self.num_objects]:
                    if bounding_box!=None:
                        bbox_start = bounding_box[0][0:2]
                        bbox_end = bounding_box[0][2:]

                        # also for subtracted frame
                        cv2.rectangle(frame,bbox_start , bbox_end, thickness=3, color=222)
                        action_label = bounding_box[1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        label_origin = (bbox_start[0], bbox_start[1]+40)
                        text_thickness = 1
                        font_scale = 1.2
                        cv2.putText(frame, action_label, label_origin , font,
                                    text_thickness, (0,0,255), text_thickness, cv2.LINE_AA)

                        # subtracted frame
                        if self.subtract_background:
                            cv2.rectangle(subtracted_frame, bbox_start, bbox_end, thickness=3, color=222)
                            cv2.putText(subtracted_frame, action_label, label_origin, font,
                                        font_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)


                if self.subtract_background:
                    out.write(subtracted_frame)
                else:
                    out.write(frame)
               # cv2.imshow('Frame', frame)


                # Press Q on keyboard to  exit
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break


        out.release()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Defines tool parameters and runs the detection and recognition tool"""
    TIME_WINDOW = 3  # seconds
    config_file = open('./configs/config_cuboid.json')
    config = json.load(config_file)

    file_paths = config["file_paths"]
    video_file_path = file_paths["video_file_path"]
    output_file = file_paths["output_file"]
    output_folder = file_paths["output_folder"]


    modes = config["modes"]
    TRN_mode = modes["trn"]
    multi_mode = modes["multi_label"]
    tsm_mode = modes["tsm_mode"]

    parameters = config["parameters"]

    video_start_time = parameters["video_start_time"]

    obj_detection_model = config["modes"]["obj_detection_model"]
    bbox_expansion_factor = config["parameters"]["bbox_expansion_factor"]

    max_number_of_objects = parameters["max_number_of_objects"]
    allow_overlapping_bboxes = parameters["allow_overlapping_bboxes"]

    subtract_background = modes["background_subtraction"]
    action_recognition_model = modes["action_recognition_model"]


    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_start_frame = video_start_time * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    my_crop_tool = VideoCropTool(output_file, output_folder, 0, cap,
                                 trn_mode=TRN_mode,multi_label=multi_mode,tsm_mode=tsm_mode,
                                 subtract_background=subtract_background, fps=fps, obj_detection_model=obj_detection_model,
                                 max_number_of_objects=max_number_of_objects,
                                 allow_overlapping_bboxes=allow_overlapping_bboxes,
                                 action_recognition_model=action_recognition_model)
    my_crop_tool.detect_and_predict()


if __name__=="__main__":


   main()






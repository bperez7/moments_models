import cv2
import os
import time
import subprocess
import json
from utils import extract_frames
#from matplotlib import pyplot as plt
import numpy as np
import torch
import models
from tsm_model import TemporalShift

from matplotlib import cm
from PIL import Image


#from test_video import get_predictions_results
#cam_capture = cv2.VideoCapture(0)
#cv2.destroyAllWindows()
from torch.nn import functional as F

""" TODO: 
1. Currently only able to crop up to a minute 
2. Just count frames instead of time string parsing
3. Directory hierarchy
4. FFMPEG error with labelled videos
5. Clear cache when labeling videos? 

"""




frame_time = 10

frame_count = 0
global_trim_time = None
crop_started = False



class VideoCropTool:


    def __init__(self, video_path, output_file, output_folder, video_start_time,
                    capture, output_label, multi_label=False,trn_mode=False,tsm_mode=False,subtract_background=False,
                 time_window_on = False,time_window=3):
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
        self.video_path = video_path
        self.output_file = output_file
        self.output_folder = output_folder
        self.output_label=output_label
        self.video_start_time = video_start_time
        self.cap = capture
    #    self.video_start_frame = video_start_frame

        #for clikc box

        #self.start = (0,0)
        self.box_started = False
        self.box_created = False
        self.box_finished = False
        self.start = None
        self.end = None


        #for cropping time
        self.global_trim_time = None
        self.global_trim_time_secs = None
        self.crop_started = False
        self.start_trim_time = None
        self.end_trim_time = None
        self.start_trim_time_secs = None
        self.end_trim_time_secs = None
        self.time_window = time_window

        self.time_crop_secs = 0


        self.recording = False


        #result
        self.result_text = ""


        #frame properties
        self.frame_width = 0
        self.frame_height = 0


        #model type
        self.multi_label=multi_label
        self.trn = trn_mode
        self.tsm = tsm_mode
        self.tsm_model = TemporalShift(torch.nn.Sequential(), n_segment=8, n_div=8, inplace=False)
        self.tsm_model.load_state_dict(torch.load("pretrained_models/"+"TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth",
                                      map_location=torch.device('cpu')),strict=False)

        # self.tsm_model = torch.load("pretrained_models/"+"TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth",
        #                             map_location=torch.device('cpu'))


        #subtract background
        self.subtract_background = subtract_background
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)



    def click_box(self,event, x,y, flags, param):

        """
        Detects and processes left and right clicks of the mouse on the opencv frame

        Args:
            event:
            x:
            y:
            flags:
            param:

        Returns: None


        """
        #Start drawing the box if the left button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:

            self.start = (x, y)

            self.box_started = True

        #Drag the box if the mouse is moving
        elif event == cv2.EVENT_MOUSEMOVE:

            self.end = (x, y)

        #Finalize the box if the left button is raised
        elif event == cv2.EVENT_LBUTTONUP:

            # global box_created
            self.final_end = (x, y)
            self.box_created = True

        elif event == cv2.EVENT_RBUTTONDOWN:

            # cropping time starts
       #     global crop_started
            if self.crop_started != True:
                self.crop_started = True
                self.start_trim_time = self.global_trim_time
                self.start_trim_time_secs = self.global_trim_time_secs
                self.recording = True

            else:
                self.crop_started = False
                self.trim_end_time = self.global_trim_time
                #self.box_created = True
                self.box_finished = True
                self.end_trim_time = self.global_trim_time
                self.end_trim_time_secs = self.global_trim_time_secs
                self.time_crop_secs = self.end_trim_time_secs-self.start_trim_time_secs
                print('crop time')
                print(self.time_crop_secs)
                self.recording = False






    def crop_and_label(self):
        """
        - Plays back the selected video in an opencv frame and allows for cropping/time selection
        - Sorts the cropped video into a folder named after the given label



        Returns: None


        """

        while (self.cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = self.cap.read()
            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame", self.click_box)

            # get vcap property (height and width)
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            # global frame_count
            # frame_count += 1

            # r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)

            if ret == True:


                if self.box_started:


                    rectangle_thickness=30
                    if self.box_created:
                        cv2.rectangle(frame, self.start, self.final_end, thickness=rectangle_thickness,color=333)



                    else:
                        cv2.rectangle(frame, self.start, self.end,thickness=rectangle_thickness, color=333)


                        # except:
                        #     cv2.rectangle(frame, self.start, self.end, color=333)


                # Display the resulting frame
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time_in_secs = round(current_time / 1000)
                self.global_trim_time_secs = current_time_in_secs

                current_time_secs = current_time_in_secs % 60
                current_time_mins = current_time_in_secs // 60

                prev_time_in_secs = current_time_in_secs - self.time_window
                prev_time_secs = prev_time_in_secs % 60
                prev_time_mins = prev_time_in_secs // 60

                if (current_time_mins // 10 == 0):  # single digit
                    current_time_mins_str = "0" + str(current_time_mins)

                else:
                    current_time_mins_str = str(current_time_mins)
                if (current_time_secs // 10 == 0):  # single digit
                    current_time_secs_str = "0" + str(current_time_secs)
                else:
                    current_time_secs_str = str(current_time_secs)

                if (prev_time_mins // 10 == 0):  # single digit
                    prev_time_mins_str = "0" + str(prev_time_mins)
                else:
                    prev_time_mins_str = str(prev_time_mins)
                if (prev_time_secs // 10 == 0):  # single digit
                    prev_time_secs_str = "0" + str(prev_time_secs)
                else:
                    prev_time_secs_str = str(prev_time_secs)

               # if (self.time_window ):  # single digit
                if (self.time_crop_secs<10):
                    #TIME_WINDOW_STR = "0" + str(self.time_window)
                    TIME_WINDOW_STR = "00:00:"+"0" + str(self.time_crop_secs)

                else:
                    TIME_WINDOW_STR = "00:00:"+str(self.time_crop_secs)

                end_time = "00:" + current_time_mins_str + ":" + current_time_secs_str

               # global global_trim_time
                self.global_trim_time = end_time

                start_time = "00:" + prev_time_mins_str + ":" + prev_time_secs_str



                # cut_time = "00:00:"+TIME_WINDOW_STR

                text = str(round(current_time, 2))
                #  try:

                # result_text = get_predictions_results()

                # except:

                org = (50, 50)
                result_origin = (50, 200)
                color = (255, 0, 0)
                thickness = 2
                fontScale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, self.result_text, result_origin, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                #Red dot while cropping
                if self.recording:

                    # Radius of circle
                    radius = 20
                    # Center coordinates
                    circle_center_coordinates = (int(self.frame_width) - radius - 20, 50)

                    # Red color in BGR
                    circle_color = (0, 0, 255)

                    # Line thickness of -1 px
                    circle_thickness = -1

                    # Using cv2.circle() method
                    # Draw a circle of red color of thickness -1 px
                    image = cv2.circle(frame, circle_center_coordinates, radius, circle_color, circle_thickness)



                cv2.imshow('Frame', frame)





                if self.box_finished:
                        left_arg = "-l " + str(self.start[0]) + " "
                        top_arg = "-t " + str(self.start[1]) + " "
                        width_arg = "-w " + str(self.final_end[0] - self.start[0]) + " "
                        height_arg = "-h " + str(self.final_end[1] -self.start[1]) + " "
                        video_arg = "-f " + self.video_path + " "
                        output_arg = "-o " + self.output_folder + "/" + self.output_label + "/" + self.output_file + " "
                        beginning_arg = "-b " + str(self.start_trim_time_secs) + " "
                        end_arg = "-e " + TIME_WINDOW_STR

                        # print("beginning and end ")
                        # print(beginning_arg)
                        # print(end_arg)
                        crop_time_start = time.time()

                        if not os.path.exists(self.output_folder+"/"+self.output_label):
                            os.makedirs(self.output_folder+"/"+self.output_label)


                        command = "bash " + "bash_scripts/crop_tool.sh " + video_arg + left_arg + top_arg + width_arg + height_arg + output_arg + beginning_arg + end_arg
                        os.chmod("./bash_scripts/output_command.sh", 0o755)

                        with open("bash_scripts/output_command.sh", "w") as text_file:
                            text_file.write('#!/bin/bash')
                            text_file.write("\n")
                            text_file.write(command + "\n")
                            text_file.write('#hello')

                        os.chmod("./bash_scripts/output_command.sh", 0o755)
                        subprocess.check_call(["./bash_scripts/output_command.sh"])

                        crop_time_end = time.time()

                        crop_elapsed_time = crop_time_end - crop_time_start

                        print("Crop Time: " + str(crop_elapsed_time))

                        # video_model_command = "python test_video.py --draw_crop_test.mp4 --arch resnet3d50"


                        # reset
                        self.box_created = False
                        self.box_started = False
                        self.box_finished = False

                        with open("custom_labels.txt", "a+") as text_file:
                           # all_labels = text_file.read()
                           label_exists = False
                           # print('all labels')
                           # print(all_labels)
                           for line in text_file:
                               if line==self.output_label:
                                   label_exists=True
                                   break

                           if not label_exists:
                               text_file.write("\n")
                               text_file.write(self.output_label)
                               print(self.output_label)







                # Press Q on keyboard to  exit
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def crop_and_predict(self):

        """
        - Plays back the selected video in an opencv frame and allows for cropping/time selection
        - Runs the moments in time model and gives the top 5 predictions for the selected segment in the terminal



        Returns: None


        """
        #for generating the background subtracted video
        mask_array = []
        mask_size = (self.frame_width, self.frame_height)
        mask_frames = []

        while (self.cap.isOpened()):



            # Capture frame-by-frame
            ret, frame = self.cap.read()


            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame", self.click_box)

            # get vcap property (height and width)
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            # global frame_count
            # frame_count += 1

            # r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)

            if ret == True:


                if self.box_started:
                        # print('boxes')
                        # print(self.start)
                        # print(self.end)

                    rectangle_thickness = 10
                    if self.box_created:
                        cv2.rectangle(frame, self.start, self.final_end, thickness=rectangle_thickness, color=333)



                    else:
                        cv2.rectangle(frame, self.start, self.end, thickness=rectangle_thickness, color=333)
                        # except:
                        #     cv2.rectangle(frame, self.start, self.end, color=333)






                # Display the resulting frame
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time_in_secs = round(current_time / 1000)
                current_time_secs = current_time_in_secs % 60
                current_time_mins = current_time_in_secs // 60
                self.global_trim_time_secs = current_time_in_secs

                prev_time_in_secs = current_time_in_secs - self.time_window
                prev_time_secs = prev_time_in_secs % 60
                prev_time_mins = prev_time_in_secs // 60

                if (current_time_mins // 10 == 0):  # single digit
                    current_time_mins_str = "0" + str(current_time_mins)
                else:
                    current_time_mins_str = str(current_time_mins)
                if (current_time_secs // 10 == 0):  # single digit
                    current_time_secs_str = "0" + str(current_time_secs)
                else:
                    current_time_secs_str = str(current_time_secs)

                if (prev_time_mins // 10 == 0):  # single digit
                    prev_time_mins_str = "0" + str(prev_time_mins)
                else:
                    prev_time_mins_str = str(prev_time_mins)
                if (prev_time_secs // 10 == 0):  # single digit
                    prev_time_secs_str = "0" + str(prev_time_secs)
                else:
                    prev_time_secs_str = str(prev_time_secs)

                #if (self.time_window // 10 == 0 and self.time_window!=10):  # single digit

                if (self.time_crop_secs < 10):
                    TIME_WINDOW_STR = "00:00:"+"0" + str(self.time_crop_secs)
                else:
                    TIME_WINDOW_STR = "00:00:"+str(self.time_crop_secs)

                end_time = "00:" + current_time_mins_str + ":" + current_time_secs_str

               # global global_trim_time
                self.global_trim_time = end_time

                start_time = "00:" + prev_time_mins_str + ":" + prev_time_secs_str

                # cut_time = "00:00:"+TIME_WINDOW_STR

                text = str(round(current_time, 2))
                #  try:

                # result_text = get_predictions_results()
                # print(result_text)
                # except:

                org = (50, 50)
                result_origin = (50, 200)
                color = (255, 0, 0)
                thickness = 2
                fontScale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, self.result_text, result_origin, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                # Red dot while cropping
                if self.recording:
                    #print('recording')


                    # Radius of circle
                    radius = 20
                    # Center coordinates
                    circle_center_coordinates = (int(self.frame_width) - radius - 20, 50)

                    # Red color in BGR
                    circle_color = (0, 0, 255)

                    # Line thickness of -1 px
                    circle_thickness = -1

                    # Using cv2.circle() method
                    # Draw a circle of red color of thickness -1 px
                    cv2.circle(frame, circle_center_coordinates, radius, circle_color, circle_thickness)


                #check if subtracting background
                if self.subtract_background==True:
                    mask = self.subtractor.apply(frame)
                    mask_array.append(mask)

                cv2.imshow('Frame', frame)




                if self.box_finished:
                        left_arg = "-l " + str(self.start[0]) + " "
                        top_arg = "-t " + str(self.start[1]) + " "
                        width_arg = "-w " + str(self.final_end[0] - self.start[0]) + " "
                        height_arg = "-h " + str(self.final_end[1] -self.start[1]) + " "
                        video_arg = "-f " + self.video_path + " "
                        output_arg = "-o " + self.output_folder + "/" + self.output_file + " "
                        beginning_arg = "-b " + str(self.start_trim_time_secs)+ " "
                        end_arg = "-e " + TIME_WINDOW_STR
                        #
                        print("beginning and end ")
                        print(beginning_arg)
                        print(end_arg)
                        crop_time_start = time.time()

                        command = "bash " + "bash_scripts/crop_tool.sh " + video_arg + left_arg + top_arg + width_arg + height_arg + output_arg + beginning_arg + end_arg
                        os.chmod("./bash_scripts/output_command.sh", 0o755)

                        with open("./bash_scripts/output_command.sh", "w") as text_file:
                            text_file.write('#!/bin/bash')
                            text_file.write("\n")
                            text_file.write(command + "\n")
                            text_file.write('#hello')

                        os.chmod("./bash_scripts/output_command.sh", 0o755)
                        subprocess.check_call(["./bash_scripts/output_command.sh"])

                        crop_time_end = time.time()

                        crop_elapsed_time = crop_time_end - crop_time_start

                        print("Crop Time: " + str(crop_elapsed_time))

                        # video_model_command = "python test_video.py --draw_crop_test.mp4 --arch resnet3d50"

                        prediction_time_start = time.time()

                        if self.trn:
                           # subprocess.call(["python", "test_video.py", "--video_file " + self.output_folder + "/" + self.output_file + ".mp4 "+
                             #         "--arch BNInception " +"--dataset something " +"--weights pretrain/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"], cwd="TRN-pytorch")
                            os.chdir("./TRN-pytorch")
                            os.system("python test_video.py --video_file " + "../"+ self.output_folder + "/" + self.output_file + ".mp4 "
                                     + "--arch BNInception --dataset something  --weights pretrain/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")

                        if self.multi_label and not self.trn:
                            os.system("python test_video.py --video_file " + self.output_folder + "/" + self.output_file + ".mp4 " + "--arch resnet3d50" + " --multi")

                        elif not self.trn:
                            os.system("python test_video.py --video_file " + self.output_folder + "/" + self.output_file + ".mp4 " + "--arch resnet3d50")
                        if self.tsm:
                            print('tsm results')
                            tsm_frames = extract_frames(self.output_folder + "/" + self.output_file + ".mp4", 8)
                            transform = models.load_transform()
                            tsm_input = torch.stack([transform(frame) for frame in tsm_frames], 1)
                            print(self.tsm_model.net)
                            tsm_predictions = self.tsm_model(tsm_input)
                            print(tsm_predictions)



                        os.chdir("/Users/brandonperez/Documents/GitHub/moments_crop/moments_models")



                        prediction_time_end = time.time()

                        prediction_elapsed_time = prediction_time_end - prediction_time_start
                        print("Prediction Time: " + str(prediction_elapsed_time))
                        # Opening prediction file
                        file1 = open('predictions.txt', 'r')
                        result_text = ""
                        for line in file1:
                            print(line)
                            result_text += line
                            break  # just first prediction
                            # result_text += "\n"

                        # reset
                        self.box_created = False
                        self.box_started = False
                        self.box_finished = False


                # Press Q on keyboard to  exit
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break





        # self.cap.release()
        # cv2.destroyAllWindows()
        # #generate mask video
        #
        # #mask_array = np.array(mask_array)
        # transform = models.load_transform()
        # pil_mask_frames = [Image.fromarray(np.uint8(cm.gist_earth(frame)*255)) for frame in mask_array]
        # #original frames
        # print(self.output_folder + "/" + self.output_file+".mp4")
        # original_frames = extract_frames(self.output_folder + "/" + self.output_file+".mp4", 8)
        # transformed_frames = torch.stack([transform(frame) for frame in original_frames]).unsqueeze(0)
        # print('pillow size')
        #
        # print('size of original')
        # print(transformed_frames.size())
        # for frame in pil_mask_frames:
        #     #print(frame.size)
        #     pass
        # mask_video = torch.stack([transform(frame) for frame in pil_mask_frames]).unsqueeze(0)
        # print('mask size')
        # print(mask_video.size())


        #load model
        model = models.load_model('resnet3d50')


        # Make video prediction
        # with torch.no_grad():
        #     logits = model(mask_video)
        #     h_x = F.softmax(logits, 1).mean(dim=0)
        #     probs, idx = h_x.sort(0, True)

        # Output the prediction.
        #video_name = args.frame_folder if args.frame_folder is not None else args.video_file
        # print('MASK RESULT')
        # predictions_results=""
        # categories = models.load_categories('category_momentsv2.txt')
        # for i in range(0, 5):
        #     print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
        #     next_result = '{:.3f} -> {}'.format(probs[i], categories[idx[i]])
        #
        #     predictions_results += next_result
        #     predictions_results += '\n'


        # mask_out_vid = cv2.VideoWriter('videos/mask_project.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, mask_size)
        # for i in range(len(mask_array)):
        #     mask_out_vid.write(mask_array[i])
        # mask_out_vid.release()


def main():
    """Defines tool parameters and runs the localization tool"""
    TIME_WINDOW = 3  # seconds
    config_file = open('./configs/config_localization.json')
    config = json.load(config_file)

    file_paths = config["file_paths"]
    video_file_path = file_paths["video_file_path"]
    output_file = file_paths["output_file"]
    output_folder = file_paths["output_folder"]
    output_label = file_paths["output_label"]

    modes = config["modes"]
    TRN_mode = modes["trn"]
    multi_mode = modes["multi_label"]
    tsm_mode = modes["tsm_mode"]

    parameters = config["parameters"]
    video_start_time = parameters["video_start_time"]


    #video_file_path = 'videos/yt8m_video_library/constructions/group_10/hJkY.mp4'

   # output_folder = "label_videos"



    multi_label = False

    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_start_frame = video_start_time * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    my_crop_tool = VideoCropTool(video_file_path, output_file, output_folder, 0, cap, output_label,
                                 trn_mode=TRN_mode,multi_label=multi_mode,tsm_mode=tsm_mode,
                                 subtract_background=False)
   # my_crop_tool.crop_and_predict()
    my_crop_tool.crop_and_label()

if __name__=="__main__":


   main()






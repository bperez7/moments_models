import cv2
import os
import time
import subprocess
#from matplotlib import pyplot as plt
import numpy as np

#from test_video import get_predictions_results
#cam_capture = cv2.VideoCapture(0)
#cv2.destroyAllWindows()

""" TODO:
1. Start video at specified time
2. Right click to indicate trimming points
3. Output file name

"""

TIME_WINDOW = 3 #seconds

video_file_path = 'videos/IMG_4884.MOV'
output_file = "custom_construction"
output_folder = "trimmed_videos"
result_text = ""
cap = cv2.VideoCapture(video_file_path)
frame_time = 10
video_start_time = None
frame_count = 0
global_trim_time = None
crop_started = False




# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")





def click_box(event, x,y, flags, param):
    print('clicked')
    global start, box_started, end, box_created, final_end

    if event==cv2.EVENT_LBUTTONDOWN:



        start = (x, y)
       # global box_started
        box_started = True
    elif event==cv2.EVENT_MOUSEMOVE:


        end = (x, y)
    elif event==cv2.EVENT_LBUTTONUP:

       # global box_created
        final_end = (x, y)
        box_created = True

    elif event==cv2.EVENT_RBUTTONDOWN:

        #cropping time starts
        global crop_started
        if crop_started!=True:
            crop_started = True
            trim_start_time = global_trim_time
        else:
            crop_started = False
            trim_end_time = global_trim_time










while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  cv2.namedWindow("Frame")
  cv2.setMouseCallback("Frame", click_box)
  global frame_count
  frame_count+=1

  #r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)

  if ret == True:

    try:
        if box_started:
            print('boxes')
            print(start)
            print(end)

            try:
                if box_created:
                    cv2.rectangle(frame, start, final_end, color=333)



                else:
                    cv2.rectangle(frame, start, end, color=333)


            except:
                cv2.rectangle(frame, start, end, color=333)





    except:
        pass
    # Display the resulting frame
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_time_in_secs = round(current_time / 1000)
    current_time_secs = current_time_in_secs % 60
    current_time_mins = current_time_in_secs // 60

    prev_time_in_secs = current_time_in_secs-TIME_WINDOW
    prev_time_secs = prev_time_in_secs%60
    prev_time_mins = prev_time_in_secs//60

    if (current_time_mins//10==0):#single digit
        current_time_mins_str = "0"+str(current_time_mins)
    else:
        current_time_mins_str =  str(current_time_mins)
    if (current_time_secs//10==0):#single digit
        current_time_secs_str = "0" + str(current_time_secs)
    else:
        current_time_secs_str = str(current_time_secs)

    if (prev_time_mins//10==0):#single digit
        prev_time_mins_str = "0"+str(prev_time_mins)
    else:
        prev_time_mins_str =  str(prev_time_mins)
    if (prev_time_secs//10==0):#single digit
        prev_time_secs_str = "0" + str(prev_time_secs)
    else:
        prev_time_secs_str = str(prev_time_secs)

    if (TIME_WINDOW//10==0):#single digit
        TIME_WINDOW_STR = "0"+str(TIME_WINDOW)
    else:
        TIME_WINDOW_STR =  str(TIME_WINDOW)


    end_time = "00:"+current_time_mins_str+":"+current_time_secs_str

    global global_trim_time
    global_trim_time = end_time

    start_time = "00:"+prev_time_mins_str+":"+prev_time_secs_str

   # cut_time = "00:00:"+TIME_WINDOW_STR




    text = str(round(current_time,2))
  #  try:

        #result_text = get_predictions_results()
       # print(result_text)
    #except:

    org = (50,50)
    result_origin = (50,200)
    color = (255,0,0)
    thickness = 2
    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, result_text, result_origin, font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    try:
        if box_created:
            left_arg = "-l " + str(start[0]) + " "
            top_arg = "-t " + str(start[1]) + " "
            width_arg = "-w " + str(final_end[0] - start[0]) + " "
            height_arg = "-h " + str(final_end[1] -start[1]) + " "
            video_arg = "-f " + video_file_path + " "
            output_arg = "-o " + output_folder+"/"+ output_file + " "
            beginning_arg = "-b " + start_time + " "
            end_arg = "-e " + TIME_WINDOW_STR

            print("beginning and end ")
            print(beginning_arg)
            print(end_arg)
            crop_time_start = time.time()


            command = "bash " + "crop_tool.sh " + video_arg + left_arg + top_arg + width_arg + height_arg + output_arg  + beginning_arg + end_arg
            os.chmod("./output_command.sh", 0o755)


            with open("output_command.sh", "w") as text_file:
                text_file.write('#!/bin/bash')
                text_file.write("\n")
                text_file.write(command + "\n")
                text_file.write('#hello')

            os.chmod("./output_command.sh", 0o755)
            subprocess.check_call(["./output_command.sh"])

            crop_time_end = time.time()

            crop_elapsed_time = crop_time_end-crop_time_start

            print("Crop Time: " + str(crop_elapsed_time))

            # video_model_command = "python test_video.py --draw_crop_test.mp4 --arch resnet3d50"


            prediction_time_start = time.time()
            os.system("python test_video.py --video_file " + output_file+".mp4 " + "--arch resnet3d50")

            prediction_time_end = time.time()

            prediction_elapsed_time = prediction_time_end-prediction_time_start
            print("Prediction Time: " + str(prediction_elapsed_time))
            # Opening prediction file
            file1 = open('predictions.txt', 'r')
            result_text = ""
            for line in file1:

                print(line)
                result_text += line
                break #just first prediction
                #result_text += "\n"

            #reset
            box_created = False
            box_started = False
    except:
            pass


    # Press Q on keyboard to  exit
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

cap.release()
cv2.destroyAllWindows()
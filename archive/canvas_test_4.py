import cv2
import os
import subprocess
from matplotlib import pyplot as plt
import numpy as np

#cam_capture = cv2.VideoCapture(0)
#cv2.destroyAllWindows()

video_file_path = 'videos/IMG_4887.MOV'
cap = cv2.VideoCapture(video_file_path)
# box_created = False
# box_started = False
# start = (0,0)
# end = (0,0)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")


def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask



# while True:
# #    # _, im0 = cam_capture.read()
#      _, im0 = cap.read()
# #
# #
#      showCrosshair = False
#      fromCenter = False
#      r = cv2.selectROI(im0, fromCenter, showCrosshair)
#      break
# Read until video is co
#mpleted
def click_box(event, x,y, flags, param):
    print('clicked')
    global start, box_started, end, box_created, final_end

    if event==cv2.EVENT_LBUTTONDOWN:

        print('down')


        start = (x, y)
       # global box_started
        box_started = True
    elif event==cv2.EVENT_MOUSEMOVE:


        end = (x, y)
    elif event==cv2.EVENT_LBUTTONUP:

       # global box_created
        final_end = (x, y)
        box_created = True


while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()

  #r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)
  cv2.namedWindow("Frame")
  cv2.setMouseCallback("Frame", click_box)
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


    cv2.imshow('Frame', frame)

    try:
        if box_created:
            left_arg = "-l " + str(start[0]) + " "
            top_arg = "-t " + str(start[1]) + " "
            width_arg = "-w " + str(final_end[0] - start[0]) + " "
            height_arg = "-h " + str(final_end[1] -start[1]) + " "
            video_arg = "-f " + video_file_path + " "
            output_arg = "-o " + "draw_crop_test"

            command = "bash " + "crop_tool.sh " + video_arg + left_arg + top_arg + width_arg + height_arg + output_arg
            os.chmod("./output_command.sh", 0o755)
            with open("output_command.sh", "w") as text_file:
                text_file.write('#!/bin/bash')
                text_file.write("\n")
                text_file.write(command + "\n")
                text_file.write('#hello')

            os.chmod("./output_command.sh", 0o755)
            subprocess.check_call(["./output_command.sh"])

            # video_model_command = "python test_video.py --draw_crop_test.mp4 --arch resnet3d50"
            os.system("python test_video.py --video_file " + "draw_crop_test.mp4 --arch resnet3d50")



            #reset
            box_created = False
            box_started = False
    except:
            pass


    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break
#
# while True:
#     _, image_frame = cam_capture.read()
#
#     rect_img = image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
#
#     sketcher_rect = rect_img
#     sketcher_rect = sketch_transform(sketcher_rect)
#
#     # Conversion for 3 channels to put back on original image (streaming)
#     sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
#
#     # Replacing the sketched image on Region xof Interest
#     image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = sketcher_rect_rgb
#
#     cv2.imshow("Sketcher ROI", image_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam_capture.release()
cap.release()
cv2.destroyAllWindows()
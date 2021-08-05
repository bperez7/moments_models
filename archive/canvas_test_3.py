import PIL.Image
#import Image
from PIL import ImageTk
from tkinter import *
import tkinter as tk, threading
import cv2
import subprocess
import os
import imageio
import stat


video_file_path = "videos/IMG_4887.MOV"
video_file_name="IMG_4887.MOV"

video = imageio.get_reader(video_file_path)



def stream(label):

    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(PIL.Image.fromarray(image))
        label.config(image=frame_image)
        label.im = frame_image
      #  label.tk_im = ImageTk.PhotoImage(label.im)
      #  label.tk_im = frame_image
        label.canvas.create_image(0, 0, anchor="nw", image=label.im)


class ExampleApp(Label):
    def __init__(self,master):
        Label.__init__(self,master=None)
        self.x = self.y = 0

        self.im = ImageTk.PhotoImage(PIL.Image.open("first_frame.jpg"))
    #    self.tk_im = ImageTk.PhotoImage(self.im)


        self.canvas = Canvas(self, width=self.im.width(), height=self.im.height(), cursor="cross")
       # self.canvas = Canvas(self, width=200, height=200, cursor="cross")
        self.sbarv=Scrollbar(self,orient=VERTICAL)
        self.sbarh=Scrollbar(self,orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        self.sbarv.grid(row=0,column=1,stick=N+S)
        self.sbarh.grid(row=1,column=0,sticky=E+W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # self.grid_rowconfigure(0, minsize=500, weight=1)
        # self.grid_columnconfigure(0, minsize=500, weight=1)

        self.rect = None

        self.start_x = None
        self.start_y = None

       # self.im = PIL.Image.open("air_hockey_frames/air_hockey_frame6.jpg")
        self.wazil,self.lard=self.im.width(), self.im.height()
        self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))

        self.canvas.create_image(0,0,anchor="nw",image=self.im)


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9*w:
            self.canvas.xview_scroll(1, 'units')
        elif event.x < 0.1*w:
            self.canvas.xview_scroll(-1, 'units')
        if event.y > 0.9*h:
            self.canvas.yview_scroll(1, 'units')
        elif event.y < 0.1*h:
            self.canvas.yview_scroll(-1, 'units')

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        print(event.x)
        print(event.y)
        left_arg = "-l " + str(self.start_x) + " "
        top_arg = "-t " + str(self.start_y) + " "
        width_arg = "-w " + str(event.x-self.start_x) + " "
        height_arg = "-h " + str(event.y-self.start_y) + " "
        video_arg = "-f " + video_file_path+" "
        output_arg = "-o " + "draw_crop_test"


        command = "bash " + "crop_tool.sh " + video_arg+left_arg+top_arg+width_arg+height_arg+output_arg
        os.chmod("./output_command.sh", 0o755)
        with open("output_command.sh", "w") as text_file:
            text_file.write('#!/bin/bash')
            text_file.write("\n")
            text_file.write(command + "\n")
            text_file.write('#hello')


        os.chmod("./output_command.sh", 0o755)
        subprocess.check_call(["./output_command.sh"])

        #video_model_command = "python test_video.py --draw_crop_test.mp4 --arch resnet3d50"
        os.system("python test_video.py --video_file " + "draw_crop_test.mp4 --arch resnet3d50")









       # subprocess.check_call(["crop_tool.sh", video_arg, left_arg, top_arg, width_arg, height_arg, output_arg])

        pass


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file



if __name__ == "__main__":

    getFirstFrame(video_file_path)

    root=Tk()
    app = ExampleApp(root)
    app.pack()

    thread = threading.Thread(target=stream, args=(app,))
    thread.daemon = 1
    thread.start()
    root.mainloop()





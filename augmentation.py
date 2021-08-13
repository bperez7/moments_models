import numpy as np
import cv2
from PIL import Image, ImageSequence
import vidaug.augmentors as va

def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames

def video_loader(path):
    frames = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()


        if ret:
            frames.append(frame)
                #frames.append(frame.convert(modality))
        #index += 1
        else:
            break
    cap.release()
    return frames, fps


def color_invert_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder + "/" + filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([

        sometimes(va.InvertColor())  # invert color
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="color_invert")

def horizontal_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.HorizontalFlip())  # horizontally flip the video with 100% probability
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="horizontal")

def blur_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.GaussianBlur())  #blurs
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="blur")
def elastic_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.ElasticTransformation())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="elastic")
def piecewise_affine_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.PiecewiseAffineTransform())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="piecewise_affine")
def superpixel_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.Superpixel())  #blurs
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="superpixel")
def salt_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.Salt())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="salt")
def pepper_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.Pepper())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="pepper")
def add_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.Add())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="add")
def multiply_transform(folder, filename, output_folder):
    frames, fps = video_loader(folder+"/"+filename)
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.Multiply())
    ])
    video_aug = seq(frames)
    write_video(video_aug, fps, output_folder, filename, augmentation_type="multiply")




def write_video(frames, fps, output_folder, filename, augmentation_type=""):
   # duration = len(frames)
    #fps = fps
    #size = (1080, 1920, 3)
    size = tuple(frames[0].shape)
    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    # out = cv2.VideoWriter('output_trial_aug.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(output_folder+"/"+filename+augmentation_type+'.mp4',
                          fourcc, float(fps), (size[1], size[0]))
    for i in range(len(frames)):
        data = frames[i]
        out.write(data)

    out.release()

# sometimes = lambda aug: va.Sometimes(1, aug) # Used to apply augmentor with 100% probability
# seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]
#     sometimes(va.HorizontalFlip()) # horizontally flip the video with 100% probability
# ])

#frames = video_loader('videos/IMG_1433.MOV')

salt_transform('videos', "IMG_1433.MOV",'videos')

# video_aug = seq(frames)
# print(video_aug)
#
# print(video_aug[0].shape)


#ize = 720*16//9, 720
#
# duration=len(video_aug)
# fps = 25
# size = (1080,1920,3)
# #fourcc = cv2.VideoWriter_fourcc(*'MP42')
# #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc(*'FMP4')

# #out = cv2.VideoWriter('output_trial_aug.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
# out = cv2.VideoWriter('output_trial_aug.mp4', fourcc, float(fps), (size[1], size[0]))
# for i in range(len(video_aug)):
#     data = video_aug[i]
#     #print(data.shape)
#    # data = np.random.randint(0, 256, size, dtype='uint8')
#     out.write(data)
#
# out.release()
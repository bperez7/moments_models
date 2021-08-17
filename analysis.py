from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from utils import extract_frames
from custom_dataset import CustomImageTrainDataset, CustomImageValDataset
import models
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from models import ResNet3D

#TODO:
# 1. Parallel bug (probably from input batch being size 1)
#



model_folder = "trained_models"
#model = ResNet3D()
model_name = "model_debug.pth"
#model.load_state_dict(torch.load(model_folder+"/"+model_name))

model = torch.load(model_folder+"/"+model_name)
model = model.cuda()
#model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
model.eval()

print(model)


training_true_labels = [0 for i in range(33)]
training_true_labels[8:20] = [1 for i in range(8,20)]
training_true_labels[20:] = [2 for i in range(20,33)]
training_pred_labels = []

val_true_labels = [0 for i in range(9)]
val_true_labels[2:6] = [1 for i in range(2,6)]
val_true_labels[6:] = [2 for i in range(6,9)]
val_pred_labels = []

train_csv_path = "dataset/train_labels_2.csv"
val_csv_path = "dataset/val_labels_2.csv"
videos_path = "videos/label_videos"
train_batch_size = 1
val_batch_size = 1

train_loader = torch.utils.data.DataLoader(
        CustomImageTrainDataset(train_csv_path, videos_path),
        batch_size=train_batch_size,shuffle=False
    )

val_loader = torch.utils.data.DataLoader(
        CustomImageValDataset(val_csv_path, videos_path),
        batch_size = val_batch_size,shuffle=False
    )

all_train_input_files = ["bulldozing/bulldozing_2.mp4",
"bulldozing/bulldozing_3.mp4",
"bulldozing/bulldozing_4.mp4",
"bulldozing/bulldozing_5.mp4",
"bulldozing/bulldozing_6.mp4",
"bulldozing/bulldozing_8.mp4",
"bulldozing/bulldozing_9.mp4",
"excavating/excavating_1.mp4",
"excavating/excavating_2.mp4",
"excavating/excavating_3.mp4",
"excavating/excavating_5.mp4",
"excavating/excavating_6.mp4",
"excavating/excavating_9.mp4",
"excavating/excavating_10.mp4",
"excavating/excavating_11.mp4",
"excavating/excavating_12.mp4",
"excavating/excavating_13.mp4",
"excavating/excavating_14.mp4",
"excavating/excavating_15.mp4",
"excavating/excavating_16.mp4",
"loading/loading_1.mp4",
"loading/loading_2.mp4",
"loading/loading_3.mp4",
"loading/loading_4.mp4",
"loading/loading_5.mp4",
"loading/loading_6.mp4",
"loading/loading_7.mp4",
"loading/loading_8.mp4",
"loading/loading_9.mp4",
"loading/loading_10.mp4",
"loading/loading_11.mp4",
"loading/loading_12.mp4",
"loading/loading_13.mp4"

]

all_val_input_files = ["bulldozing/bulldozing_12.mp4",
"bulldozing/bulldozing_13.mp4",
"excavating/excavating_17.mp4",
"excavating/excavating_18.mp4",
"excavating/excavating_19.mp4",
"excavating/excavating_20.mp4",
"loading/loading_14.mp4",
"loading/loading_15.mp4",
"loading/loading_16.mp4"
]

with torch.no_grad():
    training_correct = 0

    training_true_labels = []

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

       # target = target.cuda(async=True)
        target = target.long()
        training_true_labels.append(int(target))


       # input = input.cuda()
       # input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        print(output)
        print(target)
        training_pred_labels.append(int(output))


    val_true_labels = []
    val_pred_labels = []
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time

        # target = target.cuda(async=True)
        target = target.long()
        val_true_labels.append(int(target))
        # input = input.cuda()
        # input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        val_pred_labels.append(int(output))



    # for test_input_file in all_train_input_files:
    #     test_input_frames = extract_frames("videos/label_videos/" + test_input_file, 8)
    #     transform = models.load_transform()
    #     test_input = torch.stack([transform(frame) for frame in test_input_frames], 1).unsqueeze(0)
    #     #test_input = torch.stack([transform(frame) for frame in test_input_frames], 1)
    #     test_input = test_input.cuda()
    #     #test_input = torch.stack([transform(frame) for frame in test_input_frames], 1)
    #     print(test_input)
    #     print(test_input.size())
    #     test_output = model(test_input)
    #     print(test_input_file)
    #     print(test_output)
    #     # prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
    #     # maxk=(1,2)
    #     maxk = 2
    #     _, pred = test_output.topk(maxk)
    #     pred = pred.t()
    #     print(_)
    #     print(pred)
    #     pred_label = int(pred[0])
    #     training_pred_labels.append(pred_label)
    #
    #     if "bulldozing" in test_input_file:
    #         if int(pred[0]) == 0:
    #             training_correct += 1
    #     elif "excavating" in test_input_file:
    #         if int(pred[0]) == 1:
    #             training_correct += 1
    #     elif "loading" in test_input_file:
    #         if int(pred[0]) == 2:
    #             training_correct += 1
    #     print('Training Accuracy: ' + str(training_correct / 33))

    # val_correct = 0
    # for test_input_file in all_val_input_files:
    #     test_input_frames = extract_frames("videos/label_videos/" + test_input_file, 8)
    #     transform = models.load_transform()
    #     test_input = torch.stack([transform(frame) for frame in test_input_frames], 1).unsqueeze(0)
    #     test_input = test_input.cuda()
    #     test_output = model(test_input)
    #     print(test_input_file)
    #     print(test_output)
    #     # prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
    #     # maxk=(1,2)
    #     maxk = 2
    #     _, pred = test_output.topk(maxk)
    #     pred = pred.t()
    #     print(_)
    #     print(pred)
    #
    #     if "bulldozing" in test_input_file:
    #         if int(pred[0]) == 0:
    #             val_correct += 1
    #     elif "excavating" in test_input_file:
    #         if int(pred[0]) == 1:
    #             val_correct += 1
    #     elif "loading" in test_input_file:
    #         if int(pred[0]) == 2:
    #             val_correct += 1
    #     pred_label = int(pred[0])
    #     val_pred_labels.append(pred_label)
    # print('Validation Accuracy: ' + str(val_correct / 9))



#training
y_true = []
y_pred = []

training_cm = confusion_matrix(training_true_labels, training_pred_labels)
val_cm = confusion_matrix(val_true_labels, val_pred_labels)

print("Training Confusion Matrix")
print(training_cm)
print("Validation Confusion Matrix")
print(val_cm)

#disp = ConfusionMatrixDisplay(confusion_matrix=cm)


#print("Confusion Matrix")
#print(cm)


#disp.plot()

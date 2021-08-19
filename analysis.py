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
model_name = "model_aug_08-17.pth"
#model.load_state_dict(torch.load(model_folder+"/"+model_name))

model = torch.load(model_folder+"/"+model_name)
model = model.cuda()
#model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
model.eval()

print(model)



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


with torch.no_grad():
    training_correct = 0

    training_true_labels = []
    training_pred_labels = []

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

       # target = target.cuda(async=True)
        target = target.long()
        print(int(target))
        training_true_labels.append(int(target))


       # input = input.cuda()
       # input = input.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        print(output)
        print(target)
        print(int(torch.argmax(output)))
        training_pred_labels.append(int(torch.argmax(output)))


    val_true_labels = []
    val_pred_labels = []
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time

        # target = target.cuda(async=True)
        target = target.long()

        val_true_labels.append(int(target))
        # input = input.cuda()
        # input = input.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        val_pred_labels.append(int(torch.argmax(output)))



#training
y_true = []
y_pred = []

print(training_true_labels)
print(training_pred_labels)
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

import argparse
import os
import time
import shutil
import torch
import json
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from utils import extract_frames
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# from dataset import TSNDataSet
# from models import TSN
# from transforms import *
# from opts import parser
# import datasets_video

import matplotlib.pyplot as plt




import numpy as np
import models

from custom_dataset import CustomImageTrainDataset, CustomImageValDataset

loss_list = []

best_prec1 = 0

os.environ['KMP_DUPLICATE_LIB_OK']='True'
"""
TODO:
1. Freeze layers 
2. dimension differnece between TRN 
3. Could need to fix format of videos to be recognized by ffmpeg (see localization error an label_videos)
4. SAVING MODEL BUG and parallelization
"""

def main():
    global lr_steps, start_epoch, epochs, eval_freq, lr, momentum, weight_decay, print_freq

    training_loss_list = []

    # lr_steps = [50, 100]
    # start_epoch = 0
    # epochs = 10
    # eval_freq = 2
    # lr = .001
    # momentum = .9
    # weight_decay=5e-4
    # print_freq = 1

    config_file = open('./configs/config_file.json')
    config = json.load(config_file)

    hyperparameters = config["hyperparameters"]
    lr_steps = hyperparameters['lr_steps']
    start_epoch = hyperparameters["start_epoch"]
    epochs = hyperparameters["epochs"]
    lr = hyperparameters["lr"]
    momentum = hyperparameters["momentum"]
    weight_decay = hyperparameters["weight_decay"]
    print_freq = hyperparameters["print_freq"]
    eval_freq = hyperparameters["eval_freq"]

    num_class = hyperparameters["num_classes"]

    output_model_name = hyperparameters["output_model_name"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global args, best_prec1
   # args = parser.parse_args()
 #   check_rootfolders()

    #categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
   # categories = ?  """TODO: fix categories"""
  #  num_class = len(categories)
  #  num_class = 2


    #Load resnet pretrained model
    model = models.load_model("resnet3d50")


    #add layers to specify number of classes
    expansion = 4
    #model.fc = torch.nn.Linear(512 * expansion, 306)
    model.last_linear = torch.nn.Linear(in_features=512 * expansion, out_features=num_class, bias=True)
   # print(model)


    #For GPU parallelization
    #model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    # #if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(("=> loading checkpoint '{}'".format(args.resume)))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print(("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.evaluate, checkpoint['epoch'])))
    #     else:
    #         print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # # Data loading code (for TRN)
    # if args.modality != 'RGBDiff':
    #     normalize = GroupNormalize(input_mean, input_std)
    # else:
    #     normalize = IdentityTransform()
    #
    # if args.modality == 'RGB':
    #     data_length = 1
    # elif args.modality in ['Flow', 'RGBDiff']:
    #     data_length = 5



    train_csv_path = config["datasets"]["training_csv"]
    val_csv_path = config["datasets"]["val_csv"]
    videos_path = config["datasets"]["videos_path"]

    train_loader = torch.utils.data.DataLoader(
        CustomImageTrainDataset(train_csv_path, videos_path),
        batch_size=33,shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        CustomImageValDataset(val_csv_path, videos_path)
    )


    # define loss function (criterion) and optimizer
    # if args.loss_type == 'nll':
    criterion = torch.nn.CrossEntropyLoss().cuda()
    #criterion = torch.nn.CrossEntropyLoss().to(device)



    optimizer_name = hyperparameters["optimizer"]
    if optimizer_name=="adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    elif optimizer_name=="sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr
                                    # momentum=momentum,
                                    # weight_decay=weight_decay
                                    )


    log_training = open("log_training",'w')
    # log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    val_loss_list = []
    for epoch in range(start_epoch, epochs):
       print('epoch: ' + str(epoch))

       """Temporarily removed adaptive learning rate"""
       # adjust_learning_rate(optimizer, epoch, args.lr_steps)
       adjust_learning_rate(optimizer, epoch, lr_steps)

        # train for one epoch
       training_loss = train(train_loader, model, criterion, optimizer, epoch, log_training)
       training_loss_list.append(training_loss)

        # evaluate on validation set

       if ((epoch+1)% eval_freq==0 or epoch == epochs-1):
            #prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)
            val_loss = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)
            val_loss_list.append(val_loss)
            # remember best prec@1 and save checkpoint
          #  is_best = prec1 > best_prec1
          #  best_prec1 = max(prec1, best_prec1)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            # }, is_best)


    print("Losses per epoch: ")
    print(training_loss_list)
    print(val_loss_list)
    plt.plot(training_loss_list)
    plt.plot(val_loss_list)

    torch.save(model, "trained_models/"+output_model_name)

    #test on training data
    #test_input_file = "videos/label_videos/excavating/excavating_1.mp4"
    #test_input_file = "videos/label_videos/lowering/lowering_1.mp4"
    test_input_file = config["datasets"]["debug_video"]
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

    training_correct = 0
    training_pred_labels = []
    for test_input_file in all_train_input_files:
        test_input_frames = extract_frames("videos/label_videos/"+test_input_file, 8)
        transform = models.load_transform()
        test_input =  torch.stack([transform(frame) for frame in test_input_frames], 1).unsqueeze(0)
        test_output = model(test_input)
        print(test_input_file)
        print(test_output)
        # prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
        # maxk=(1,2)
        maxk = 2
        _, pred = test_output.topk(maxk)
        pred = pred.t()
        print(_)
        print(pred)

        if "bulldozing" in test_input_file:
            if int(pred[0])==0:
                training_correct+=1
        elif "excavating" in test_input_file:
            if int(pred[0])==1:
                training_correct+=1
        elif "loading" in test_input_file:
            if int(pred[0])==2:
                training_correct+=1
        pred_label = int(pred[0])
        training_pred_labels.append(pred_label)

    print('Training Accuracy: ' + str(training_correct/33))

    val_correct = 0
    val_pred_labels = []
    for test_input_file in all_val_input_files:
        test_input_frames = extract_frames("videos/label_videos/" + test_input_file, 8)
        transform = models.load_transform()
        test_input = torch.stack([transform(frame) for frame in test_input_frames], 1).unsqueeze(0)
        test_output = model(test_input)
        print(test_input_file)
        print(test_output)
        # prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
        # maxk=(1,2)
        maxk = 2
        _, pred = test_output.topk(maxk)
        pred = pred.t()
        print(_)
        print(pred)

        if "bulldozing" in test_input_file:
            if int(pred[0]) == 0:
                val_correct += 1
        elif "excavating" in test_input_file:
            if int(pred[0]) == 1:
                val_correct += 1
        elif "loading" in test_input_file:
            if int(pred[0]) == 2:
                val_correct += 1
        pred_label = int(pred[0])
        val_pred_labels.append(pred_label)
    print('Validation Accuracy: ' + str(val_correct / 9))

    training_true_labels = [0 for i in range(33)]
    training_true_labels[8:20] = [1 for i in range(8, 20)]
    training_true_labels[20:] = [2 for i in range(20, 33)]


    val_true_labels = [0 for i in range(9)]
    val_true_labels[2:6] = [1 for i in range(2, 6)]
    val_true_labels[6:] = [2 for i in range(6, 9)]


    training_cm = confusion_matrix(training_true_labels, training_pred_labels)
    val_cm = confusion_matrix(val_true_labels, val_pred_labels)

    print("Training Confusion Matrix")
    print(training_cm)
    print("Validation Confusion Matrix")
    print(val_cm)

    f=open("result_train_1.txt", "w")

    training_results = ['Training Accuracy: ' + str(training_correct/33) + "\n",
                        'Validation Accuracy: ' + str(val_correct / 9) + "\n",
                        "Training Confusion Matrix" +"\n",
                        str(training_cm),"\n",
                        "Validation Confusion Matrix" + "\n",
                        str(val_cm)
                        ]
    for l in training_results:
        f.write(l)





    # for i in range(10):
    #     test_output = model(test_input)
    #     print(test_output)
    #     #prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
    #    # maxk=(1,2)
    #     maxk=2
    #     _, pred = test_output.topk(maxk)
    #     pred = pred.t()
    #     print(_)
    #     print(pred)


    plt.savefig('losses_plot.png')



def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topK = AverageMeter()

    # switch to train mode
    model.train()

    #top k config
    config_file = open('./configs/config_file.json')
    config = json.load(config_file)
    max_k = config["misc"]["topk"]

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        target = target.cuda(async=True)


        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        print(loss)


        # measure accuracy and record loss
        prec1, prec_k = accuracy(output.data, target, topk=(1,max_k))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        topK.update(prec_k, input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:

            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@K {topK.val:.3f} ({topK.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, topK=topK, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        return float(loss)



def validate(val_loader, model, criterion, iter, log):
    global print_freq

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topK = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #max k config
    config_file = open('./configs/config_file.json')
    config = json.load(config_file)
    max_k = config["misc"]["topk"]

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
     #   input_var = torch.autograd.Variable(input, volatile=True)
        with torch.no_grad():
            input_var = input
            target_var = target
       # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec_k = accuracy(output.data, target, topk=(1,max_k))

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        topK.update(prec_k, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@K {topK.val:.3f} ({topK.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, topK=topK))
            print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {topK.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, topK=topK, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

   #return top1.avg
    return float(loss)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr, weight_decay
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    """TODO: Should I decay? """
    lr = lr
    #lr = lr * decay
    decay = weight_decay
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_group['lr_mult']x
    #     param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

  #  print('max k: '+str(maxk))
    _, pred = output.topk(maxk)
    pred = pred.t()
  #  print(target)
  #  print(pred)

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

if __name__ == '__main__':
    main()

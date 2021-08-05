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

    config_file = open('config_file.json')
    config = json.load(config_file)

    hyperparameters = config["hyperparameters"]
    lr_steps = hyperparameters['lr_steps']
    start_epoch = hyperparameters["start_epoch"]
    epochs = hyperparameters["epochs"]
    lr = .001
    momentum = hyperparameters["momentum"]
    weight_decay = hyperparameters["weight_decay"]
    print_freq = hyperparameters["print_freq"]
    eval_freq = hyperparameters["eval_freq"]

    num_class = hyperparameters["num_classes"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global args, best_prec1
   # args = parser.parse_args()
 #   check_rootfolders()

    #categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
   # categories = ?  """TODO: fix categories"""
  #  num_class = len(categories)
  #  num_class = 2



   # args.store_name = '_'.join(['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments])
    #print('storing name: ' + args.store_name)


    # model = TSN(num_class, args.num_segments, args.modality,
    #             base_model=args.arch,
    #             consensus_type=args.consensus_type,
    #             dropout=args.dropout,
    #             img_feature_dim=args.img_feature_dim,
    #             partial_bn=not args.no_partialbn)

    model = models.load_model("resnet3d50")


    expansion = 4
    model.fc = torch.nn.Linear(512 * expansion, 306)
    model.last_linear = torch.nn.Linear(in_features=512 * expansion, out_features=num_class, bias=True)
   # print(model)




    # crop_size = model.crop_size
    # scale_size = model.scale_size
    # input_mean = model.input_mean
    # input_std = model.input_std
  #  policies = model.get_optim_policies()
  #   policies = model.base.policies()
   # train_augmentation = model.get_augmentation()

    #For GPU parallelization
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

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

    # # Data loading code
    # if args.modality != 'RGBDiff':
    #     normalize = GroupNormalize(input_mean, input_std)
    # else:
    #     normalize = IdentityTransform()
    #
    # if args.modality == 'RGB':
    #     data_length = 1
    # elif args.modality in ['Flow', 'RGBDiff']:
    #     data_length = 5

    # train_loader = torch.utils.data.DataLoader(
    #     TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
    #                new_length=data_length,
    #                modality=args.modality,
    #                image_tmpl=prefix,
    #                transform=torchvision.transforms.Compose([
    #                    train_augmentation,
    #                    Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
    #                    ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
    #                    normalize,
    #                ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    train_csv_path = config["datasets"]["training_csv"]
    val_csv_path = config["datasets"]["val_csv"]
    videos_path = config["datasets"]["videos_path"]

    train_loader = torch.utils.data.DataLoader(
        CustomImageTrainDataset(train_csv_path, videos_path),
        batch_size=2,shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        CustomImageValDataset(val_csv_path, videos_path)
    )

    # val_loader = torch.utils.data.DataLoader(
    #     TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
    #                new_length=data_length,
    #                modality=args.modality,
    #                image_tmpl=prefix,
    #                random_shift=False,
    #                transform=torchvision.transforms.Compose([
    #                    GroupScale(int(scale_size)),
    #                    GroupCenterCrop(crop_size),
    #                    Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
    #                    ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
    #                    normalize,
    #                ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    # if args.loss_type == 'nll':
    criterion = torch.nn.CrossEntropyLoss().cuda()
    #criterion = torch.nn.CrossEntropyLoss().to(device)

    # else:
    #     raise ValueError("Unknown loss type")

    # for group in policies:
    #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay)


    # if args.evaluate:
    #     validate(val_loader, model, criterion, 0)
    #     return

    log_training = open("log_training",'w')
    # log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    #for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(start_epoch, epochs):
       print('epoch: ' + str(epoch))
       # adjust_learning_rate(optimizer, epoch, args.lr_steps)
       adjust_learning_rate(optimizer, epoch, lr_steps)

        # train for one epoch
       training_loss = train(train_loader, model, criterion, optimizer, epoch, log_training)
     #  print('training loss')
     #  print(training_loss)
       training_loss_list.append(training_loss)

        # evaluate on validation set
       # if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
       if ((epoch+1)% eval_freq==0 or epoch == epochs-1):
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            # }, is_best)

    #plot loss function
    #global loss_list

    #plt.plot(loss_list)
    print(training_loss_list)
    plt.plot(training_loss_list)

    #test on training data
    #test_input_file = "videos/label_videos/excavating/excavating_1.mp4"
    test_input_file = "videos/label_videos/lowering/lowering_1.mp4"
    test_input_frames = extract_frames(test_input_file, 8)
    transform = models.load_transform()
    test_input =  torch.stack([transform(frame) for frame in test_input_frames], 1).unsqueeze(0)
    for i in range(10):
        test_output = model(test_input)
        print(test_output)
        #prec1, prec5 = accuracy(test_output.data, target, topk=(1, 2))
       # maxk=(1,2)
        maxk=2
        _, pred = test_output.topk(maxk)
        pred = pred.t()
        print(_)
        print(pred)


    plt.savefig('losses_plot.png')



def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #
    # if args.no_partialbn:
    #     model.module.partialBN(False)
    # else:
    #     model.module.partialBN(True)

    # switch to train mode
    model.train()

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

       # print(loss)
        # global loss_list
        # loss_list.append(losses)
        # print('losses')
        # print(losses)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,2))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

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
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            #print(output)
            log.write(output + '\n')
            log.flush()
        return float(loss)



def validate(val_loader, model, criterion, iter, log):
    global print_freq

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,2))

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
           # print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
   # print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
   # print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg


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
    lr = lr * decay
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
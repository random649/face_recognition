import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Backbone, ArcFace, FocalLoss
from torch.utils.tensorboard import SummaryWriter


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def train(args):
    DEVICE = torch.device(("cuda:%d"%args.gpu[0]) if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_root)
    train_transform = transforms.Compose([transforms.Resize([int(128*args.input_size/112), int(128*args.input_size/112)]),
                                        transforms.RandomCrop([args.input_size, args.input_size]),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[args.rgb_mean,args.rgb_mean,args.rgb_mean], std=[args.rgb_std,args.rgb_std,args.rgb_std])
                                        ])
    train_dataset = datasets.ImageFolder(args.data_root, train_transform)
    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    NUM_CLASS = len(train_loader.dataset.classes)

    BACKBONE = Backbone([args.input_size, args.input_size])
    HEAD = ArcFace(args.emb_dims, NUM_CLASS, device_id=args.gpu)
    LOSS = FocalLoss()
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)
    _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    optimizer = optim.SGD([{'params': backbone_paras_wo_bn+head_paras_wo_bn, 'weight_decay': args.weight_decay},
                        {'params': backbone_paras_only_bn}], lr=args.lr, momentum=args.momentum)
    BACKBONE = nn.DataParallel(BACKBONE, device_ids=args.gpu)
    BACKBONE = BACKBONE.to(DEVICE)

    dispaly_frequency = len(train_loader) // 100
    NUM_EPOCH_WARM_UP = args.num_epoch // 25
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    batch = 0
    print('Start training at %s!' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for epoch in range(args.num_epoch):
        if epoch==args.stages[0] or epoch==args.stages[1] or epoch==args.stages[2]:
            for params in optimizer.param_groups:
                params['lr'] /= 10.
        BACKBONE.train()
        HEAD.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for inputs, labels in train_loader:
            if (epoch+1 <= NUM_EPOCH_WARM_UP) and (batch+1 <= NUM_BATCH_WARM_UP):
                for params in optimizer.param_groups:
                    params['lr'] = (batch+1) * args.lr / NUM_BATCH_WARM_UP
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1,5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1
            if batch % dispaly_frequency == 0:
                print('%s Epoch %d/%d Batch %d/%d: train loss %f, train prec@1 %f, train prec@5 %f' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch, args.num_epoch, batch, len(train_loader)*args.num_epoch, losses.avg, top1.avg, top5.avg))
        writer.add_scalar('Train_Loss', losses.avg, epoch+1)
        writer.add_scalar('Train_Top1_Accuracy', top1.avg, epoch+1)
        writer.add_scalar('Train_Top5_Accuracy', top5.avg, epoch+1)
        torch.save(BACKBONE.module.state_dict(), os.path.join(args.ckpt_root, 'backbone_epoch%d.pth'%(epoch+1)))
        torch.save(HEAD.state_dict(), os.path.join(args.ckpt_root, 'head_epoch%d.pth'%(epoch+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/CASIA-clean-112')
    parser.add_argument('--ckpt_root', type=str, default='ckpt')
    parser.add_argument('--log_root', type=str, default='log')
    parser.add_argument('--gpu', type=list, default=[0])
    parser.add_argument('--input_size', type=int, default=112)
    parser.add_argument('--rgb_mean', type=float, default=0.5)
    parser.add_argument('--rgb_std', type=float, default=0.5)
    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=125)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--stages', type=list, default=[35, 65, 95])
    args = parser.parse_args()
    print(str(args))
    os.makedirs(args.ckpt_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)
    train(args)

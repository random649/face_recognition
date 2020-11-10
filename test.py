import os
import cv2
import argparse
import time
from datetime import datetime
import numpy as np
import torch
from model import Backbone, ArcFace, FocalLoss
import torchvision.transforms as transforms
import tqdm

import config


def get_feature(img_path, transform, model, device):
    img = cv2.imread(img_path)
    img_flip = cv2.flip(img, 1)
    img = transform(img)
    img_flip = transform(img_flip)
    img = img.unsqueeze(0).to(device)
    img_flip = img_flip.unsqueeze(0).to(device)
    feat = model(img)[0].data.cpu().numpy()
    feat_flip = model(img_flip)[0].data.cpu().numpy()
    feat = np.concatenate((feat, feat_flip))
    return feat


def cal_accuracy(sims, labels):
    best_acc = 0
    best_th = 0
    for i in range(len(sims)):
        th = sims[i]
        pred = (sims >= th)
        acc = np.mean((pred == labels).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def test(args):
    device = torch.device(('cuda:%d'%args.gpu) if torch.cuda.is_available() else 'cpu')
    BACKBONE = Backbone([args.input_size, args.input_size], args.num_layers, args.mode)
    BACKBONE.load_state_dict(torch.load(args.ckpt_path))
    BACKBONE.to(device)
    BACKBONE.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    print('Start test at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # accuracy
    with open(args.pair_file, 'r') as f:
        pairs = f.readlines()
    sims = []
    labels = []
    for pair_id, pair in tqdm.tqdm(enumerate(pairs)):
        # print('processing %d/%d...' % (pair_id, len(pairs)), end='\r')
        splits = pair.split()
        feat1 = get_feature(os.path.join(args.data_root, splits[0]), transform, BACKBONE, device)
        feat2 = get_feature(os.path.join(args.data_root, splits[1]), transform, BACKBONE, device)
        label = int(splits[2])
        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1)*np.linalg.norm(feat2))
        sims.append(sim)
        labels.append(label)
    acc, th = cal_accuracy(np.array(sims), np.array(labels))
    print('acc=%f with threshold=%f' % (acc, th))
    print('Finish test at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=config.TEST_DATA)
    parser.add_argument('--pair_file', type=str, default=config.TEST_PAIR)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=112)
    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=152, choices=(50, 100, 152))
    parser.add_argument('--mode', type=str, default='ir', choices=('ir', 'ir_se'))
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(config.TEST_CKPT, 'backbone_epoch{}.pth'.format(config.TEST_EPOCH)))
    args = parser.parse_args()
    print(str(args))
    test(args)

import sys
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from pascal_voc_loader import PascalVOCLoader
from enet import Enet, Encoder
from metrics import scores
from loss import cross_entropy2d


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']


def train(args):
    data_path = get_data_path("pascal")
    img_size = 512
    loader = PascalVOCLoader(data_path, is_transform=True, img_size=img_size, label_scale=8)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size)
    another_loader = PascalVOCLoader(data_path, split='trainval', is_transform=True, img_size=img_size)
    valloader = data.DataLoader(
        another_loader, batch_size=args.batch_size)
    # Setup Model
    model = Encoder(n_classes)
    print(model)
    if torch.cuda.is_available():
        model.cuda(0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    pooling_stack = None
    for epoch in xrange(args.epochs):
        model.train()
        file = open('../{}_{}.txt'.format('enet', epoch), 'w')
        gts, preds = [], []
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad
            outputs, pooling_stack = model(images)
            loss = cross_entropy2d(outputs, labels, ignore_index=0)
            loss.backward()
            optimizer.step()
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        
        score, class_iou = scores(gts, preds, n_class=n_classes)
        for k, v in score.items():
            file.write('{} {}\n'.format(k, v))

        for i in range(n_classes):
            file.write('{} {}\n'.format(i, class_iou[i]))
        torch.save(model.state_dict(), "../{}_{}_{}.pkl".format(
            'enet_encoder', 'pascal', epoch))
        file.close()
        
        '''
        model.eval()
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        score, class_iou = scores(gts, preds, n_class=n_classes)
        for k, v in score.items():
            print k, v

        for i in range(n_classes):
            print i, class_iou[i]
        '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperprams')
    parser.add_argument('--epochs', nargs='?', type=int, default=600,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--lr_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    args = parser.parse_args()
    train(args)

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
from loader import get_loader
from torch.optim.lr_scheduler import StepLR

hist = np.array([10682767, 14750079, 623349, 20076880, 2845085, 6166762, 743859, 714595, 3719877, 405385, 184967, 2503995]).astype('float')


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']


def train(args):
    data_path = get_data_path(args.dataset)
    data_loader = get_loader(args.dataset)
    label_scale = 1
    if (args.model == 'encoder'):
        label_scale = 8
    loader = data_loader(data_path, is_transform=True, label_scale=label_scale)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size)
    another_loader = data_loader(data_path, split='val', is_transform=True, label_scale=label_scale)
    valloader = data.DataLoader(
        another_loader, batch_size=args.batch_size)
    # Setup Model
    if (args.model == 'encoder'):
        model = Encoder(n_classes, train=True)
    else:
        encoder_weight = torch.load('enet_encoder_camvid_299.pkl')
        del encoder_weight['classifier.bias']
        del encoder_weight['classifier.weight']
        model = Enet(n_classes)
        model.encoder.load_state_dict(encoder_weight)

    # compute weight for cross_entropy2d
    norm_hist = hist / hist.sum()
    weight = 1 / np.log(norm_hist + 1.02)
    # unlabeled data is not used. 
    weight[-1] = 0
    weight = torch.FloatTensor(weight)

    if torch.cuda.is_available():
        model.cuda(0)
        weight = weight.cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.w_decay)
    scheduler = StepLR(optimizer, step_size=100, gamma=args.lr_decay)
    pooling_stack = None
    model.train()
    for epoch in xrange(args.epochs):
        scheduler.step()
        loss_list = []
        file = open('../another_' + args.model + '/{}_{}.txt'.format('enet', epoch), 'w')
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad()
            if (args.model == 'encoder'):
                outputs, pooling_stack = model(images)
            else:
                outputs = model(images)
            loss = cross_entropy2d(outputs, labels, weight=weight)
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()
        
        file.write(str(np.average(loss_list)))
        
        model.eval()
        gts, preds = [], []
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
            file.write('{} {}\n'.format(k, v))

        for i in range(n_classes):
            file.write('{} {}\n'.format(i, class_iou[i]))
        torch.save(model.state_dict(), "../another_" + args.model + "/{}_{}_{}.pkl".format(
            'enet_' + args.model, args.dataset, epoch))
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperprams')
    parser.add_argument('--epochs', nargs='?', type=int, default=600,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch Size')
    parser.add_argument('--lr_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--w_decay', nargs='?', type=float, default=2e-4,
                        help='Weight Decay')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=1e-1,
                        help='Learning Rate Decay')                    
    parser.add_argument('--dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--model', nargs='?', type=str, default='encoder',
                        help='Model to train [\'encoder, decoder\']')
    args = parser.parse_args()
    train(args)

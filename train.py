import sys
import torch
import argparse
import visdom
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from pascal_voc_loader import PascalVOCLoader
from enet import Enet
from metrics import scores


def train(args):
    data_path = '/home/vietdoan/Workingspace/Enet/pascal/VOCdevkit/VOC2012/'
    loader = PascalVOCLoader(data_path, is_transform=True)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size, shuffle=True)
    another_loader = PascalVOCLoader(data_path, split='trainval', is_transform=True)
    valloader = data.DataLoader(
        another_loader, batch_size=args.batch_size)
    # Setup Model
    model = Enet(n_classes)
    print(model)
    if torch.cuda.is_available:
        model.cuda(0)
    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    for epoch in xrange(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 200 == 0:
                print("Epoch [%d/%d] Loss: %.4f" %
                      (epoch + 1, args.epochs, loss.data[0]))
        torch.save(model, "{}_{}_{}.pkl".format(
            'enet', 'pascal', epoch))

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
            print k, v

        for i in range(n_classes):
            print i, class_iou[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperprams')
    parser.add_argument('--epochs', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch Size')
    parser.add_argument('--lr_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    args = parser.parse_args()
    train(args)

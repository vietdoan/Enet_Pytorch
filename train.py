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


def train(args):
    data_path = '/home/vietdoan/Workingspace/Enet/pascal/VOCdevkit/VOC2012/'
    loader = PascalVOCLoader(data_path, is_transform=True)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size, shuffle=True)
    # Setup visdom for visualization
    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibaches',
                                     ylabel='Loss',
                                     title='Traning Loss',
                                     legend=['Loss']))
    # Setup Model
    model = Enet(n_classes)
    print model
    test_image, test_segmap = loader[0]
    if torch.cuda.is_available():
        model.cuda(0)
        test_image = Variable(test_image.unsqueeze(0).cuda(0))
    else:
        test_image = Variable(test_image.unsqueeze(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    critetion = nn.NLLLoss2d()
    for epoch in xrange(args.epochs):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad
            outputs = model(images)
            loss = critetion(outputs, labels)
            optimizer.step()
            if (i + 1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" %
                      (epoch + 1, args.n_epoch, loss.data[0]))
        torch.save(model, "{}_{}_{}_{}.pkl".format(
            args.arch, args.dataset, args.feature_scale, epoch))


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

import os
import random
import collections
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torch.utils import data
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Pad
from torchvision.transforms import ToTensor, ToPILImage


class CamvidLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=None, augment=False):
        self.root = root
        self.split = split
        self.img_size = [480, 640]
        self.is_transform = is_transform
        self.augment = augment
        self.n_classes = 12
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = [file for file in os.listdir(root + '/' + split) if file.endswith('.png')]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('P')

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = Scale(self.img_size, Image.BILINEAR)(img)
        lbl = Scale(self.img_size, Image.NEAREST)(lbl)
        if (self.augment):
            hflip = random.random()
            if (hflip < 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            img = ImageOps.expand(img, border=(transX, transY, 0, 0), fill=0)
            lbl = ImageOps.expand(lbl, border=(transX, transY, 0, 0), fill=11)
            img = img.crop((0, 0, img.size[0]-transX, img.size[1]-transY))
            lbl = lbl.crop((0, 0, lbl.size[0]-transX, lbl.size[1]-transY))
        img = ToTensor()(img)
        img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        lbl = np.array(lbl)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [255, 69, 0]
        Pavement = [128, 64, 128]
        Tree = [60, 40, 222]
        SignSymbol = [128, 128, 0]
        Fence = [192, 128, 128]
        Car = [64, 64, 128]
        Pedestrian = [64, 0, 128]
        Bicyclist = [64, 64, 0]
        Unlabeled = [0, 128, 192]

        label_colours = np.array([Sky, Building, Pole, Road,
                                  Pavement, Tree, SignSymbol, Fence, Car,
                                  Pedestrian, Bicyclist, Unlabeled])
        r = np.zeros_like(temp)
        g = np.zeros_like(temp)
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        rgb = np.array(rgb, dtype=np.uint8)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/vietdoan/Workingspace/Enet/camvid'
    dst = CamvidLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 3:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img *= np.array([0.229, 0.224, 0.225])
            img += np.array([0.485, 0.456, 0.406])
            img *= 255
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.show()
            dst.decode_segmap(labels.numpy()[0], plot=True)
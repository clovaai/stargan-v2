"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from core.data_loader import get_eval_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size)
    print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE
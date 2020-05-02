"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Lines (19 to 80) were adapted from https://github.com/1adrianb/face-alignment
Lines (83 to 235) were adapted from https://github.com/protossw512/AdaptiveWingLoss
"""

from collections import namedtuple
from copy import deepcopy
from functools import partial

from munch import Munch
import numpy as np
import cv2
from skimage.filters import gaussian
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_preds_fromhm(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    return preds


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(nn.Module):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)  # (2, height, width)

            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))  # (height, width)
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)

            self.coords = coords.unsqueeze(0).to(device)  # (1, 2 or 3, height, width)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.repeat(x.size(0), 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class FAN(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256, 256, True, False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_module('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_module('top_m_0', ConvBlock(256, 256))
        self.add_module('conv_last0', nn.Conv2d(256, 256, 1, 1, 0))
        self.add_module('bn_end0', nn.BatchNorm2d(256))
        self.add_module('l0', nn.Conv2d(256, num_landmarks+1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                              if k in model_weights})
        self.load_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._modules['m0'](x, tmp_out)
        ll = self._modules['top_m_0'](ll)
        ll = F.relu(self._modules['bn_end0']
                    (self._modules['conv_last0'](ll)), True)

        # Predict heatmaps
        tmp_out = self._modules['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @torch.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=256, mode='bilinear')
        x_01 = x*0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.size(2) // heatmaps.size(2)
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps

    @torch.no_grad()
    def get_landmark(self, x):
        ''' outputs landmarks of x.shape '''
        heatmaps = self.get_heatmap(x, b_preprocess=False)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        return landmarks


# ========================== #
#   Align related functions  #
# ========================== #


def tensor2numpy255(tensor):
    """Converts torch tensor to numpy array."""
    return ((tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype('uint8')


def np2tensor(image):
    """Converts numpy array to torch tensor."""
    return torch.FloatTensor(image).permute(2, 0, 1) / 255 * 2 - 1


class FaceAligner():
    def __init__(self, fname_wing, fname_celeba_mean, output_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fan = FAN(fname_pretrained=fname_wing).to(self.device).eval()
        scale = output_size // 256
        self.CELEB_REF = np.float32(np.load(fname_celeba_mean)['mean']) * scale
        self.xaxis_ref = landmarks2xaxis(self.CELEB_REF)
        self.output_size = output_size

    def align(self, imgs, output_size=256):
        ''' imgs = torch.CUDATensor of BCHW '''
        imgs = imgs.to(self.device)
        landmarkss = self.fan.get_landmark(imgs).cpu().numpy()
        for i, (img, landmarks) in enumerate(zip(imgs, landmarkss)):
            img_np = tensor2numpy255(img)
            img_np, landmarks = pad_mirror(img_np, landmarks)
            transform = self.landmarks2mat(landmarks)
            rows, cols, _ = img_np.shape
            rows = max(rows, self.output_size)
            cols = max(cols, self.output_size)
            aligned = cv2.warpPerspective(img_np, transform, (cols, rows), flags=cv2.INTER_LANCZOS4)
            imgs[i] = np2tensor(aligned[:self.output_size, :self.output_size, :])
        return imgs

    def landmarks2mat(self, landmarks):
        T_origin = points2T(landmarks, 'from')
        xaxis_src = landmarks2xaxis(landmarks)
        R = vecs2R(xaxis_src, self.xaxis_ref)
        S = landmarks2S(landmarks, self.CELEB_REF)
        T_ref = points2T(self.CELEB_REF, 'to')
        matrix = np.dot(T_ref, np.dot(S, np.dot(R, T_origin)))
        return matrix


def points2T(point, direction):
    point_mean = point.mean(axis=0)
    T = np.eye(3)
    coef = -1 if direction == 'from' else 1
    T[:2, 2] = coef * point_mean
    return T


def landmarks2eyes(landmarks):
    idx_left = np.array(list(range(60, 67+1)) + [96])
    idx_right = np.array(list(range(68, 75+1)) + [97])
    left = landmarks[idx_left]
    right = landmarks[idx_right]
    return left.mean(axis=0), right.mean(axis=0)


def landmarks2mouthends(landmarks):
    left = landmarks[76]
    right = landmarks[82]
    return left, right


def rotate90(vec):
    x, y = vec
    return np.array([y, -x])


def landmarks2xaxis(landmarks):
    eye_left, eye_right = landmarks2eyes(landmarks)
    mouth_left, mouth_right = landmarks2mouthends(landmarks)
    xp = eye_right - eye_left  # x' in pggan
    eye_center = (eye_left + eye_right) * 0.5
    mouth_center = (mouth_left + mouth_right) * 0.5
    yp = eye_center - mouth_center
    xaxis = xp - rotate90(yp)
    return xaxis / np.linalg.norm(xaxis)


def vecs2R(vec_x, vec_y):
    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = vec_y / np.linalg.norm(vec_y)
    c = np.dot(vec_x, vec_y)
    s = np.sqrt(1 - c * c) * np.sign(np.cross(vec_x, vec_y))
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return R


def landmarks2S(x, y):
    x_mean = x.mean(axis=0).squeeze()
    y_mean = y.mean(axis=0).squeeze()
    # vectors = mean -> each point
    x_vectors = x - x_mean
    y_vectors = y - y_mean

    x_norms = np.linalg.norm(x_vectors, axis=1)
    y_norms = np.linalg.norm(y_vectors, axis=1)

    indices = [96, 97, 76, 82]  # indices for eyes, lips
    scale = (y_norms / x_norms)[indices].mean()

    S = np.eye(3)
    S[0, 0] = S[1, 1] = scale
    return S


def pad_mirror(img, landmarks):
    H, W, _ = img.shape
    img = np.pad(img, ((H//2, H//2), (W//2, W//2), (0, 0)), 'reflect')
    small_blurred = gaussian(cv2.resize(img, (W, H)), H//100, multichannel=True)
    blurred = cv2.resize(small_blurred, (W * 2, H * 2)) * 255

    H, W, _ = img.shape
    coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    weight_y = np.clip(coords[0] / (H//4), 0, 1)
    weight_x = np.clip(coords[1] / (H//4), 0, 1)
    weight_y = np.minimum(weight_y, np.flip(weight_y, axis=0))
    weight_x = np.minimum(weight_x, np.flip(weight_x, axis=1))
    weight = np.expand_dims(np.minimum(weight_y, weight_x), 2)**4
    img = img * weight + blurred * (1 - weight)
    landmarks += np.array([W//4, H//4])
    return img, landmarks


def align_faces(args, input_dir, output_dir):
    import os
    from torchvision import transforms
    from PIL import Image
    from core.utils import save_image

    aligner = FaceAligner(args.wing_path, args.lm_path, args.img_size)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    fnames = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    fnames.sort()
    for fname in fnames:
        image = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        x = transform(image).unsqueeze(0)
        x_aligned = aligner.align(x)
        save_image(x_aligned, 1, filename=os.path.join(output_dir, fname))
        print('Saved the aligned image to %s...' % fname)


# ========================== #
#   Mask related functions   #
# ========================== #


def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    x = x.contiguous()
    N, C, H, W = x.size()
    x_ = x.view(N*C, -1)
    max_val = torch.max(x_, dim=1, keepdim=True)[0]
    min_val = torch.min(x_, dim=1, keepdim=True)[0]
    x_ = (x_ - min_val) / (max_val - min_val + eps)
    out = x_.view(N, C, H, W)
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)


def resize(x, p=2):
    """Resize heatmaps."""
    return x**p


def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H-N)

    if up:
        head = torch.arange(H-N)+N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H-N)
        tail = torch.arange(H-N)

    # permutation indices
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:, :, perm, :]
    return out


IDXPAIR = namedtuple('IDXPAIR', 'start end')
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple('OPPAIR', 'shift resize')


def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.size()
    x = truncate(x)
    x = normalize(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7*sw, 2),
                       nostrils=OPPAIR(8*sw, 4),
                       lipupper=OPPAIR(-8*sw, 4),
                       liplower=OPPAIR(8*sw, 4),
                       lipinner=OPPAIR(-2*sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = torch.cat([torch.arange(0, index_map.chin.start),
                          torch.arange(index_map.chin.end, 33),
                          torch.LongTensor([index_map.eyebrowsedges.start,
                                            index_map.eyebrowsedges.end,
                                            index_map.lipedges.start,
                                            index_map.lipedges.end])])
    x[:, zero_out] = 0

    start, end = index_map.nose
    x[:, start+1:end] = shift(x[:, start+1:end], 4*sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
        shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51

    x = torch.sum(x, dim=1, keepdim=True)  # (N, 1, H, W)
    x2 = torch.sum(x2, dim=1, keepdim=True)  # mask without faceline and mouth

    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    return x.clamp_(0, 1), x2.clamp_(0, 1)
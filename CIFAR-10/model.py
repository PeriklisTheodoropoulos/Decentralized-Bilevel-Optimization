import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader , Dataset
import warnings
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data.sampler import SubsetRandomSampler
import math
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CNNCifar10(nn.Module):
    """
    ResNet-8 for CIFAR-10 / federated non-IID settings.
    Architecture: 3×3 stem (16ch) → 3 residual blocks (16→32→64) → GAP → FC
    ~78k parameters. No unused kwargs.
    """
    def __init__(self, n_class=10, device='cpu'):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock(16, 16, stride=1)   # 32×32
        self.layer2 = ResBlock(16, 32, stride=2)   # 16×16
        self.layer3 = ResBlock(32, 64, stride=2)   #  8×8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)          # ← added
        self.fc  = nn.Linear(64, n_class)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)   
        return self.fc(x)

class ResBlockTanh(nn.Module):
    """Basic residual block with optional projection shortcut — Tanh activations."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.tanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.tanh(out)          # tanh after residual add


class CNNCifar10Tanh(nn.Module):
    """
    ResNet-8 for CIFAR-10 with Tanh activations.
    Architecture: 3×3 stem (16ch) → 3 residual blocks (16→32→64) → GAP → FC
    ~78k parameters.
    """
    def __init__(self, n_class=10, device='cpu'):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        self.layer1 = ResBlockTanh(16, 16, stride=1)   # 32×32
        self.layer2 = ResBlockTanh(16, 32, stride=2)   # 16×16
        self.layer3 = ResBlockTanh(32, 64, stride=2)   #  8×8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)          # ← added
        self.fc  = nn.Linear(64, n_class)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier is better than Kaiming for Tanh
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)                             # ← added
        return self.fc(x)


class ResBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class CNNCifar10Wide(nn.Module):
    """
    ResNet-8 Wide for CIFAR-10.
    Architecture: 3×3 stem (32ch) → 3 residual blocks (32→64→128) → GAP → FC
    ~320k parameters.
    """
    def __init__(self, n_class=10, device='cpu'):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock(32,  32,  stride=1)   # 32×32
        self.layer2 = ResBlock(32,  64,  stride=2)   # 16×16
        self.layer3 = ResBlock(64,  128, stride=2)   #  8×8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, n_class)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNCifar10Deep(nn.Module):
    """
    ResNet-10 for CIFAR-10.
    Architecture: stem (32ch) → 4 residual blocks (32→64→128→128) → GAP → FC
    ~420k parameters.
    """
    def __init__(self, n_class=10, device='cpu'):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock(32,  64,  stride=1)   # 32×32
        self.layer2 = ResBlock(64,  128, stride=2)   # 16×16
        self.layer3 = ResBlock(128, 128, stride=2)   #  8×8
        self.layer4 = ResBlock(128, 256, stride=2)   #  4×4
        # self.layer4 = ResBlock(128, 128, stride=2)   #  4×4

        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop_fc = nn.Dropout(p=0.3)
        self.fc  = nn.Linear(256, n_class)
        # self.fc = nn.Linear(128, n_class)             # update to match new channel width


        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.drop3( self.layer3(x) )
        x = self.drop4( self.layer4(x) )
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop_fc( x )
        return self.fc(x)

class CNNCifar10BN2(nn.Module):
    """
    ResNet-14 style for CIFAR-10.
    Architecture: stem (64ch) → 3 stages × 2 ResBlocks (64→128→256) → GAP → Dropout → FC
    ~480k parameters.
    """
    def __init__(self, n_class=10, dropout=0.3, device='cpu'):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1 — 64ch, stride=1 (32×32)
        self.stage1 = nn.Sequential(
            ResBlock(64,  64,  stride=1),
            ResBlock(64,  64,  stride=1),
        )
        # Stage 2 — 128ch, stride=2 (16×16)
        self.stage2 = nn.Sequential(
            ResBlock(64,  128, stride=2),
            ResBlock(128, 128, stride=1),
        )
        # Stage 3 — 256ch, stride=2 (8×8)
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
        )

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, n_class)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.dropout(x.view(x.size(0), -1))
        return self.fc(x)
    

# ─────────────────────────────────────────────────────────────
# Helper used exclusively by CNNCifar10ResNet9
# ─────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, pool=False):
    """Conv → BN → ReLU block, with optional MaxPool."""
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CNNCifar10ResNet9(nn.Module):
    """
    ResNet-9 for CIFAR-10 (DAWNBench-style).
    Architecture:
        prep  : Conv(3→64)  + BN + ReLU                          — 32×32
        layer1: Conv(64→128) + BN + ReLU + MaxPool  + ResBlock   — 16×16
        layer2: Conv(128→256)+ BN + ReLU + MaxPool               —  8×8
        layer3: Conv(256→512)+ BN + ReLU + MaxPool  + ResBlock   —  4×4
        head  : MaxPool(4) → Flatten → FC(512→n_class)
    ~6.6M parameters. Reaches ~93 % on CIFAR-10.
    """
    def __init__(self, n_class=10, dropout=0.2, device='cpu'):
        super().__init__()

        # Preparation layer
        self.prep = _conv_bn_relu(3, 64)

        # Layer 1 with residual
        self.layer1_head = _conv_bn_relu(64, 128, pool=True)
        self.layer1_res  = nn.Sequential(
            _conv_bn_relu(128, 128),
            _conv_bn_relu(128, 128),
        )

        # Layer 2 — no residual
        self.layer2 = _conv_bn_relu(128, 256, pool=True)

        # Layer 3 with residual
        self.layer3_head = _conv_bn_relu(256, 512, pool=True)
        self.layer3_res  = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )

        self.pool    = nn.MaxPool2d(4)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512, n_class)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prep(x)                          # 32×32, 64ch

        x = self.layer1_head(x)                   # 16×16, 128ch
        x = x + self.layer1_res(x)               # residual add

        x = self.layer2(x)                        #  8×8, 256ch

        x = self.layer3_head(x)                   #  4×4, 512ch
        x = x + self.layer3_res(x)               # residual add

        x = self.pool(x)                          #  1×1, 512ch
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
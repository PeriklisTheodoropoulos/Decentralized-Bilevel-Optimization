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

class CNNFashion_Mnist(nn.Module):
    def __init__(self, n_class, conv_size, kernel_size, hidden_size, padding, device):
        super(CNNFashion_Mnist, self).__init__()
        # 1. Switch to Tanh for bounded activations and better stability in [-1, 1] range
        self.activation = torch.tanh
        # self.activation = torch.relu 
        
        self.layer1 = nn.Conv2d(1, conv_size[0], kernel_size=kernel_size, padding=padding)
        self.layer2 = nn.Conv2d(conv_size[0], conv_size[1], kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2)     
        self.fc1 = nn.Linear(conv_size[1] * 7 * 7, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_class)

        # 2. Apply Xavier Initialization (Better for Tanh than Kaiming/He)
        # This prevents the initial weights from being too large, helping avoid the 10% plateau
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.requires_grad_(True)
        self.to(device)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x) # Logits for CrossEntropyLoss
        
        return x

class Mnist_model(nn.Module):
  def __init__(self, n_class, hidden_size, device):
      super(Mnist_model,  self).__init__()
      self.n_class = n_class
      self.device = device
      self.fc1 = nn.Linear(28*28, hidden_size[0])
      self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
      self.fc3 = nn.Linear(hidden_size[1],n_class)
      self.activation = F.leaky_relu
      self.requires_grad_(True) 
      self.to(self.device)

  def forward(self, x):
      x = torch.flatten(x,start_dim=1)
      x = self.activation(self.fc1(x))
      x = self.activation(self.fc2(x))
      x = self.fc3(x)
      return x
   
import numpy as np
import pickle as pkl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision
import torchvision.models as models

class TomoModel(nn.Module):
    def __init__(self):
        super(TomoModel, self).__init__()
        num_classes = 10
        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(2208, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output

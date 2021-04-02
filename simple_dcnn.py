import torch
import torch.nn as nn 
import torch.nn.functional as F 

from DConv import DConv2d

class DCNN_model(nn.Module):
    def __init__(self):
        super(DCNN_model, self).__init__()

        self.dcnn_model = nn.Sequential(
            DConv2d(3, 32, 3),
            nn.ReLU(),
            DConv2d(32, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.dcnn_model(x)
        return logits
from torch import nn
import torch.nn.functional as functional


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3), padding=1)
        
    def forward(self, x):
        pass

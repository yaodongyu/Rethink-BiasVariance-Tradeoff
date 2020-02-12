import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, width=1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, width, bias=False)
        self.fc2 = nn.Linear(width, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(torch.flatten(x, 1))
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):

    def __init__(self, num_action, hiden_size = 512):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 4, 2)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.mp3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64, hiden_size)
        self.fc2 = nn.Linear(hiden_size, 20)
        self.fc3 = nn.Linear(20, num_action)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = F.relu(self.mp3(self.conv3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

inp = torch.randn(32,1,80,80)
dqn = DQN(3)
print(dqn.forward(inp))
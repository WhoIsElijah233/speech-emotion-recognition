# # model
# import torch
# import torch.nn as nn

# class EmotionModel(nn.Module):
#   def __init__(self):
#     super().__init__()

#     self.fc1 = nn.Linear(16000, 128)

#     self.relu = nn.ReLU()

#     self.fc2 = nn.Linear(128, 4)

#   def forward(self, x):
#     x = self.fc1(x)

#     x = self.relu(x)

#     x = self.fc2(x)

#     return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):

    def __init__(self, num_classes=8):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 20 * 75, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

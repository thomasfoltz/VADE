import torch.nn as nn

class VADE_CNN(nn.Module):
    def __init__(self, feature_dim):
        super(VADE_CNN, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, 256, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x.squeeze(-1).T

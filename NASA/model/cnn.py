import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2),  # 16 - 2 + 1 = 15
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 15 - 2 + 1 = 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),  # 14 - 2 + 1 = 13
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 13 - 2 + 1 = 12
        )
        self.Linear1 = nn.Linear(128 * 12, 50)
        self.Linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.size())    # 15, 128, 12
        x = x.view(x.size(0), -1)
        # print(x.size())    # 15, 1536
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        x = x.view(x.shape[0], -1)

        return x
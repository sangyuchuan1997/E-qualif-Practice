import logging
import torch
import torch.nn as nn


class CustomNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.fc1(x)
        return x

def main():
    pass

if __name__ == "__main__":
    main()
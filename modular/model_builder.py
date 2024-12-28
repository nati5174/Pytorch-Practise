import torch
from torch import nn

class fashionMINSTv3(nn.Module):
  def __init__(self, input_shape, hidden_layers, out_shape):
    super().__init__()
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_layers,kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers,kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=(hidden_layers*7*7), out_features=out_shape)
    )

  def forward(self, x):
    x = self.block_1(x)
    x = self.block_2(x)
    x = self.classifier(x)
    return x
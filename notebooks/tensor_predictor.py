import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

writer = SummaryWriter('data/tensorboard/draft')

# Data
writer.add_scalar(
    tag="test",
    scalar_value=500
)

# Model

class OrganelleRegressor(nn.Module):
    def __init__(self,n_input):
        """
        output = Sum_i(A_i*x_i)+Sum_ij(x_i*B_ij*x_j)+C 
        """
        super().__init__()
        self.n_input = n_input
        self.A = nn.Parameter(torch.randn(self.n_input))
        self.B = nn.Parameter(torch.randn((self.n_input,self.n_input)))
        self.C = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return self.A @ x + x @ self.B @ x + self.C

class OrganelleClassifier(nn.Module):
    def __init__(self,n_input,n_output) -> None:
        super().__init__()
        self.n_input  = n_input
        self.n_output = n_output
        self.A = nn.Parameter((torch.randn(self.n_output),torch.randn(self.n_input)))
        self.B = nn.Parameter(torch.randn((self.n_input,torch.randn(self.n_output),self.n_input)))
        self.C = nn.Parameter(torch.zeros(self.n_output))

    def forward(self,x):
        return self.A @ x + x @ self.B @ x + self.C

model = OrganelleRegressor(6)
writer.add_graph(model)

# Pipeline

# def __main__():

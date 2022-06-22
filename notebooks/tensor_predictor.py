import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from organelle_measure.data import read_results


# Chores: Using GPU, TensorBoard
print("Whether CUDA is available: ",torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

writer = SummaryWriter('data/tensorboard/draft')


# Data
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
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
        self.A = nn.Parameter((torch.randn(self.n_output,self.n_input)))
        self.B = nn.Parameter(torch.randn((self.n_input,self.n_output,self.n_input)))
        self.C = nn.Parameter(torch.zeros(self.n_output))

    def forward(self,x):
        return self.A @ x + x @ self.B @ x + self.C

model = OrganelleClassifier(6,4) # had bugs
writer.add_graph(model,input_to_model=torch.zeros(6))


# Pipeline

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
    return None


# def __main__():

# hyperparameters
learning_rate = 0.1
epochs = 2
batch_size = 64
training_type = "regress"
# training_type = "classify"

# data
px_x,px_y,px_z = 0.41,0.41,0.20
organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]
subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine",
    "EYrainbowWhi5Up_betaEstrodiol"
]
folder_i = Path("./data/results")

df_bycell = read_results(folder_i,subfolders,(px_x,px_y,px_z))
x_train, y_train, x_valid, y_valid = 0,0,0,0

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
train_dataloader, valid_dataloader = get_data(train_dataset, valid_dataset, batch_size)

# model
model = OrganelleRegressor(n_input=6)
model.to(dev)
loss_function = F.cross_entropy if training_type=="classify" else F.mse_loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9)

# training
fit(epochs, model, loss_function, optimizer, train_dataloader, valid_dataloader)

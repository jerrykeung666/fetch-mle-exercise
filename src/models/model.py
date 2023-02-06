import torch
import torch.nn as nn

class Model(nn.Module):
    "Model"

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
 
    def forward(self, x):
        y_pred = self.linear(x["date"])
        return y_pred
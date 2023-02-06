import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    "Model"

    def __init__(self,
                 input_size=1,
                 output_size=1,
                 hidden_dim=32,
                 n_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.encoder = nn.Linear(1, 8)
        self.decoder = nn.Linear(hidden_dim+8, output_size)

    def forward(self, x):
        x1, _ = self.rnn(x["history"].unsqueeze(1))
        x2 = self.encoder(x["date"])
        x = torch.cat((x1[-1, :, :].squeeze(), x2), 0)
        # return self.rnn(x)
        return self.decoder(x)


# DEBUG
if __name__ == "__main__":
    model = Model()
    a = torch.randn(3, 5, 1)
    res = model(a)
    print(len(res))
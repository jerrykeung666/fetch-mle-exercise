import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset.dataset import FetchDataset
from models.model import Model

def train(args):
    # basic param
    time_length = args.time_length
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset
    dataset = FetchDataset(args.path, time_length, device)

    # create model
    model = Model()
    model.to(device)

    # optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-12)
    # optimizer = optim.Adam(model.parameters(), lr=10, betas=(0.9, 0.999), eps=1e-12)

    # loss
    mse = nn.MSELoss()

    # train
    for epoch in range(epochs):
        loss = 0
        for idx in tqdm(range(time_length, len(dataset))):
            estimate = model(dataset[idx])
            loss += mse(dataset[idx]["value"], estimate)
        loss /= (len(dataset) - time_length)
        
        if epoch % 50 == 0:
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_params = {"model": model.state_dict()}
    torch.save(save_params, os.path.join(model_dir, "model_latest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/wei/codes/fetch-mle-exercise/data/data_daily.csv")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--time_length', type=int, default=28)
    args = parser.parse_args()

    train(args)
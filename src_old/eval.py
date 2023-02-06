import os
import argparse
from tracemalloc import Snapshot

import numpy as np
import torch
import torch.nn as nn

from dataset.dataset import FetchDataset
from models.model import Model


def eval(args):
    # basic param
    model_path = args.model_path
    path = args.path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_length = args.time_length

    # create dataset
    dataset = FetchDataset(path, time_length, device, datatype="test") 

    # load trained model
    model = Model()
    snapshot = torch.load(model_path)["model"]
    model.load_state_dict(snapshot)
    model.to(device)

    # generate results of next year (each day)
    res = []
    curr_data = dataset[-1]
    for day in range(365):
        val = model(curr_data)
        res.append(val)

        # update curr_data
        history = torch.zeros(time_length, 1).to(device)
        history[:-1] = curr_data["history"][1:]
        history[-1] = val
        curr_data = {"date": torch.Tensor([day+1]).to(device),
                     "value": 0,
                     "history": history}

    # aggregate to get results of each month
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/wei/codes/fetch-mle-exercise/data/data_daily.csv")
    parser.add_argument('--model_path', type=str, default="/home/wei/codes/fetch-mle-exercise/results/model_latest.pth")
    parser.add_argument('--time_length', type=int, default=28)
    args = parser.parse_args()

    eval(args)
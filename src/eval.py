import os
import argparse
from tracemalloc import Snapshot

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dataset.dataset import FetchDataset
from models.model import Model
from util.util import int2date

scale = 1e7

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
    res_day = []
    res_month = []
    date = []
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
             "Aug", "Sept", "Oct", "Nov", "Dec"]
    curr_data = dataset[-1]
    for day in range(365, 730):
        val = model(curr_data)
        date.append(int2date(2022, day-365))
        res_day.append((val*scale).detach().cpu().numpy())

        # month
        if day == 395:
            res_month.append(sum(res_day))
        elif day in [423, 454, 484, 515, 545, 576, 607, 637, 668, 698, 729]:
            res_month.append(sum(res_day)-res_month[-1])

        # update curr_data
        history = torch.zeros(time_length, 1).to(device)
        history[:-1] = curr_data["history"][1:]
        history[-1] = val
        curr_data = {"date": torch.Tensor([day]).to(device),
                     "value": 0,
                     "history": history}

    # aggregate to get results of each day/month & vis results
    plt.scatter(date, res_day)
    plt.savefig(os.path.join(os.getcwd(), "results", "estimated_number_day.png"))
    plt.clf()
    
    plt.scatter(month, res_month)
    plt.savefig(os.path.join(os.getcwd(), "results", "estimated_number_month.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/wei/codes/fetch-mle-exercise/data/data_daily.csv")
    parser.add_argument('--model_path', type=str, default="/home/wei/codes/fetch-mle-exercise/results/model_latest.pth")
    parser.add_argument('--time_length', type=int, default=28)
    args = parser.parse_args()

    eval(args)
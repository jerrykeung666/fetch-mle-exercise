import os

import numpy as np
import torch
from torch.utils.data import Dataset

class FetchDataset(Dataset):
    """Fetch Script Dataset"""

    def __init__(self, data_path, time_length, device, datatype="train"):
        super().__init__()
        
        # load in data
        scale = 1e7
        with open(data_path, 'r') as f:
            data = f.readlines()[1:]
        
        # preprocess data: (idx, value, history)
        self.data = []
        if datatype == "train":
            for idx, elem in enumerate(data):
                history = torch.zeros(time_length, 1).to(device)
                if idx < time_length:
                    continue
                elif idx == time_length:
                    for index in range(time_length):
                        history[index] = torch.Tensor([float(data[index].split(',')[-1][:-1])]).to(device)/scale
                else:
                    history[:-1] = self.data[-1]["history"][1:]
                    history[-1] = self.data[-1]["value"]

                curr_data = {"date": torch.Tensor([idx]).to(device),
                             "value": torch.Tensor([float(data[idx].split(',')[-1][:-1])]).to(device)/scale,
                             "history": history}
                self.data.append(curr_data)
        elif datatype == "test":
            history = torch.zeros(time_length, 1).to(device)
            for index in range(time_length):
                history[index] = torch.Tensor([float(data[-index].split(',')[-1][:-1])]).to(device)/scale
            curr_data = {"date": torch.Tensor([len(data)]).to(device),
                         "value": 0,
                         "history": history}
            self.data.append(curr_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# DEBUG
if __name__ == "__main__":
    dataset = FetchDataset("/home/wei/codes/fetch-mle-exercise/data/data_daily.csv", 28)
    print(dataset.data[-1]["history"].shape)
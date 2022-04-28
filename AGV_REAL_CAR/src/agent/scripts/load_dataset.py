import os
import sys
pathx = os.path.abspath(".")
sys.path.insert(0,pathx + "/src/agent/scripts")
import torch
from torch.utils.data import DataLoader, TensorDataset

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        context = f.readlines()
        for line in context:
            line = line.strip('\n')
            sep_str = line.split(',')
            temp = []
            for i in range(len(sep_str)):
                temp.append(float(sep_str[i]))
            data.append(temp)        
    return data


def generate_train_dataset(data):
    train_x = data[:-1]
    train_y = data[1:]
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)[:, 0:3]
    
    batch_size = train_x.shape[0]
    
    train_dataset = TensorDataset(train_x, train_y)
    train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    for i in train_data:
        x, y = i
    
    return x, y


def load_dataset(filename):
    data = read_data(filename)
    train_x, train_y = generate_train_dataset(data)
    return train_x, train_y

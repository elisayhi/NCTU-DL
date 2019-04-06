import sys
import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from model import *
from plot import plot
from dataloader import read_bci_data

def train(model, loss_func, loader, device, optimizer):
    """
    training phase of model, per epoch
    model: model to train
    loader: output of Dataloader
    LR: learning rate of optimizer
    """
    model.train()
    train_loss = 0
    pred = []
    for step, (x, y) in enumerate(loader):
        x = x.float()   # x shape: (batch_size, 1 layer, 2 channels, attrs) = (512, 1, 2, 720)
        y = y.long()
        optimizer.zero_grad()
        output = model(x.to(device))
        pred += output.cpu().tolist()
        loss = loss_func(output, y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()
    return train_loss, pred

def test(model, loss_func, dataloader, device):
    """
    testing phase of model, per epoch
    model: model to predict
    data: testing data
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.float()
            target = target.long()
            output = model(data.to(device))
            loss += loss_func(output, target.to(device)).cpu().item()
    return loss, output

def accuracy(true, pred):
    pred = [1 if i[1]>i[0] else 0 for i in pred]
    same = np.sum(np.array(true) == np.array(pred))
    acc = same/len(true)*100
    return acc

def train_eval(MODEL, name):
    LR = 0.05
    EPOCH = 2000
    batch_size = 512
    device = torch.device('cuda:1')
    # load data
    tr_X, tr_y, ts_X, ts_y = read_bci_data() 
    tr_X, tr_y, ts_X, ts_y = torch.from_numpy(tr_X), torch.from_numpy(tr_y), torch.from_numpy(ts_X), torch.from_numpy(ts_y)
    train_dataloader = Data.DataLoader(Data.TensorDataset(tr_X, tr_y), batch_size=batch_size)
    test_dataloader = Data.DataLoader(Data.TensorDataset(ts_X, ts_y), batch_size=len(ts_y))

    loss_func = nn.CrossEntropyLoss()
    model = MODEL
    model = model.to(device)
    print(model)
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma = 0.8)
    for epoch in range(EPOCH):
        schedular.step()
        tr_loss, tr_pred = train(model, loss_func, train_dataloader, device, optimizer)
        ts_loss, ts_pred = test(model, loss_func, test_dataloader, device)
        train_loss.append(tr_loss)
        test_loss.append(ts_loss)
        train_acc.append(accuracy(tr_y, tr_pred))
        test_acc.append(accuracy(ts_y, ts_pred))

        if epoch%100 == 0:
            print(f'[epoch {epoch}]\ttrain loss: {train_loss[-1]}\ttrain acc: {train_acc[-1]}%', end='\t')
            print(f'test loss: {test_loss[-1]}\ttest acc: {test_acc[-1]}%')
        torch.cuda.empty_cache()
    plot([train_loss, test_loss, train_acc, test_acc], ['train loss', 'test loss', 'train acc', 'test acc'], 'result/'+name+'.png')

if __name__ == '__main__':
    if sys.argv[1] == 'EEG':
        train_eval(EEGNet(), 'EEG')
    elif sys.argv[1] == 'deepc':
        train_eval(DeepConvNet(), 'DeepConvNet')

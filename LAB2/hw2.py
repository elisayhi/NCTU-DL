import sys
import copy
import torch
import pickle
import numpy as np
import torch.utils.data as Data
from torch import nn
from argparse import ArgumentParser
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

def parser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='MODEL')
    parser.add_argument('-a', '--activate', dest='activate')
    parser.add_argument('-gpu', dest='gpu', default='1')
    parser.add_argument('-l', '--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('-e', '--epoch', dest='EPOCH', default=2000, type=int)
    parser.add_argument('-b', '--bsize', dest='BSIZE', default=512, type=int)
    # scheduler
    parser.add_argument('-s', help='use scheduler or not', dest='scheduler', default=1, type=int)
    parser.add_argument('-ms', '--milestones', dest='MILESTONES', default=[750, 1000, 1250], type=list)
    parser.add_argument('-g', '-gamma', dest='GAMMA', default=0.5, type=float)

    args = parser.parse_args()
    #name = f'{args.MODEL}_{args.activate}_{args.lr}_{args.EPOCH}_{args.BSIZE}_{args.scheduler}_{args.MILESTONES}_{args.GAMMA}'
    name = f'{args.MODEL}_{args.activate}_{args.lr}_2000_{args.BSIZE}_{args.scheduler}_{args.MILESTONES}_{args.GAMMA}'
    
    return args, name

if __name__ == '__main__':
    # parameters
    lr_step = 200
    args, name = parser()
    LR = args.lr
    EPOCH = args.EPOCH
    #BSIZE = args.BSIZE
    BSIZE = 2000
    GAMMA = args.GAMMA
    MILESTONES = [int(ms) for ms in args.MILESTONES]
    print(name)

    device = torch.device(f'cuda:{args.gpu}')

    # load data
    tr_X, tr_y, ts_X, ts_y = read_bci_data() 
    tr_X, tr_y, ts_X, ts_y = torch.from_numpy(tr_X), torch.from_numpy(tr_y), torch.from_numpy(ts_X), torch.from_numpy(ts_y)
    train_dataloader = Data.DataLoader(Data.TensorDataset(tr_X, tr_y), batch_size=BSIZE)
    test_dataloader = Data.DataLoader(Data.TensorDataset(ts_X, ts_y), batch_size=len(ts_y))

    # model
    loss_func = nn.CrossEntropyLoss()
    if args.MODEL == 'EEG':
        model = EEGNet(args.activate)
    elif args.MODEL == 'deep':
        model = DeepConvNet(args.activate)
    model = model.to(device)
    #print(model)

    weight = model.state_dict()
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma = GAMMA)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma = GAMMA)
    for epoch in range(EPOCH+1):
        if args.scheduler:
            schedular.step()
        tr_loss, tr_pred = train(model, loss_func, train_dataloader, device, optimizer)
        ts_loss, ts_pred = test(model, loss_func, test_dataloader, device)
        train_loss.append(tr_loss)
        test_loss.append(ts_loss)
        train_acc.append(accuracy(tr_y, tr_pred))
        test_acc.append(accuracy(ts_y, ts_pred))
        if epoch > 3 and test_acc[-1] > test_acc[-2]:
            weight = model.state_dict()

        #if epoch%500 == 0:
        #    print(f'[epoch {epoch}]\ttrain loss: {train_loss[-1]:3.3f}\ttrain acc: {train_acc[-1]:3.3f}%', end='\t')
        #    print(f'test loss: {test_loss[-1]:3.3f}\ttest acc: {test_acc[-1]:3.3f}%')
        torch.cuda.empty_cache()

    #plot([train_acc, test_acc], ['train acc', 'test acc'], 'result/'+name+'.png')
    with open(f'result/{name}.pickle', 'wb') as f:
        pickle.dump({'model':model, 'LR':LR, 'EPOCH':EPOCH, 
            'train_loss':train_loss, 'test_loss':test_loss, 
            'train_acc':train_acc, 'test_acc':test_acc, 
            'lr_step':lr_step, 'MILESTONES':MILESTONES, 'scheduler': args.scheduler}
            , f)
    with open(f'result/{name}_weight.pickle', 'wb') as f:
        pickle.dump({'model_weight':weight, 'best_acc':max(test_acc)}, f)
    print(f'max test acc: {max(test_acc)}')

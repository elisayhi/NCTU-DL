import pickle
import numpy as np
import torch.utils.data as Data
from torch import nn
from resNet import *
from models import ResNet as pResNet
from argparse import ArgumentParser
from dataloader import RetinopathyLoader

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
    true = []
    for step, (x, y) in enumerate(loader):
        x = x.float()   # x shape: (batch_size, 1 layer, 2 channels, attrs) = (512, 1, 2, 720)
        y = y.long()
        #print(x.size(), y.size())
        optimizer.zero_grad()
        output = model(x.to(device))
        pred += output.cpu().tolist()
        true += y.cpu().tolist()
        loss = loss_func(output, y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()
    acc = accuracy(pred, true)
    return train_loss, pred, acc

def test(model, loss_func, dataloader, device):
    """
    testing phase of model, per epoch
    model: model to predict
    data: testing data
    """
    model.eval()
    loss = 0
    pred = []
    true = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.float()
            target = target.long()
            output = model(data.to(device))
            pred += output.cpu().tolist()
            true += target.cpu().tolist()
            #print(f'ts_pred {output} ts_true {target}')
            loss += loss_func(output, target.to(device)).cpu().item()
    acc = accuracy(pred, true)
    return loss, acc, true, pred

def accuracy(preds, true):
    pred_label = [np.argmax(pred) for pred in preds]
    same = np.sum(np.array(true) == np.array(pred_label))
    acc = same/len(true)*100
    return acc

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', dest='model', default=18)
    parser.add_argument('-p', help='use pertrain or not', dest='pretrain', default=0, type=int)
    args = parser.parse_args()
    model = args.model
    # parameter
    EPOCH = 10
    device = torch.device('cuda:1')

    # load data
    train_dataloader = Data.DataLoader(RetinopathyLoader('data', 'train'), batch_size=4)
    test_dataloader = Data.DataLoader(RetinopathyLoader('data', 'test'), batch_size=4)

    # model
    loss_func = nn.CrossEntropyLoss()
    if not args.pretrain:
        if model == '18':
            model = ResNet(BasicBlock, [2,2,2,2]).to(device)
        else:
            model = ResNet(BottleNeck, [3,4,23,3]).to(device)
    else:
        model = pResNet(int(model)).to(device)
    print(model)

    train_acc, test_acc = [], []
    ts_pred, ts_true = None, None
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
    for epoch in range(EPOCH+1):
        train_loss, tr_pred, tr_acc = train(model, loss_func, train_dataloader, device, optimizer)
        test_loss, ts_acc, ts_pred, ts_true = test(model, loss_func, test_dataloader, device)
        print(f'train loss {train_loss} test loss {test_loss}')
        print(f'train acc {tr_acc} test acc {ts_acc}')
        train_acc.append(tr_acc)
        test_acc.append(ts_acc)
        torch.cuda.empty_cache()

    with open(f'result/{args.model}_{args.pretrain}_output.pickle', 'wb') as f:
        pickle.dump({'pred': ts_pred, 'true': ts_true,
            'train_acc': train_acc, 'test_acc': test_acc}, f)
    with open(f'result/{args.model}_{args.pretrain}_weight.pickle', 'wb') as f:
        pickle.dump(model.state_dict(), f)
        


import pickle
import torchvision.models
import numpy as np
import torch.utils.data as Data
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from torch import nn
from resNet import *
from models import ResNet as pResNet
from argparse import ArgumentParser
from torchvision import transforms
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
    for step, (x, y) in enumerate(tqdm(loader)):
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
    return train_loss, acc

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
    parser.add_argument('-b', dest='bsize', default=8, type=int)
    parser.add_argument('-l', dest='lr', default=5e-3, type=float)
    parser.add_argument('--load', dest='path', default=None)
    args = parser.parse_args()
    # parameter
    EPOCH = 10
    lr = args.lr
    BSIZE = args.bsize
    device = torch.device('cuda:1')
    name = f'{args.model}_{args.pretrain}_{args.bsize}'

    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomVerticalFlip(p=0.5),])
    train_dataloader = Data.DataLoader(RetinopathyLoader('data', 'train', transform=transform), batch_size=BSIZE)
    test_dataloader = Data.DataLoader(RetinopathyLoader('data', 'test'), batch_size=BSIZE)

    # model
    loss_func = nn.CrossEntropyLoss()
    if not args.pretrain:
        if args.model == '18':
            model = ResNet(BasicBlock, [2,2,2,2])
        else:
            model = ResNet(BottleNeck, [3,4,23,3])
            EPOCH = 5
    else:
        model = torchvision.models.__dict__['resnet{}'.format(int(args.model))](pretrained=True)
        # replace the last 2 layer in the pretrained model to match the dataset
        if args.model == '18':
            model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512, 5)
        else:
            model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(2048, 5)
            EPOCH = 5
    if args.path:
        with open(args.path, 'rb') as f:
            model.load_state_dict(pickle.load(f))

    model = model.to(device)
    print(model)

    train_acc, test_acc = [], []
    ts_pred, ts_true = None, None
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    for epoch in range(1, EPOCH+1):
        train_loss, tr_acc = train(model, loss_func, train_dataloader, device, optimizer)
        test_loss, ts_acc, ts_true, ts_pred = test(model, loss_func, test_dataloader, device)
        print(f'train loss {train_loss} test loss {test_loss}')
        print(f'train acc {tr_acc} test acc {ts_acc}')
        train_acc.append(tr_acc)
        test_acc.append(ts_acc)
        torch.cuda.empty_cache()

        with open(f'result/{name}_output.pickle', 'wb') as f:
            pickle.dump({'pred': ts_pred, 'true': ts_true,
                'train_acc': train_acc, 'test_acc': test_acc}, f)
        if len(train_acc) < 2 or test_acc[-1] > test_acc[-2]:
            with open(f'result/{name}_{test_acc[-1]}_weight.pickle', 'wb') as f:
                pickle.dump(model.state_dict(), f)
    with open(f'result/{name}_weight_final.pickle', 'wb') as f:
        pickle.dump(model.state_dict(), f)
            


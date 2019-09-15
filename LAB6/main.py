import torch
import pickle
import numpy as np
from argparse import ArgumentParser
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from model import *
from train import *

device = torch.device('cuda:1')

def gen_by_random_noise(G, idx):
    pic_cnt = 10
    noise = torch.FloatTensor(pic_cnt, 54).uniform_(-1., 1.).to(device)
    #idx = np.resize(np.arange(10), 10*10)
    #idx = np.array([1, 7, 6, 0, 4, 3, 2, 9, 8, 5])
    idx = np.array(idx).repeat(10)
    onehot = np.zeros((pic_cnt, 10))
    onehot[range(pic_cnt), idx] = 1
    z = torch.cat([noise, torch.FloatTensor(onehot).to(device)], 1).view(-1, 64, 1, 1)
    #print(z[0])
    #print(noise.size(), idx.shape, z.size())
    G_result = G(z)
    save_image(G_result.data, 'result/reconstruct.png', nrow=1, normalize=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', dest='fname', default=None)
    parser.add_argument('-n', dest='noise', default=False, type=bool)
    parser.add_argument('-i', dest='idx', default=0, type=int)
    args = parser.parse_args()

    D = Discriminator_C()
    G = Generator()
    Q = Discriminator_Q()
    FE = Discriminator_F()
    device = torch.device('cuda:1')
    #print(G)
    #print(FE)
    #print(D)
    #print(Q)

    for i in [D, Q, G ,FE]:
        i.to(device)
        i.apply(weights_init)

    if args.fname:
        with open(args.fname, 'rb') as f:
            weights = pickle.load(f)
        D.load_state_dict(weights['D'])
        FE.load_state_dict(weights['FE'])
        Q.load_state_dict(weights['Q'])
        G.load_state_dict(weights['G'])

    if args.noise:
        gen_by_random_noise(G, args.idx)
    else:
        trainer = Trainer(G, FE, D, Q)
        trainer.train()



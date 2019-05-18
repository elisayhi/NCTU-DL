import torch
from model import *
from train import *

if __name__ == '__main__':
    D = Discriminator_C()
    G = Generator()
    Q = Discriminator_Q()
    FE = Discriminator_F()
    device = torch.device('cuda:1')

    for i in [D, Q, G ,FE]:
        i.to(device)
        i.apply(weights_init)

    trainer = Trainer(G, FE, D, Q)
    trainer.train()

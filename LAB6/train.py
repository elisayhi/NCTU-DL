import torch
import numpy as np

from torch import nn
from torch import optim
from torch import autograd
from torchvision import transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from model import *

device = torch.device('cuda:1')

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:
    def __init__(self, G, FE, D, Q):
        self.noise_dim = 64
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.batch_size = 100

    def _noise_sample(self, dis_c, noise, bs):

        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        #con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c], 1).view(-1, self.noise_dim, 1, 1)

        return z, idx

    def train(self):
        real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).to(device)
        label = torch.FloatTensor(self.batch_size, 1, 1, 1).requires_grad_(False).to(device)
        dis_c = torch.FloatTensor(self.batch_size, 10).to(device)
        #con_c = torch.FloatTensor(self.batch_size, 2).to(device)
        noise = torch.FloatTensor(self.batch_size, 54).to(device)

        criterionD = nn.BCELoss().to(device)
        criterionQ_dis = nn.CrossEntropyLoss().to(device)
        #criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = dset.MNIST('./data', transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, 54).uniform_(-1, 1)

        writer = SummaryWriter()
        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):
                # train by real data
                optimD.zero_grad()

                x, _ = batch_data
                #print('[y]', _.view(100, 1, 1, 1).size(), _[1], _.view(100, 1, 1, 1)[1])

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1, 1, 1)
                dis_c.data.resize_(bs, 10)
                #con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 54)

                real_x.data.copy_(x)
                fe_out = self.FE(real_x)
                probs_real = self.D(fe_out)
                label.fill_(1.)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # train by fake data
                ## Discriminator
                z, idx = self._noise_sample(dis_c, noise, bs)
                fake_x = self.G(z)
                fe_out = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out)    # detach: G and D need to be update respectly
                label.fill_(0.)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake
                optimD.step()   ####### update loss_real and loss_fake?

                ## G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.fill_(1.0)   # set to 1 to updaate G

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                q_logits = q_logits.squeeze()
                target = torch.LongTensor(idx).to(device)
                #target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                #con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss# + con_loss
                G_loss.backward()
                optimG.step()

                # tensorboard
                #writer.add_scalars('result/losses', {'real': loss_real, 'fake': loss_fake, 'reconstruct': reconstruct_loss, 'dis': dis_loss}, num_iters)
                writer.add_scalars(f'result/losses/{epoch}', {'D_loss': D_loss.data.cpu().numpy(), 'G_loss': reconstruct_loss.data.cpu().numpy(),
                    'Q_loss': dis_loss.data.cpu().numpy()}, num_iters)

                if num_iters % 500 == 0:

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                      epoch, num_iters, D_loss.data.cpu().numpy(),
                      G_loss.data.cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    #con_c.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([noise, dis_c], 1).view(-1, self.noise_dim, 1, 1)
                    x_save = self.G(z)
                    save_image(x_save.data, './tmp/c.png', nrow=10)
                    G_result = make_grid(x_save.data, nrow=10, padding=0)
                    writer.add_image('generator_result', G_result, epoch)

                    #con_c.data.copy_(torch.from_numpy(c2))
                    #z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
                    #x_save = self.G(z)
                    #save_image(x_save.data, './tmp/c2.png', nrow=10)

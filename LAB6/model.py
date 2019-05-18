import torch
import torchvision.datasets as dset
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter

class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=64, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator_F(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf=64):
        super(Discriminator_F, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, in_tensor):
        return self.main(in_tensor)

class Discriminator_C(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator_C, self).__init__()
        self.classify = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, in_tensor):
        return self.classify(in_tensor)

class Discriminator_Q(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator_Q, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(ndf*8, 128, 1, bias=False),
            nn.ReLU(),
        )
        self.Q_disc = nn.Conv2d(128, 10, 4, bias=True)
        self.Q_mu = nn.Conv2d(128, 2, 1)
        self.Q_var = nn.Conv2d(128, 2, 1)

    def forward(self, in_tensor):
        output = self.Q(in_tensor)
        mu = self.Q_mu(output)
        var = self.Q_var(output).exp()
        output = self.Q_disc(output)
        return output, mu, var










class Discriminator(nn.Module):
    def __init__(self, mode, ngpu=1, nc=1, ndf=64):
        """
        mode: 'D'(discriminator) or 'Q' or 'F'(main part)
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.mode = mode
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(
            nn.Linear(in_features=8192, out_features=100, bias=True),
            nn.ReLU(),
        )
        self.Q_disc = nn.Linear(in_features=100, out_features=10, bias=True)
        self.Q_mu = nn.Linear(100, 2)
        self.Q_var = nn.Linear(100, 2)

    def forward(self, in_tensor):
        if in_tensor.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, in_tensor, range(self.ngpu))
        else:
            output = self.main(in_tensor)
            if self.mode == 'D':
                output = self.discriminator(output)
                return output
            elif self.mode == 'G':
                output = self.Q(output)
                mu = self.Q_mu(output)
                var = self.Q_var(output).exp()
                output = self.Q_disc(output)
                return output, mu, var
            else:
                print('[class discriminator] wrong mode')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

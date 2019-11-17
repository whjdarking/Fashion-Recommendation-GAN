import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(72, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),
            nn.Sigmoid()
        )
#        self.tconv1 = nn.ConvTranspose2d(74, 1024, 1, 1, bias=False)
#        self.bn1 = nn.BatchNorm2d(1024)
#
#        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
#        self.bn2 = nn.BatchNorm2d(128)
#
#        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(64)
#
#        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
#        x = F.relu(self.bn1(self.tconv1(x)))
#        x = F.relu(self.bn2(self.tconv2(x)))
#        x = F.relu(self.bn3(self.tconv3(x)))
#
#        img = torch.sigmoid(self.tconv4(x))
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
#        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
#
#        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
#        self.bn2 = nn.BatchNorm2d(128)
#
#        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
#        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
#        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
#        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
#        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return self.main(x)

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,1), #1024b不稳定
            nn.Sigmoid()
        )
#        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
#        output = torch.sigmoid(self.conv(x))

        return self.main(x)

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
#class QHead(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.main = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(1024,10), #1024b不稳定
#            nn.sigmoid()
#        )
#
#    def forward(self, x):
#        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
#
#        disc_logits = self.conv_disc(x).squeeze()
#
#        mu = self.conv_mu(x).squeeze()
#        var = torch.exp(self.conv_var(x).squeeze())
#
#        return disc_logits, mu, var
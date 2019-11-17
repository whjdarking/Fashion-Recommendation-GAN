import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
# n_dis_c部分换成GPU
#    idx = np.zeros((n_dis_c, batch_size))
#    idx = torch.zeros(n_dis_c, batch_size, dtype=torch.uint8, device=device)
    if(n_dis_c != 0):
#        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device).to(device)
        idx = torch.randint(0, dis_c_dim, size=(batch_size, n_dis_c), device=device)
        #idx = torch.randint_like(idx, low=0, high=dis_c_dim, device=device).to(device)
        one_hot = torch.nn.functional.one_hot(idx, dis_c_dim).type(torch.FloatTensor).to(device)
#        print(one_hot.size())
#        for i in range(n_dis_c):
#             #从0到dim随机 存在np自动扩展idx[][]第二维
#            print(idx[i].size())
#            dis_c[torch.arange(0, batch_size), i, one_hot] = 1.0
        dis_c = one_hot.view(batch_size, -1, 1, 1)
    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1 #最后部分待修?

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    idx = torch.transpose(idx, 1, 0)
#    print(idx.size())
#    print(idx.is_cuda)
    return noise, idx
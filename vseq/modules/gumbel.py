import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class CVAE(nn.Module):
    def __init__(self, N, K, tau):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, N*K)
        self.fc4 = nn.Linear(N*K, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.N = N
        self.temperature = tau


    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits):
        sample = Variable(self._sample_gumbel(logits.size()[-1]))
        if logits.is_cuda:
            sample = sample.cuda()
        y = logits + sample
        return self.softmax(y / self.temperature)

    def _gumbel_softmax(self, logits, hard=False):
        y = self._gumbel_softmax_sample(logits)
        return y

    def _encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        logits_y = h3.view(-1, self.K)
        q_y = self.softmax(logits_y)
        return q_y, logits_y

    def _decoder(self, y):
        h3 = self.relu(self.fc4(y))
        h4 = self.relu(self.fc5(h3))
        logits_x = self.fc6(h4)
        p_x = self.sigmoid(logits_x)
        return p_x, logits_x

    def forward(self, x):
        q_y, logits_y = self._encoder(x.view(-1, 784))
        y = self._gumbel_softmax(logits_y).view(-1, self.N * self.K)
        p_x, logits = self._decoder(y)
        return p_x, q_y, logits

def loss_fn(q_y, p_x, x, N, K, logits):
    prior = Variable(torch.log(torch.from_numpy(np.array([1.0/K]))).type(torch.FloatTensor)).cuda()
    kl = (q_y * (torch.log(q_y+1e-20) - prior)).view(-1, N, K)
    KL = torch.sum(kl)
    data = x.view(-1, 784)
    bceloss = nn.BCELoss(size_average=False)
    elbo = -bceloss(p_x, data) - KL
    loss = -(1.0/p_x.size()[0])*elbo
    return loss

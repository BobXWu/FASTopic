import torch
from torch import nn
import torch.nn.functional as F
from ._model_utils import pairwise_euclidean_distance


class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha, init_a_dist=None, init_b_dist=None, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        # Sinkhorn's algorithm
        M = pairwise_euclidean_distance(x, y)
        device = M.device

        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp

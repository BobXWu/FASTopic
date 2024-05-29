import torch
from torch import nn
import torch.nn.functional as F

from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance


class fastopic(nn.Module):
    def __init__(self,
                 num_topics: int,
                 DT_alpha: float=3.0,
                 TW_alpha: float=2.0,
                 theta_temp: float=1.0
                ):
        super().__init__()

        self.num_topics = num_topics
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp

    def init(self,
             vocab_size: int,
             doc_embeddings,
            ):

        self.doc_embeddings = nn.Parameter(torch.from_numpy(doc_embeddings), requires_grad=False)

        embed_size = self.doc_embeddings.shape[1]

        self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    @property
    def transp_DT(self):
        _, transp = self.DT_ETP(self.doc_embeddings, self.topic_embeddings)
        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]

        return beta

    # only for testing
    def get_theta(self, doc_embeddings):
        dist = pairwise_euclidean_distance(doc_embeddings, self.topic_embeddings)
        train_dist = pairwise_euclidean_distance(self.doc_embeddings, self.topic_embeddings)

        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)

        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)

        return theta

    def forward(self, train_bow):
        loss_DT, transp_DT = self.DT_ETP(self.doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        # Dual Semantic-relation Reconstruction
        recon = torch.matmul(theta, beta)

        loss_DSR = -(train_bow * recon.log()).sum(axis=1).mean()

        loss = loss_DSR + loss_ETP

        rst_dict = {
            'loss': loss,
        }

        return rst_dict

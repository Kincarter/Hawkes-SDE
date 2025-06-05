import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mlp import MLP


def sample_gumbel(shape, eps=1e-10):

    U = torch.rand(shape).float()

    return -torch.log(eps - torch.log(U + eps))

def get_laplacian(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    L = -D@A@D
    return L

def get_normalized_adjacency(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    A = D@A@D
    return A


def get_edge_prob(logits, gumbel_noise=False, beta=0.5, hard=False):

    if gumbel_noise:
        y = logits + sample_gumbel(logits.size()).to(logits.device)
    else:
        y = logits

    edge_prob_soft = torch.softmax(beta * y, dim=0)

    if hard:
        _, edge_prob_hard = torch.max(edge_prob_soft.data, dim=0)
        edge_prob_hard = F.one_hot(edge_prob_hard)
        edge_prob_hard = edge_prob_hard.permute(1,0)



        edge_prob = edge_prob_hard - edge_prob_soft.data + edge_prob_soft
    else:
        edge_prob = edge_prob_soft
    return edge_prob

class A_MSG(nn.Module):


    def __init__(self, num_vertex, h_dim, hidden_dim, num_hidden, act):
        super(A_MSG, self).__init__()

        self.num_vertex = num_vertex
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        # A model
        self.edge_wise = MLP(dim_in=2 * h_dim, dim_out=1, num_hidden=num_hidden, sigma= 0.01, activation = act)
        self.jump_wise = MLP(dim_in=2 * h_dim, dim_out=h_dim, num_hidden=num_hidden, sigma=0.01, activation=act)
        self.aggregation_wise= MLP(dim_in=2 * h_dim, dim_out=h_dim, num_hidden=num_hidden, sigma=0.01, activation = act)



    def forward(self, A_H, x,seq_mask):


        seq_len = x.shape[1]

        agg_msg_A = A_H.clone()
        agg_msg_A = agg_msg_A.unsqueeze(dim=1).repeat(1,seq_len,1)

        edge_A_H = self.edge_wise(torch.cat((agg_msg_A, x), dim=-1)).squeeze(-1)
        edge_A_H_mask = edge_A_H.masked_fill_(seq_mask,-1e9)

        weight_A_H = torch.softmax((edge_A_H_mask),dim=1)
        weight_A_H = weight_A_H.unsqueeze(dim=2)

        jump_A_H = self.jump_wise(torch.cat((agg_msg_A, x), dim=-1))

        update_A_H = torch.sum(jump_A_H*weight_A_H,dim=1)

        update_A_H = self.aggregation_wise(torch.cat((update_A_H,A_H),dim=-1))
        A_H = A_H + update_A_H
        return A_H,jump_A_H









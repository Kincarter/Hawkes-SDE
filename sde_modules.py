import torch
import torch.nn as nn
from mlp import MLP


class SDEFunc(nn.Module):

    def __init__(self, dim_h, dim_hidden, num_hidden, activation=nn.Tanh()):

        super(SDEFunc, self).__init__()
        self.dim_h = dim_h


        self.self_func = MLP(dim_in = dim_h + 2, dim_out = dim_h, dim_hidden=dim_hidden, num_hidden=num_hidden, sigma=0.01, activation=activation)

        self.lambda_func = MLP(dim_in = dim_h, dim_out = 1, dim_hidden=dim_hidden, num_hidden=0, sigma=0.1, activation=activation)

        self.diffusion_net = MLP(dim_in = dim_h, dim_out = dim_h, dim_hidden = dim_hidden, num_hidden=num_hidden, sigma=0.01, activation=nn.Tanh())

        self.jump_func = MLP(dim_in = dim_h, dim_out = dim_h, dim_hidden = dim_hidden, num_hidden=num_hidden, sigma=0.1, activation=activation)

    def g(self, H, t):
        return self.diffusion_net(H)


    def f(self, H, t_diff, last_t):
        H_ = torch.cat((H, t_diff, last_t), dim=-1)
        return self.self_func(H_)


    def h(self, h_i):
        return self.jump_func(h_i)


    def e(self, H):
        return nn.functional.softplus(self.lambda_func(H).squeeze(dim=-1))


class LNSDEFunc(nn.Module):


    def __init__(self, dim_h,  dim_hidden, num_hidden, activation=nn.Tanh()):

        super(LNSDEFunc, self).__init__()
        self.dim_h = dim_h

        self.self_func = MLP(dim_in = dim_h + 2, dim_out = dim_h, dim_hidden=dim_hidden, num_hidden=num_hidden, sigma=0.01, activation=activation)

        self.lambda_func = MLP(dim_in = dim_h, dim_out = 1, dim_hidden=dim_hidden, num_hidden=0, sigma=0.1, activation=activation)

        self.diffusion_net = MLP(dim_in = 1, dim_out = dim_h, dim_hidden = dim_hidden, num_hidden=num_hidden, sigma=0.01, activation=activation)

        self.jump_func = MLP(dim_in = dim_h, dim_out = dim_h, dim_hidden = dim_hidden, num_hidden=num_hidden, sigma=0.1, activation=activation)

    def g(self, H, t):

        return H * self.diffusion_net(t)


    def f(self, H, t_diff, last_t):
        H_ = torch.cat((H, t_diff, last_t), dim=-1)
        return self.self_func(H_)

    def h(self, h_i):
        return self.jump_func(h_i)

    def e(self, H):

        return nn.functional.softplus(self.lambda_func(H).squeeze(dim=-1))

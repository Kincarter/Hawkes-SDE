import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from load_data.datamodule import creat_dataloader, obtain_N
from model_HSDE import HSDE
import torch.nn as nn
import time
def test():

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HSDE.setting_model_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_data = creat_dataloader("test", args.data_name, args.observation, args.batch_size, shuffle=False)

    N = obtain_N(args.data_name, args.observation) + 1
    model = HSDE(N=N, m=args.m, hidden_dim=args.hidden_dim, temporal_dim=args.temporal_dim,
                     num_heads=args.num_heads, dropout_rate=args.dropout_rate, attn_dropout_rate=args.attn_dropout_rate,
                     ffn_dim=args.ffn_dim, num_layers=args.num_layers, lr=args.lr, weight_decay=args.weight_decay,
                     alpha=args.alpha, beta=args.beta, lr_decay_step=args.lr_decay_step,
                     lr_decay_gamma=args.lr_decay_gamma,
                     LPE=args.LPE, TE=args.TE, SPE=args.SPE, TIE=args.TIE, LCA=args.LCA, SD_A=args.SD_A, SD_B=args.SD_B,
                     num_hidden=2, func_type=args.f, act=nn.Tanh(), device=device)


    checkpoint_path = ''
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])


    model.eval()


    trainer = pl.Trainer.from_argparse_args(args)


    res = trainer.test(model, dataloaders=test_data)


if __name__ == '__main__':
    test()
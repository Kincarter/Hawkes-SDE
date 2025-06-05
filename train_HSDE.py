import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import time
from load_data.datamodule import creat_dataloader, obtain_N
from model_HSDE import HSDE
import torch.nn as nn

def train():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HSDE.setting_model_args(parser)
    args = parser.parse_args()

    torch.multiprocessing.spawn(train_on_gpu, args=(args,), nprocs=len(args.gpu_lst))


def train_on_gpu(rank, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    global result

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"lightning_logs/{timestamp}"



    torch.cuda.set_device(args.gpu_lst[rank])


    train_data = creat_dataloader("train", args.data_name, args.observation, args.batch_size, shuffle=True)
    print('OK')
    valid_data = creat_dataloader("valid", args.data_name, args.observation, args.batch_size, shuffle=False)
    test_data = creat_dataloader("test", args.data_name, args.observation, args.batch_size, shuffle=False)


    N = obtain_N(args.data_name, args.observation) + 1
    model = HSDE(N=N, m=args.m, hidden_dim=args.hidden_dim, temporal_dim=args.temporal_dim,
                     num_heads=args.num_heads, dropout_rate=args.dropout_rate, attn_dropout_rate=args.attn_dropout_rate,
                     ffn_dim=args.ffn_dim, num_layers=args.num_layers, lr=args.lr, weight_decay=args.weight_decay,
                     alpha=args.alpha,beta=args.beta,  lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma,
                     LPE=args.LPE, TE=args.TE, SPE=args.SPE, TIE=args.TIE, LCA=args.LCA, SD_A=args.SD_A, SD_B=args.SD_B,
                     num_hidden=2,func_type=args.f,act=nn.Tanh(),device=device)



    print('total params:', sum(p.numel() for p in model.parameters()))


    checkpoint_callback = ModelCheckpoint(monitor="valid_loss",
                                          dirpath=log_dir,
                                          filename=args.data_name + '-{epoch:03d}-{valid_loss:.4f}',
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True)


    trainer = pl.Trainer(callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
                         gradient_clip_val=args.clip_val, max_epochs=args.total_epochs, gpus=[args.gpu_lst[rank]],
                         accumulate_grad_batches=5, strategy='ddp',default_root_dir=log_dir)
    trainer.from_argparse_args(args)

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)
    res = trainer.test(model, test_data)
    print(f"Logging directory: {log_dir}")
    print(args.total_epochs)
    print(args.batch_size)
    print(args.observation)
    print(args.data_name)
    print('step=3,nojump,no attn ')


    result = res


if __name__ == '__main__':
    train()


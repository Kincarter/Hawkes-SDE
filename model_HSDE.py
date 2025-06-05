import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_modules import A_MSG, get_edge_prob
from sde_modules import SDEFunc, LNSDEFunc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_atten_mask(graph_nodes):
    batch_size = graph_nodes.size(0)
    seq_len = graph_nodes.size(1)

    pad_attn_mask = graph_nodes.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

def get_seq_mask(graph_nodes):
    batch_size = graph_nodes.size(0)
    seq_len = graph_nodes.size(1)

    pad_attn_mask = (graph_nodes.data != 0)
    return pad_attn_mask


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):

        x = self.elu(self.fc1(x))
        x = self.layer_norm(x)

        x = self.fc2(x)
        return x


class Time2vec(nn.Module):
    def __init__(self, temporal_dim):
        super(Time2vec, self).__init__()
        self.temporal_dim = temporal_dim
        self.linear_trans = nn.Linear(1, temporal_dim // 2)
        self.cos_trans = nn.Linear(1, temporal_dim // 2)

    def forward(self, t):
        ta = self.linear_trans(t)
        tb = self.cos_trans(t)
        te = torch.cat([ta, tb], -1)
        return te






class PredictionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear3 = nn.Linear(hidden_dim // 4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(self.dropout(x))
        return x


class HSDE(pl.LightningModule):
    @property
    def device(self):
        return self._device

    def __init__(self, N, m, hidden_dim, temporal_dim,
                 num_heads, dropout_rate, attn_dropout_rate, ffn_dim,
                 num_layers, lr, weight_decay, alpha, beta, lr_decay_step,
                 lr_decay_gamma, LPE, TE, SPE, TIE, LCA, SD_A, SD_B,num_hidden,func_type = 'sde',act=nn.Tanh(),num_divide=5,device=None):

        super(HSDE, self).__init__()
        self.num_heads = num_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha, self.beta = alpha, beta
        self.lr_decay_step, self.lr_decay_gamma = lr_decay_step, lr_decay_gamma
        self.LPE, self.TE, self.SPE, self.TIE, self.LCA, self.SD_A, self.SD_B = LPE, TE, SPE, TIE, LCA, SD_A, SD_B
        self.hidden_dim = hidden_dim
        self.num_divide = num_divide
        self.fnn_norm=nn.LayerNorm(hidden_dim)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
        else:
            self.device = device


        self.user_encoder = nn.Embedding(N, hidden_dim, padding_idx=0)

        if self.LPE:
            self.lpe_encoder = nn.Linear(m, hidden_dim)

        if self.SPE:
            self.spl_encoder = nn.Embedding(64, num_heads, padding_idx=0)

        if self.TIE:
            self.td_encoder = nn.Linear(1, num_heads)

        if self.LCA:
            self.lca_encoder = nn.Linear(hidden_dim, num_heads)

        if self.TE:
            self.te = Time2vec(temporal_dim)

        self.graphEncoderLayers = nn.ModuleList(
            [GraphEncoderLayer(hidden_dim, num_heads, dropout_rate, attn_dropout_rate, ffn_dim)
             for _ in range(num_layers)])

        self.Linear = TwoLayerNet(2 * hidden_dim, hidden_dim , hidden_dim)
        self.func_type = func_type
        self.MSG = A_MSG(1, hidden_dim, hidden_dim, 1, act).to(device)  # Avoid too deep

        if func_type == 'sde':
            self.func = SDEFunc(dim_h=hidden_dim,  dim_hidden=hidden_dim, num_hidden=num_hidden,
                                activation=act).to(device)
        elif func_type == 'lnsde':
            self.func = LNSDEFunc(dim_h=hidden_dim,  dim_hidden=hidden_dim, num_hidden=num_hidden,
                                  activation=act).to(device)
        else:
            raise Exception("this type of func dose not exsit")

        self.f_layernorm = nn.LayerNorm(hidden_dim)
        self.prediction = PredictionLayer(hidden_dim, dropout_rate)

        self.ctloss_train = CTLoss_train()
        self.ctloss_test = CTLoss_test()
        self.save_hyperparameters()

    def EulerSolver(self, a_h_initial, adjacent_events, num_divide=None):

        if num_divide is None:
            num_divide = self.num_divide

        dt = torch.diff(adjacent_events, dim=1) / num_divide

        ts = torch.cat([adjacent_events[:, 0].unsqueeze(dim=1) + dt * j for j in range(num_divide + 1)], dim=1)


        a_l_ts = torch.Tensor().to(a_h_initial.device)

        a_l_initial = self.func.e(a_h_initial)

        a_l_ts = torch.cat((a_l_ts, a_l_initial.unsqueeze(2)), dim=2)

        h_dt = dt.unsqueeze(dim=1)

        h_last = adjacent_events[:, [0]].unsqueeze(dim=1)

        h_dt = torch.clamp(h_dt, min=0)



        for i in range(num_divide):
            h_diff_t = h_dt * (i + 1)

            a_h_output = (a_h_initial + self.func.f(a_h_initial, h_diff_t, h_last) * h_dt
                         +self.func.g(a_h_initial, h_diff_t) * torch.sqrt(h_dt) * torch.randn_like(a_h_initial))



            a_l_output = self.func.e(a_h_output.clone())

            a_l_ts = torch.cat((a_l_ts, a_l_output.unsqueeze(2)), dim=2)

            a_h_initial = a_h_output
            a_l_initial = a_l_output
        return ts, a_l_ts, a_h_initial

    def jump_and_msg_passing(self, A_H,x,seq_mask):

        A_H = self.MSG(A_H, x,seq_mask)


        return A_H

    def forward(self, batch_data):

        graph_nodes, temporal_list, lpe, spl_matrix, interval_matrix, lca_matrix, labels, real_len = batch_data
        batch_size, seq_len = graph_nodes.size()
        temporal_lst = temporal_list.unsqueeze(-1)
        x = self.user_encoder(graph_nodes)
        if self.LPE:

            sign_filp = torch.rand(lpe.size(2)).to(x.device)

            sign_filp[sign_filp >= 0.5] = 1.0
            sign_filp[sign_filp < 0.5] = -1.0

            lpe = lpe * sign_filp.unsqueeze(0).unsqueeze(0)

            lpe_trans = self.lpe_encoder(lpe)
            x += lpe_trans
        if self.TE:
            x += self.te(temporal_lst)
        x_origin = x
        attn_bias = torch.zeros((batch_size, seq_len, seq_len, self.num_heads), device=x.device)
        if self.SPE:

            spl_bias = self.spl_encoder(spl_matrix)
            attn_bias += spl_bias
        if self.TIE:

            interval_bias = torch.cos(
                self.td_encoder(interval_matrix.unsqueeze(-1)))
            attn_bias += interval_bias
        if self.LCA:

            bz, n = graph_nodes.size()

            aux_source_idx, aux_lca_idx, aux_target_idx = lca_matrix[:, :, :n], lca_matrix[:, :, n:2 * n], lca_matrix[:,
                                                                                                           :, 2 * n:]

            source_vectors, target_vectors, lca_vectors = F.softmax(self.user_encoder(aux_source_idx), -1), F.softmax(
                self.user_encoder(aux_target_idx), -1), F.softmax(self.user_encoder(aux_lca_idx), -1)

            lca_bias = 1 - (
                        F.kl_div(lca_vectors.log(), source_vectors, reduction='none') + F.kl_div(target_vectors.log(),
                                                                                                 lca_vectors,
                                                                                                 reduction='none'))

            lca_bias = self.lca_encoder(lca_bias)
            attn_bias += lca_bias

        attn_bias = attn_bias.transpose(1, 3).contiguous()
        ########################
        attn_lst = []
        batch_relations = []
        attn_mask = get_atten_mask(graph_nodes)

        for encoder in self.graphEncoderLayers:
            x, attn = encoder(x, attn_bias, attn_mask)



        seq_mask=get_seq_mask(graph_nodes)



        a_l_batch_l = torch.zeros(batch_size, seq_len, 1, device=self.device)

        a_l_batch_r = torch.zeros(batch_size, seq_len, 1, device=self.device)

        a_h_batch_l = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)

        a_h_batch_r = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)


        a_h0 = x_origin[:,0,:]
        a_l0 = self.func.e(a_h0)#
        a_l0 = a_l0.unsqueeze(1)




        a_l_batch_l[:, 0, :] = a_l0
        a_h_batch_l[:, 0 ,:] = a_h0

        #first jump
        a_h0,x_h_new =self.jump_and_msg_passing(a_h0,x_origin,seq_mask)
        a_h_batch_r[:, 0 ,:] = a_h0







        a_l_batch_r[:, 0, :] = self.func.e(a_h0).unsqueeze(1)

        tsave = torch.Tensor().to(self.device)
        a_l_tsave = torch.Tensor().to(self.device)

        savet = torch.Tensor().to(self.device)
        tsave_a_l = torch.Tensor().to(self.device)

        for i in range(seq_len - 1):

            adjacent_events =temporal_list[:, i:i + 2]
            ts, a_l_ts_l, a_h0 = self.EulerSolver(a_h0.unsqueeze(1), adjacent_events)




            tsave = torch.cat((tsave, ts), dim=1)
            savet = torch.cat((savet, ts[:,:-1]), dim=1)
            a_l_tsave = torch.cat((a_l_tsave, a_l_ts_l), dim=2)
            tsave_a_l = torch.cat((tsave_a_l, a_l_ts_l[:,:,:-1]), dim=2)


            a_l_batch_l[:, i + 1, :] = a_l_ts_l[:, :, -1]
            a_h_batch_l[:, i + 1, :] = a_h0.squeeze(1)
            a_l_ts_r = a_l_ts_l.clone()



            a_h0,x_h_new = self.jump_and_msg_passing(a_h0.squeeze(1), x_h_new,seq_mask)
            a_h_batch_r[:, i + 1, :] = a_h0


            a_l_batch_r[:, i + 1, :] = self.func.e(a_h0.unsqueeze(1))




        a_l_batch_l = a_l_batch_l.squeeze(-1)


        seq_mask = seq_mask.float()

        a_masked_l_time_l = torch.log(a_l_batch_l ) * seq_mask  # lambda --> log(lambda)





        a_sum_term = torch.sum(a_masked_l_time_l,dim=1)




        mask_without_first_col = seq_mask[:, 1:]



        expanded_mask = mask_without_first_col.unsqueeze(2).repeat(1, 1, self.num_divide + 1).view(batch_size, -1)

        expanded_mask = expanded_mask.unsqueeze(1)


        a_l_tsave = a_l_tsave * expanded_mask  # mask the eta_tsave

        expanded_mask_savet = mask_without_first_col.unsqueeze(2).repeat(1, 1, self.num_divide).view(batch_size, -1)
        #expanded_mask:batch*1*((seq_len-1)*(num_divide+1))
        expanded_mask_savet = expanded_mask_savet.unsqueeze(1)
        tsave_a_l = tsave_a_l * expanded_mask_savet
        savet = savet * expanded_mask_savet.squeeze(1)



        expanded_diff_tsave = torch.diff(tsave).unsqueeze(1)




        a_integral_term = torch.sum(
            (a_l_tsave[:, :, :-1] * expanded_mask[:, :, :-1] + a_l_tsave[:, :, 1:] * expanded_mask[:, :, 1:]) * (
                    expanded_diff_tsave * expanded_mask[:, :, 1:]) / 2 ,dim=2)





        a_nll =a_integral_term.squeeze(1)-a_sum_term










        x = x.sum(1)
        x = self.f_layernorm(x)

        out = self.prediction(x) + expect_popularity
        return out,x,a_nll


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(
                                      optimizer,
                                      step_size=self.lr_decay_step,
                                      gamma=self.lr_decay_gamma),
                        'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch_data, batch_idx):

        y_pre, _,a_nll,_,_ = self(batch_data)

        tgt = batch_data[-2].view(-1, 1)
        loss = self.ctloss_train(y_pre, tgt, 1, self.alpha, self.beta, self.SD_A, self.SD_B,a_nll)

        self.log('train_loss', loss, sync_dist=True)

        return loss


    def validation_step(self, batch_data, batch_idx):
        y_pre, _, _ ,_,_= self(batch_data)
        tgt = batch_data[-2].view(-1, 1)
        return {'y_pre': y_pre, 'tgt': tgt}

    def validation_epoch_end(self, outputs):

        y_pre = torch.cat([o['y_pre'] for o in outputs])
        tgt = torch.cat([o['tgt'] for o in outputs])
        #计算总的loss值。
        loss = self.ctloss_train(y_pre, tgt)
        self.log('valid_loss', loss, sync_dist=True)
    #输出测试集的预测值和真实值
    def test_step(self, batch_data, batch_idx):
        y_pre,  _, _,savet,tsave_a_l = self(batch_data)
        tgt = batch_data[-2].view(-1, 1)
        return {'y_pre': y_pre.cpu(), 'tgt': tgt.cpu(),'savet': savet.cpu(),'tsave_a_l': tsave_a_l.cpu()}
    #
    def test_epoch_end(self, outputs):
        y_pre = torch.cat([o['y_pre'] for o in outputs])
        tgt = torch.cat([o['tgt'] for o in outputs])


        result = y_pre.cpu().float().numpy()
        real = tgt.cpu().float().numpy()





        torch.save(np.vstack((result, real)), 'res.pt')
        self.y_pre = y_pre
        self.tgt = tgt
        msle,mape,R2 = self.ctloss_test(y_pre, tgt)
        self.log('test_MSLE', msle, sync_dist=True)
        self.log('test_MAPE', mape, sync_dist=True)
        self.log('test_R2', R2, sync_dist=True)



    @staticmethod
    def setting_model_args(parent_parser):
        parser = parent_parser.add_argument_group("HSDE")
        parser.add_argument('--hidden_dim', type=int, default=32, help="The hidden dimension of models.")
        parser.add_argument('--ffn_dim', type=int, default=32, help="The hidden dimension of FFN layers.")
        parser.add_argument('--temporal_dim', type=int, default=32, help="Time embedding dimension.")
        parser.add_argument('--num_heads', type=int, default=8, help='The num heads of attention.')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout prob.')
        parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='Attention dropout prob.')
        parser.add_argument('--num_layers', type=int, default=6, help="Num of Transformer encoder layers.")
        parser.add_argument('--lr', type=float, default=0.002, help='Learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
        parser.add_argument('--m', type=int, default=32, help="The smallest m eigenvalues.")
        parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
        parser.add_argument('--alpha', type=float, default=1, help="Self attention scores distillation loss rate.")
        parser.add_argument('--beta', type=float, default=1, help="Batch attention scores distillation loss rate.")
        parser.add_argument('--clip_val', type=float, default=5.0, help="Gradient clipping values. ")
        parser.add_argument('--total_epochs', type=int, default=50, help="Max epochs of model training.")
        parser.add_argument('--lr_decay_step', type=int, default=25, help="Learning rate decay step size.")
        parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
        #其中nargs='+'表示可以接受多个参数，代表可以接受多个gpu
        parser.add_argument('--gpu_lst', nargs='+', type=int, default=[2],
                            help="Which gpu to use. Use -1 to use all available GPUs.")
        #其中action='store_false'代表默认值为true表示当前不适用该参数，因此通过控制action可以用来方便进行消融实验。
        parser.add_argument('--LPE', action='store_false', help="Laplacian positional encoding.")
        parser.add_argument('--TE', action='store_false', help="Temporal positional encoding.")
        parser.add_argument('--SPE', action='store_true', help="Shortest path bias encoding.")
        parser.add_argument('--TIE', action='store_true', help="Temporal interval bias encoding.")
        parser.add_argument('--LCA', action='store_true', help="LCA bias encoding.")
        parser.add_argument('--SD_A', action='store_true', help="Self-Distillation attention scores.")
        parser.add_argument('--SD_B', action='store_true', help="Self-Distillation batch relation scores.")
        parser.add_argument('--observation', type=str, default="3")
        parser.add_argument('--data_name', type=str, default="aps")
        parser.add_argument('-f', type=str, choices=['sde', 'lnsde'], default='lnsde',
                            help=' structure')
        # parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use.")
        return parent_parser

    @device.setter
    def device(self, value):
        self._device = value


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)


    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.attn_dim = attn_dim = hidden_dim // num_heads

        self.scale = attn_dim ** -0.5

        #定义Q、K、V的线性层的权重矩阵，维度为hidden_dim*hidden_dim,
        self.linear_Q = nn.Linear(hidden_dim, num_heads * attn_dim)
        self.linear_K = nn.Linear(hidden_dim, num_heads * attn_dim)
        self.linear_V = nn.Linear(hidden_dim, num_heads * attn_dim)
        #线性输出层
        self.linear_out = nn.Linear(num_heads * attn_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, q, k, v, attn_bias=None, attn_mask=None):

        d_q = d_k = d_v = self.attn_dim

        batch_size = q.size(0)


        q = self.linear_Q(q).view(batch_size, -1, self.num_heads, d_q)  # (batch_size, seq_len, num_heads, attn_dim)
        k = self.linear_K(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_V(v).view(batch_size, -1, self.num_heads, d_v)


        q, k, v = q.transpose(1, 2), k.transpose(1, 2).transpose(2, 3), v.transpose(1, 2)

        x = torch.matmul(q, k) * self.scale
        if attn_bias is not None:
            x = x + attn_bias

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        x.masked_fill_(attn_mask, -1e9)

        attention_scores = torch.softmax(x, dim=-1)

        x = self.attn_dropout(attention_scores)

        x = torch.matmul(x, v)

        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, self.num_heads * self.attn_dim)  # (batch_size, seq_len, num_heads * attn_dim)

        x = self.linear_out(x)
        return x, attention_scores


class GraphEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate, attn_dropout_rate, ffn_dim):
        super(GraphEncoderLayer, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.attention = MultiHeadAttention(hidden_dim, num_heads, attn_dropout_rate)


        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim)

    def forward(self, x, attn_bias=None, attn_mask=None):

        y = self.attention_norm(x) #计算LN(Hp(s-1)) y=batch*seq_len*hidden_dim

        y, attn = self.attention(y, y, y, attn_bias, attn_mask) #计算MHA(LN(Hp(s-1))

        y = self.attention_dropout(y)

        x = x + y


        y = self.ffn_norm(x)

        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, attn


class CTLoss_test(nn.Module):
    def __init__(self):
        super(CTLoss_test, self).__init__()

        self.mse_loss = nn.MSELoss()

    def forward(self, y_pre, tgt, is_sd=0, sad=None, sbd=None, alpha=0, beta=0, SD_A=False, SD_B=False):

        msle = self.mse_loss(y_pre, tgt)
        mape = torch.mean(torch.abs((y_pre-tgt)/(tgt+1)))
        R2 = 1-torch.sum((y_pre - tgt) ** 2)/(torch.sum((tgt -torch.mean(tgt))**2)+1)
        return msle,mape,R2

class CTLoss_train(nn.Module):
    def __init__(self):
        super(CTLoss_train, self).__init__()

        self.mse_loss = nn.MSELoss()

    def forward(self, y_pre, tgt, is_sd=0, alpha=0, beta=0, SD_A=False, SD_B=False,a_nll=None):

        if a_nll is not None:
            loss = self.mse_loss(y_pre, tgt)+torch.mean(a_nll)
        else:
            loss = self.mse_loss(y_pre, tgt)


        return loss



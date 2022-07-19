import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from scipy import stats
from utils import relative_to_abs
import math
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h,Relation):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        # attn = self.softmax(attn)
        attn = torch.mul(Relation.unsqueeze(1).repeat(1, self.n_head, 1, 1).cuda(), attn)
        attn = self.softmax(attn)

        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class seqBatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(seqBatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, h_prime)

        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout))

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x, Relation):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x, Relation)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x

class seqGAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(seqGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                seqBatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout))

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def neig_direction(self,diffx,diffy):
        if diffx != 0:
            dire = 180 * math.atan2(diffy, diffx) / (math.pi)
            if dire < 0:
                dire = 360 + dire
        else:
            if diffy > 0:
                dire = 90
            elif diffy < 0:
                dire = 270
            else:
                dire = 0
        return dire

    def relation_Matrix(self, curr_dire):
        currdata = curr_dire[:,:,2:4]
        F, N, D = currdata.size()
        d= np.zeros((F, N, N))
        r=np.zeros((F, N, N))
        l = 156


        currdata = currdata.cuda().data.cpu().numpy()
        for cur_f in range(F):
            d[cur_f] = squareform(pdist(currdata[cur_f], metric='euclidean'))
        d = np.where(d<=l,1,0)

        for cur_f in range(F):
            for cur_n in range(N):
                a = curr_dire[cur_f, cur_n, 5]
                up = a + 62  #62
                down = a - 62

                for n_neig in range(N):
                    if (d[cur_f, cur_n, n_neig] == 1):
                        dire_n_neig = self.neig_direction(currdata[cur_f, n_neig, 0] - currdata[cur_f, cur_n, 0], currdata[cur_f, n_neig, 1] - currdata[cur_f, cur_n, 1])
                        if up> 360:
                            if (down <= dire_n_neig <= 360) or (0 <= dire_n_neig <= (up - 360)):
                                r[cur_f, cur_n, n_neig] = 1
                        elif 62 <= up <= 124:
                            if (down + 360 <= dire_n_neig <= 360) or (0 <= dire_n_neig <= up):
                                r[cur_f, cur_n, n_neig] = 1
                        else:
                            r[cur_f, cur_n, n_neig] = 1
        r = torch.FloatTensor(r)
        return r


    def forward(self, obs_traj_embedding, seq_start_end, obs_dire):
        graph_embeded_data = []

        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_obs_dire = obs_dire[:, start:end, :]
            Relation = self.relation_Matrix(curr_obs_dire)
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj,Relation)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class seqGATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(seqGATEncoder, self).__init__()
        self.seq_gat_net = seqGAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.seq_gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,   # 2
        traj_lstm_hidden_size,  # 32
        n_units,
        n_heads,
        graph_network_out_dims, # 64
        dropout,
        alpha,
        graph_lstm_hidden_size, #64
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
    ):
        super(TrajectoryGenerator, self).__init__()
        self.embedding_size=embedding_size
        self.light_input_size=light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size=light_embedding_size
        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )
        self.seqgatencoder = seqGATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size    # 64
        self.traj_lstm_hidden_size = traj_lstm_hidden_size  # 32

        self.pred_lstm_hidden_size = (
                self.light_embedding_size + self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0]
        )

        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )
        
        self.light_embedding=nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size,self.light_embedding_size),
            nn.ReLU()
        )

        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size + self.light_embedding_size, 2)
        
        self.traj_gat_hidden2pos = nn.Linear(
            self.light_embedding_size + self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2
        )
        self.pred_hidden2pos = nn.Linear(self.light_embedding_size + self.pred_lstm_hidden_size, 2)

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )
        

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def init_hidden_light_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim

        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)

        return decoder_h

    def get_last_state(self,obs_traj_pos,obs_state):
        """
        Inputs:
        - obs_traj_pos: pytorch tensor of shape (seq_len, batch, 11)
        Outputs:
        - state: pytorch tensor of shape (batch, 5)
        """

        dis = torch.sqrt((obs_traj_pos[-1,:,2]-obs_state[-1,:,0])**2 + (obs_traj_pos[-1,:,3] - obs_state[-1,:,1])**2)
        disx = obs_traj_pos[-1, :, 2] - obs_state[-1,:,0]
        disy = obs_traj_pos[-1, :, 3] - obs_state[-1,:,1]
        light_state=obs_state[-1,:,2:4]
        dis_state=torch.stack([dis,disx,disy],dim=1)
        state_last=torch.cat((dis_state,light_state),dim=1)

        return state_last   # NV，5
    def get_next_state(self,pred_traj_rel,obs_traj_pos,pred_state):
        """
        Inputs:
        - pred_traj_rel: list of pytorch tensor  (seq_pred, batch, 2)
        - obs_traj_pos: pytorch tensor of shape (seq_len, batch, 11)
        Outputs:
        - state: pytorch tensor of shape (batch, 5)
        """
        pred_traj_rel=torch.stack(pred_traj_rel)    # T NV C
        step=pred_traj_rel.size(0)

        start_pos=obs_traj_pos[-1,:,2:4]    # NV,2
        real_pos=relative_to_abs(pred_traj_rel,start_pos)   # TVC
        dis = torch.sqrt((real_pos[-1, :, 0] - pred_state[-1, :, 0]) ** 2  + (real_pos[-1, :, 1] - pred_state[-1, :, 1]) ** 2)  # 计算目标当前位置与对应交通灯之间的距离   NV
        disx = real_pos[-1, :, 0] - pred_state[-1, :, 0]
        disy = real_pos[-1, :, 1] - pred_state[-1, :, 1]
        dis_state=torch.stack([dis,disx,disy],dim=1)
        
        last_state=pred_state[step-1,:,2:4]
       

        state_last=torch.cat((dis_state,last_state),dim=1) # NV,5

        return state_last


    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        obs_state,
        pred_state,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        training_step=3,
    ):
        batch = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)    # 初始化h,c
        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(obs_traj_rel[: self.obs_len].size(0), dim=0)):
            inputtraj = input_t[:, :, 2:4]
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]

        kl = 6
        obs_dire = obs_traj_pos[:, :, 0:6]
        obs_dire[:, :, 5] = obs_traj_pos[:, :, 9]
        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
        )
        staend = torch.zeros((1, 2), dtype=torch.int)

        with torch.no_grad():
            for j in range(self.obs_len):
                if j <= kl:
                    staend[0, 1] = j + 1
                    graph_inter_input = self.seqgatencoder(graph_lstm_input[0:(j + 1)].permute(1, 0, 2), staend)
                else:
                    staend[0, 1] = kl + 1
                    graph_inter_input = self.seqgatencoder(graph_lstm_input[(j - kl):(j + 1)].permute(1, 0, 2),
                                                           staend)
                graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]

        light_state = self.get_last_state(obs_traj_pos, obs_state)
        light_state_embedding = self.light_embedding(light_state)
        encoded_before_noise_hidden = torch.cat(
            (light_state_embedding, traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]),
            dim=1)

        pred_lstm_hidden = self.add_noise(
            encoded_before_noise_hidden, seq_start_end
        )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        obs_traj_rel = obs_traj_rel[:, :, 2:4]
        output = obs_traj_rel[self.obs_len - 1]
        if self.training:
            for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len:].chunk(
                        obs_traj_rel[-self.pred_len:].size(0), dim=0
                    )  # 12帧
            ):
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)  # 136
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                else:
                    light_state = self.get_next_state(pred_traj_rel, obs_traj_pos, pred_state)
                light_state_embedding = self.light_embedding(light_state)
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)  # 136+32=168
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]
            outputs = torch.stack(pred_traj_rel)
        else:
            for i in range(self.pred_len):
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_hidden, pred_lstm_c_t)
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                else:
                    light_state = self.get_next_state(pred_traj_rel, obs_traj_pos, pred_state)
                light_state_embedding = self.light_embedding(light_state)
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]
            outputs = torch.stack(pred_traj_rel)

        return outputs


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        part_lstm_input_size,
        part_lstm_hidden_size,
        merge_lstm_input_size,
        merge_lstm_hidden_size,
        dropout,
        light_input_size=4,
        embedding_size=32,
        light_embedding_size=16,
    ):
        super(TrajectoryDiscriminator, self).__init__()
        self.light_input_size=light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size=light_embedding_size
        self.embedding_size=embedding_size
        self.light_embedding=nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size,self.light_embedding_size),
            nn.ReLU()
        )
        self.pos_embedding=nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )
        self.merge_embedding=nn.Sequential(
            nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32,1),
            # nn.ReLU()
        )
        self.part_lstm_input_size=part_lstm_input_size
        self.part_lstm_hidden_size=part_lstm_hidden_size
        self.state_part_lstm=nn.LSTMCell(self.part_lstm_input_size,self.part_lstm_hidden_size)
        self.pos_part_lstm=nn.LSTMCell(self.part_lstm_input_size,self.part_lstm_hidden_size)
        self.merge_lstm_input_size=merge_lstm_input_size # 64
        self.merge_lstm_hidden_size=merge_lstm_hidden_size #64
        self.merge_lstm=nn.LSTMCell(self.merge_lstm_input_size,self.merge_lstm_hidden_size)
    def init_hidden_part_lstm(self, batch):
        return (
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
        )

    def init_hidden_merge_lstm(self, batch):
        return (
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
        )
    def forward(self,traj,state,seq_start_end):
        batch = traj.shape[1]       # count 64*V
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_part_lstm(batch)
        state_lstm_h_t, state_lstm_c_t = self.init_hidden_part_lstm(batch)
        merge_lstm_h_t, merge_lstm_c_t = self.init_hidden_merge_lstm(batch) 
        traj_lstm_hidden_states = []
        state_lstm_hidden_states = []

        for i, input_t in enumerate(traj[:].chunk(traj[:].size(0), dim=0)):
            input_t=self.pos_embedding(input_t.squeeze(0))
            traj_lstm_h_t, traj_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]
        for i, input_t in enumerate(state[:].chunk(state[:].size(0), dim=0)):
            input_t=self.light_embedding(input_t.squeeze(0))
            state_lstm_h_t, state_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (state_lstm_h_t, state_lstm_c_t))
            state_lstm_hidden_states += [state_lstm_h_t]
        for i in range(len(traj_lstm_hidden_states)):
            input_t=torch.cat((traj_lstm_hidden_states[i],state_lstm_hidden_states[i]),dim=1)
            merge_lstm_h_t, merge_lstm_c_t=self.merge_lstm(input_t,(merge_lstm_h_t, merge_lstm_c_t))
        output=self.merge_embedding(merge_lstm_h_t)
        output=output
        return output


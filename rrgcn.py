import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from model import BaseRGCN
from decoder import ConvTransE, ConvTransR



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb, pe_init=self.pe_init, pe_dim=self.pe_dim)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb, method=0):
        if self.encoder_name == "uvrgcn":
            if method == 0:
                node_id = g.ndata['id'].squeeze()
                g.ndata['h'] = init_ent_emb[node_id]
            elif method == 1:
                g.ndata['h'] = init_ent_emb
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder, encoder, gnn, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=True, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, sequence='rnn', use_cuda=False, analysis=False, pe_init="rw", pe_dim=3):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder
        self.encoder_name = encoder
        self.gnn = gnn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.sequence = sequence
        self.emb_rel = None

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        # self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.emb_rel)

        # self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        # torch.nn.init.normal_(self.dynamic_emb)
        self.dynamic_emb = None

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.rgcn = None
        # self.rgcn = RGCNCell(num_ents,
        #                      h_dim,
        #                      h_dim,
        #                      num_rels * 2,
        #                      num_bases,
        #                      num_basis,
        #                      num_hidden_layers,
        #                      dropout,
        #                      self_loop,
        #                      skip_connect,
        #                      encoder,
        #                      self.opn,
        #                      self.emb_rel,
        #                      use_cuda,
        #                      analysis)
        if self.sequence == 'regcn':
            self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
            self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
            nn.init.zeros_(self.time_gate_bias)
        elif self.sequence == 'rnn':
            self.rnn_cell = nn.GRUCell(self.h_dim, self.h_dim)


        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        # if decoder == "convtranse":
        #     self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # else:
        #     raise NotImplementedError

    def forward(self, g_list, static_graph=None, device=None):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(device)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(device)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel[0:self.num_rels *2])    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            if self.gnn == 'regcn':
                current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            elif self.gnn == 'rgat':
                g.edata['r_h'] = self.h_0[g.edata['etype']]
                #g.edata['r_h'] = self.emb_rel[g.edata['etype']]
                current_h = self.rgcn(g, self.h)
            if self.sequence == 'regcn':
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
                self.h = time_weight * current_h + (1-time_weight) * self.h
            elif self.sequence == 'rnn':
                self.h = self.rnn_cell(current_h, self.h)
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, device):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).to(device)
        loss_rel = torch.zeros(1).to(device)
        loss_static = torch.zeros(1).to(device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(device)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, device)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

class BiRecurrentRGCN(nn.Module):
    def __init__(self, decoder, encoder, gnn, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=True, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, sequence='rnn', use_cuda=False, analysis=False, pe_init="rw", pe_dim=3):
        super(BiRecurrentRGCN, self).__init__()

        self.decoder_name = decoder
        self.encoder_name = encoder
        self.gnn = gnn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.sequence = sequence
        self.emb_rel = None

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)


        # self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        # torch.nn.init.normal_(self.dynamic_emb)
        self.dynamic_emb = None

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.rgcn = None
        self.burgcn = None
        if self.sequence == 'regcn':
            self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
            self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
            nn.init.zeros_(self.time_gate_bias)

            self.reverse_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
            nn.init.xavier_uniform_(self.reverse_time_gate_weight, gain=nn.init.calculate_gain('relu'))
            self.reverse_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
            nn.init.zeros_(self.reverse_time_gate_bias)  
        elif self.sequence == 'rnn':
            self.rnn_cell = nn.GRUCell(self.h_dim, self.h_dim)
            self.rnn_cell2 = nn.GRUCell(self.h_dim, self.h_dim)                         


        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.relation_cell_2 = nn.GRUCell(self.h_dim*2, self.h_dim)


        # decoder
        # if decoder == "convtranse":
        #     self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # else:
        #     raise NotImplementedError

    def forward(self, g_list, static_graph=None, device=None):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(device)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        
        self.h1 = self.h

        history_embs = []
        reverse_history_embs = []
        reverse_g_list = g_list[::-1]
        for i, g in enumerate(reverse_g_list):
            g = g.to(device)
            temp_e = self.h1[g.r_to_e]  # 存储了与当前图中关系关联的实体嵌入
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:    
                x_input = torch.cat((self.reverse_emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_1 = self.relation_cell_2(x_input, self.reverse_emb_rel[0:self.num_rels *2])    # 第1层输入       
                self.h_1 = F.normalize(self.h_1) if self.layer_norm else self.h_1
            else:
                x_input = torch.cat((self.reverse_emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_1 = self.relation_cell_2(x_input, self.h_1)  # 第2层输出==下一时刻第一层输入
                self.h_1 = F.normalize(self.h_1) if self.layer_norm else self.h_1
            
            # self.h_1 = self.h_1+ e
            # current_h = current_h + p
            if self.gnn == "regcn":
                current_h = self.burgcn.forward(g, self.h1, [self.h_1, self.h_1])
            elif self.gnn == "rgat":
                current_h = self.h_1[g.edata['type']]
                current_h = self.burgcn(g, self.h1)
            
            if self.sequence == "regcn":
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                time_weight = F.sigmoid(torch.mm(self.h1, self.reverse_time_gate_weight) + self.reverse_time_gate_bias)
                self.h1 = time_weight * current_h + (1-time_weight) * self.h1
            elif self.sequence == "rnn":
                self.h1 = self.rnn_cell2(current_h, self.h)
            
            reverse_history_embs.append(self.h1)

        for i, g in enumerate(g_list):
            g = g.to(device)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel[0:self.num_rels *2])    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            if self.gnn == 'regcn':
                current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            elif self.gnn == 'rgat':
                g.edata['r_h'] = self.h_0[g.edata['etype']]
                #g.edata['r_h'] = self.emb_rel[g.edata['etype']]
                current_h = self.rgcn(g, self.h)
            if self.sequence == 'regcn':
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
                self.h = time_weight * current_h + (1-time_weight) * self.h
            elif self.sequence == 'rnn':
                self.h = self.rnn_cell(current_h, self.h)
            self.h = self.h + reverse_history_embs[len(g_list)-i-1]
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, device):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).to(device)
        loss_rel = torch.zeros(1).to(device)
        loss_static = torch.zeros(1).to(device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(device)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, device)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

#!/user/bin/env python3
# -*- coding: utf-8 -*-
# rwse、lappe、signnet、deepnet

import torch
from scipy import sparse as sp
import dgl
import numpy as np


def lap_positional_encoding_tkg(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    g_list: 包含每个时间戳的子图列表
    pos_enc_dim: 位置编码的维度
    """
    # for g in g_list:
    # 计算每个子图的拉普拉斯矩阵并生成位置编码
    A = g.adj_external(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 保存位置编码
    g.ndata['p'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().to("cuda")

    return g

def lap_positional_encoding_tkg(g, pos_enc_dim):
    A = g.adj_external(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # 在拉普拉斯矩阵对角线上加上一个小的正则化项
    epsilon = 1e-5
    L = L + sp.eye(g.number_of_nodes()) * epsilon

    # 转换为 torch 稠密矩阵并移动到 GPU
    L_torch = torch.from_numpy(L.toarray()).float().to("cuda")

    # 使用 torch.linalg.eigh 进行特征分解
    EigVal, EigVec = torch.linalg.eigh(L_torch, UPLO='L')

    # 保存位置编码
    g.ndata['p'] = EigVec[:, 1:pos_enc_dim + 1].float()

    return g




def rw_positional_encoding_tkg(g, pos_enc_dim):
    """
    Initializing positional encoding with RWPE for Temporal Knowledge Graphs (TKG)

    Args:
        g_list: 包含每个时间戳的子图列表
        pos_enc_dim: 位置编码的维度
        type_init: 初始化方式，当前支持 'rand_walk'（随机游走）

    Returns:
        g_list: 包含位置编码的子图列表
    """

    n = g.number_of_nodes()

    # Geometric diffusion features with Random Walk
    A = g.adj_external(scipy_fmt="csr")  # 计算邻接矩阵
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1 
    RW = A * Dinv  # 计算随机游走矩阵
    M = RW  # 一阶随机游走矩阵

    # Iterate to get positional encoding
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M  # 计算 K阶随机游走矩阵
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    # print("PE.shape==>", PE.shape)
    g.ndata['p'] = PE  # (batc_size, pos_enc_dim)

    return g






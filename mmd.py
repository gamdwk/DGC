import torch
import torch.nn as nn
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


def compute_mmd(samples_p, samples_q, sigma=1.0, device='cpu'):
    # samples_p = samples_p.to('cpu')
    # samples_q = samples_q.to('cpu')
    # print(samples_p)
    # print(samples_q)
    n_p = samples_p.size(0)
    n_q = samples_q.size(0)

    # 计算核矩阵 K_p
    pairwise_distances_p = torch.cdist(samples_p, samples_p, p=2)
    K_p = torch.exp(-pairwise_distances_p ** 2 / (2 * sigma ** 2))

    # 计算核矩阵 K_q
    pairwise_distances_q = torch.cdist(samples_q, samples_q, p=2)
    K_q = torch.exp(-pairwise_distances_q ** 2 / (2 * sigma ** 2))

    # 计算混合核矩阵 K_mix,内存消耗很大
    pairwise_distances_mix = torch.cdist(samples_p, samples_q, p=2)
    K_mix = torch.exp(-pairwise_distances_mix ** 2 / (2 * sigma ** 2))

    # 计算 MMD
    mmd = K_p.sum() / (n_p * (n_p - 1)) + K_q.sum() / (n_q * (n_q - 1)) - 2 * K_mix.sum() / (n_p * n_q)

    return mmd


def compute_mmd2(dist1, dist2):
    """
    计算两个分布的MMD（Maximum Mean Discrepancy）

    参数:
        dist1 (torch.Tensor): 第一个分布的样本，形状为 (N, D)
        dist2 (torch.Tensor): 第二个分布的样本，形状为 (M, D)

    返回:
        torch.Tensor: MMD的值
    """
    # 计算两个分布的样本数量
    N = dist1.size(0)
    M = dist2.size(0)

    # 计算两个分布的维度
    D = dist1.size(1)

    # 将样本堆叠在一起，形成形状为 (N+M, D) 的新样本集合
    dists = torch.cat([dist1, dist2], dim=0)

    # 计算 Gram 矩阵
    gram_matrix = torch.matmul(dists, dists.t())

    # 计算第一个分布的 Gram 矩阵部分的均值
    gram_matrix1 = gram_matrix[:N, :N]
    mean1 = gram_matrix1.sum() / (N * N)

    # 计算第二个分布的 Gram 矩阵部分的均值
    gram_matrix2 = gram_matrix[N:, N:]
    mean2 = gram_matrix2.sum() / (M * M)

    # 计算跨两个分布的 Gram 矩阵部分的均值
    gram_matrix12 = gram_matrix[:N, N:]
    mean12 = gram_matrix12.sum() / (N * M)

    # 计算 MMD 的值
    mmd = mean1 + mean2 - 2 * mean12
    return mmd


def compute_mmd3(dist1, dist2, sigma=1.0, device='cpu'):
    """
    通过核技巧近似计算两个分布的MMD（Maximum Mean Discrepancy）

    参数:
        dist1 (torch.Tensor): 第一个分布的样本，形状为 (N, D)
        dist2 (torch.Tensor): 第二个分布的样本，形状为 (M, D)
        sigma (float): RBF核函数的带宽参数
        device (str): 使用的设备，可以是 'cpu' 或 'cuda'

    返回:
        torch.Tensor: MMD的值
    """
    # 将样本移动到指定的设备上
    # dist1 = dist1.cpu()
    # dist2 = dist2.cpu()

    # 计算样本之间的核矩阵
    N = dist1.size(0)
    M = dist2.size(0)
    #print("mmd:N:{},M:{}".format(N, M))
    dist1 = dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()
    if N > 20000:
        dist1 = random_sample(dist1, N, 20000)
    K11 = torch.tensor(rbf_kernel(dist1, dist1, gamma=1.0 / (2 * sigma ** 2))).to(device)
    K22 = torch.tensor(rbf_kernel(dist2, dist2, gamma=1.0 / (2 * sigma ** 2))).to(device)
    K12 = torch.tensor(rbf_kernel(dist1, dist2, gamma=1.0 / (2 * sigma ** 2))).to(device)

    # 计算 MMD 的值

    mmd = (K11.sum() / (N * (N - 1))) + (K22.sum() / (M * (M - 1))) - 2 * (K12.sum() / (N * M))
    mmd = mmd.requires_grad_(True)
    return mmd


def random_sample(dist, n, sample_sizes):
    idx = np.random.randint(0, n, size=sample_sizes)
    return dist[idx]


if __name__ == '__main__':
    # 示例用法
    # 创建样本张量

    samples_p = torch.randn(100, 10)  # P 分布样本，大小为 100x10
    samples_q = torch.randn(200, 10)  # Q 分布样本，大小为 200x10
    samples_p = samples_p.numpy()
    random_sample(samples_p, 100, 10)

    sigma = 1.0  # 高斯核函数的带宽参数

    # 计算 MMD
    mmd_value = compute_mmd(samples_p, samples_q, sigma)
    mmd_value2 = compute_mmd2(samples_p, samples_q)
    # mmd_value3 = compute_mmd3(samples_p, samples_q)
    print("MMD value:", mmd_value.item())
    x_known = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_known = torch.tensor([[2.0, 3.0]])
    known_mmd = 0.0567

    # 计算MMD
    computed_mmd = compute_mmd(x_known, y_known)
    computed_mmd2 = compute_mmd2(x_known, y_known)
    # computed_mmd3 = compute_mmd3(x_known, y_known)

import torch
import torch.nn as nn
nn.CrossEntropyLoss

def compute_mmd(samples_p, samples_q, sigma=1.0):
    n_p = samples_p.size(0)
    n_q = samples_q.size(0)

    # 计算核矩阵 K_p
    pairwise_distances_p = torch.cdist(samples_p, samples_p, p=2)
    K_p = torch.exp(-pairwise_distances_p ** 2 / (2 * sigma ** 2))

    # 计算核矩阵 K_q
    pairwise_distances_q = torch.cdist(samples_q, samples_q, p=2)
    K_q = torch.exp(-pairwise_distances_q ** 2 / (2 * sigma ** 2))

    # 计算混合核矩阵 K_mix
    pairwise_distances_mix = torch.cdist(samples_p, samples_q, p=2)
    K_mix = torch.exp(-pairwise_distances_mix ** 2 / (2 * sigma ** 2))

    # 计算 MMD
    mmd = K_p.sum() / (n_p * (n_p - 1)) + K_q.sum() / (n_q * (n_q - 1)) - 2 * K_mix.sum() / (n_p * n_q)

    return mmd


if __name__ == '__main__':
    # 示例用法
    # 创建样本张量
    samples_p = torch.randn(100, 10)  # P 分布样本，大小为 100x10
    samples_q = torch.randn(200, 10)  # Q 分布样本，大小为 200x10
    sigma = 1.0  # 高斯核函数的带宽参数

    # 计算 MMD
    mmd_value = compute_mmd(samples_p, samples_q, sigma)
    print("MMD value:", mmd_value.item())

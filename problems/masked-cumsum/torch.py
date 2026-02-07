import torch

def solution(input, mask, output, batch_size, length, dim):
    output[:] = torch.cumsum(input * mask, dim=dim)

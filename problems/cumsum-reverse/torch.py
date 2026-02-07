import torch

def solution(input, output, batch_size, length, dim):
    output[:] = torch.cumsum(input.flip(dim), dim=dim).flip(dim)

import torch

def solution(input, output, n, m):
    output[:] = input / (1 + torch.abs(input)) 
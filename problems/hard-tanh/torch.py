import torch

def solution(input, output, n, m):
    output[:] = torch.nn.functional.hardtanh(input, min_val=-1.0, max_val=1.0) 
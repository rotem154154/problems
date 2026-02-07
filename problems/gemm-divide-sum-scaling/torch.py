import torch

def solution(input, weight, output, scaling_factor, batch_size, input_size, hidden_size):
    out = torch.matmul(input, weight.T)
    out = out / 2.0
    out = torch.sum(out, dim=1, keepdim=True)
    output[:] = out * scaling_factor

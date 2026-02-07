import torch

def solution(input, weight, bias, output, scaling_factor, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    original = out
    out = out * scaling_factor
    output[:] = out + original

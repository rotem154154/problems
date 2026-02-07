import torch

def solution(input, weight, bias, output, scaling_factor, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * torch.sigmoid(out)
    output[:] = out * scaling_factor

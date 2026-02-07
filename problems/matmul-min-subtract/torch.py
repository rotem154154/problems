import torch

def solution(input, weight, bias, constant, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.min(out, constant)
    output[:] = out - constant

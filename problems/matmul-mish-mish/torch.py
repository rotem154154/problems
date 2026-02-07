import torch

def solution(input, weight, bias, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.mish(out)
    output[:] = torch.nn.functional.mish(out)

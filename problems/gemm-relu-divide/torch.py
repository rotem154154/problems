import torch

def solution(input, weight, bias, output, divisor, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.relu(out)
    output[:] = out / divisor

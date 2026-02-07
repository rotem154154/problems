import torch

def solution(input, weight, bias, output, multiplier, negative_slope,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * multiplier
    output[:] = torch.nn.functional.leaky_relu(out, negative_slope=negative_slope)

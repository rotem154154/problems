import torch

def solution(input, weight, bias, output, divisor, clamp_min, clamp_max, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * torch.sigmoid(out)
    out = out / divisor
    out = torch.clamp(out, min=clamp_min, max=clamp_max)
    out = torch.tanh(out)
    output[:] = torch.clamp(out, min=clamp_min, max=clamp_max)

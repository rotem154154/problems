import torch

def solution(input, weight, bias, output, scaling_factor, hardtanh_min, hardtanh_max,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * scaling_factor
    out = torch.nn.functional.hardtanh(out, min_val=hardtanh_min, max_val=hardtanh_max)
    output[:] = torch.nn.functional.gelu(out)

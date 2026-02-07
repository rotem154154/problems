import torch

def solution(input, weight, bias, gn_weight, gn_bias, output, num_groups, hardtanh_min, hardtanh_max,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    output[:] = torch.nn.functional.hardtanh(out, min_val=hardtanh_min, max_val=hardtanh_max)

import torch

def solution(input, weight, bias, add_bias, gn_weight, gn_bias, output, num_groups,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out + add_bias
    out = torch.nn.functional.hardtanh(out)
    out = torch.nn.functional.mish(out)
    output[:] = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)

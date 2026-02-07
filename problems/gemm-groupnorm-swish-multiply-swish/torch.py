import torch

def solution(input, weight, bias, gn_weight, gn_bias, multiply_weight, output,
            num_groups, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    out = out * torch.sigmoid(out)
    out = out * multiply_weight
    output[:] = out * torch.sigmoid(out)

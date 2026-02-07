import torch

def solution(input, weight, linear_bias, gn_weight, gn_bias, add_bias, output,
            num_groups, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, linear_bias)
    out = out.view(out.shape[0], out.shape[1], 1, 1)
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    out = torch.min(out, dim=1, keepdim=True)[0]
    output[:] = out + add_bias

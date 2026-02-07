import torch

def solution(input, weight, bias, gn_weight, gn_bias, output, num_groups,
            batch_size, input_size, hidden_size):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
    output[:] = out + out

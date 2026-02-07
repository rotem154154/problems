import torch

def solution(input, weight, bias, output, subtract_value, multiply_value,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out - subtract_value
    out = out * multiply_value
    output[:] = torch.relu(out)

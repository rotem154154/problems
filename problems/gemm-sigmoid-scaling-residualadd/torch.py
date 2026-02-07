import torch

def solution(input, weight, bias, output, scaling_factor, batch_size, input_size, hidden_size):
    out = torch.nn.functional.linear(input, weight, bias)
    original = out
    out = torch.sigmoid(out)
    out = out * scaling_factor
    output[:] = out + original

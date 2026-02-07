import torch

def solution(input, weight, bias, add_value, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out + add_value
    out = out * torch.sigmoid(out)
    out = torch.tanh(out)
    out = torch.nn.functional.gelu(out)
    output[:] = torch.nn.functional.hardtanh(out, min_val=-1.0, max_val=1.0)

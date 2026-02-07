import torch

def solution(input, weight, bias, subtract, output, batch_size, in_features, out_features):
    original = input
    out = torch.nn.functional.linear(input, weight, bias)
    out = out - subtract
    out = torch.mean(out, dim=1, keepdim=True)
    out = torch.logsumexp(out, dim=1, keepdim=True)
    out = torch.nn.functional.gelu(out)
    output[:] = out + original

import torch

def solution(input, weight, bias, output, max_dim, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.max(out, dim=max_dim, keepdim=True).values
    out = out - out.mean(dim=1, keepdim=True)
    output[:] = torch.nn.functional.gelu(out)

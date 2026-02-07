import torch

def solution(input, weight, bias, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.logsumexp(out, dim=1, keepdim=True)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
    out = torch.nn.functional.gelu(out)
    output[:] = torch.nn.functional.gelu(out)

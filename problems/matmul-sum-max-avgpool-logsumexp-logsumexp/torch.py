import torch

def solution(input, weight, bias, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.sum(out, dim=1, keepdim=True)
    out = torch.max(out, dim=1, keepdim=True)[0]
    out = torch.mean(out, dim=1, keepdim=True)
    out = torch.logsumexp(out, dim=1, keepdim=True)
    output[:] = torch.logsumexp(out, dim=1, keepdim=True)

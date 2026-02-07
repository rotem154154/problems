import torch

def solution(input, weight, bias, output, kernel_size, scale_factor,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.max_pool1d(out.unsqueeze(1), kernel_size=kernel_size).squeeze(1)
    out = torch.sum(out, dim=1)
    output[:] = out * scale_factor

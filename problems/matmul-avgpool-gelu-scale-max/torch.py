import torch

def solution(input, weight, bias, output, pool_kernel, scale_factor,
            batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.avg_pool1d(out.unsqueeze(1), kernel_size=pool_kernel).squeeze(1)
    out = torch.nn.functional.gelu(out)
    out = out * scale_factor
    output[:] = torch.max(out, dim=1).values

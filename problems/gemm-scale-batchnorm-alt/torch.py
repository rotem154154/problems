import torch

def solution(input, weight, bias, scale, bn_weight, bn_bias, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * scale
    running_mean = torch.zeros(out.shape[1], device=out.device, dtype=out.dtype)
    running_var = torch.ones(out.shape[1], device=out.device, dtype=out.dtype)
    output[:] = torch.nn.functional.batch_norm(
        out,
        running_mean,
        running_var,
        weight=bn_weight,
        bias=bn_bias,
        training=True,
    )

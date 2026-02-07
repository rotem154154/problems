import torch

def solution(input, weight, bias, bn_weight, bn_bias, scale, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    running_mean = torch.zeros(out.shape[1], device=out.device, dtype=out.dtype)
    running_var = torch.ones(out.shape[1], device=out.device, dtype=out.dtype)
    out = torch.nn.functional.batch_norm(
        out,
        running_mean,
        running_var,
        weight=bn_weight,
        bias=bn_bias,
        training=True,
    )
    out = scale * out
    output[:] = torch.softmax(out, dim=1)

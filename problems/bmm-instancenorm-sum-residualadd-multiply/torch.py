import torch

def solution(input, residual, weight, bias, output, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.instance_norm(
        out.unsqueeze(1).unsqueeze(1),
        running_mean=None,
        running_var=None,
        use_input_stats=True,
    ).squeeze(1).squeeze(1)
    out = out + residual
    output[:] = out * residual

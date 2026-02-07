import torch

def solution(input, weight, bias, output, scale_factor, clamp_min, clamp_max,
            batch_size, input_size, hidden_size):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out * scale_factor
    out = out + out
    out = torch.clamp(out, clamp_min, clamp_max)
    out = torch.logsumexp(out, dim=1, keepdim=True)
    output[:] = out * torch.nn.functional.mish(out)

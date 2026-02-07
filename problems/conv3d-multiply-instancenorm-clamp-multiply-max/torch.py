import torch

def solution(input, kernel, conv_bias, multiplier, output, clamp_min, clamp_max, batch_size, in_channels,
            out_channels, depth, height, width, kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out * multiplier
    out = torch.nn.functional.instance_norm(out, running_mean=None, running_var=None, weight=None, bias=None)
    out = torch.clamp(out, min=clamp_min, max=clamp_max)
    out = out * multiplier
    output[:] = torch.max(out, dim=1)[0]

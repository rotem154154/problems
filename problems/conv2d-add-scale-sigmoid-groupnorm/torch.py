import torch

def solution(input, kernel, bias, scale, gn_weight, gn_bias, output, batch_size, in_channels,
            out_channels, height, width, kernel_h, kernel_w, num_groups):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out + bias
    out = out * scale
    out = torch.sigmoid(out)
    output[:] = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)

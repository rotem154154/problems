import torch

def solution(input, kernel, conv_bias, gn_weight, gn_bias, scale, output, num_groups, pool_kernel,
            clamp_min, clamp_max, batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    out = out * scale
    out = torch.nn.functional.max_pool2d(out, kernel_size=pool_kernel)
    output[:] = torch.clamp(out, min=clamp_min, max=clamp_max)

import torch

def solution(input, kernel, gn_weight, gn_bias, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, num_groups):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.hardswish(out)
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    output[:] = out.mean(dim=[2, 3, 4])

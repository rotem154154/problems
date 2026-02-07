import torch

def solution(input, kernel, gn_weight, gn_bias, output, groups, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.relu(out)
    output[:] = torch.nn.functional.group_norm(out, num_groups=groups, weight=gn_weight, bias=gn_bias)

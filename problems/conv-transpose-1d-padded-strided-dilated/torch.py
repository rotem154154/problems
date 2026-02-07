import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            length, kernel_size, stride, padding, output_padding, dilation, groups):
    output[:] = torch.nn.functional.conv_transpose1d(
        input,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

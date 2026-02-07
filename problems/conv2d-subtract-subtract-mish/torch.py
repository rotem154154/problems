import torch

def solution(input, kernel, conv_bias, output, subtract_value_1, subtract_value_2,
            batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out - subtract_value_1
    out = out - subtract_value_2
    output[:] = torch.nn.functional.mish(out)

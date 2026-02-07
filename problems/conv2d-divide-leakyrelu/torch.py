import torch

def solution(input, kernel, bias, output, divisor, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out / divisor
    output[:] = torch.nn.functional.leaky_relu(out, negative_slope=0.01)

import torch

def solution(input, kernel, conv_bias, add_bias, output, scaling_factor, pool_kernel,
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
    out = torch.tanh(out)
    out = out * scaling_factor
    out = out + add_bias
    output[:] = torch.nn.functional.max_pool2d(out, kernel_size=pool_kernel)

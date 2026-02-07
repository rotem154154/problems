import torch

def solution(input, kernel, output, subtract_value, pool_kernel, batch_size, in_channels,
            out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out - subtract_value
    out = torch.nn.functional.hardswish(out)
    out = torch.nn.functional.max_pool2d(out, kernel_size=pool_kernel)
    output[:] = torch.nn.functional.mish(out)

import torch

def solution(input, kernel, output, subtract1_value, subtract2_value, pool_kernel,
            batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out - subtract1_value
    out = torch.tanh(out)
    out = out - subtract2_value
    output[:] = torch.nn.functional.avg_pool2d(out, kernel_size=pool_kernel)

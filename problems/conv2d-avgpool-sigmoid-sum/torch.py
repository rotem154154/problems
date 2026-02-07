import torch

def solution(input, kernel, bias, output, pool_kernel, batch_size, in_channels, out_channels,
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
    out = torch.nn.functional.avg_pool2d(out, kernel_size=pool_kernel)
    out = torch.sigmoid(out)
    output[:] = torch.sum(out, dim=(1, 2, 3))

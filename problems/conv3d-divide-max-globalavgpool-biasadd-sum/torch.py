import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, divisor, pool_size, sum_dim):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out / divisor
    out = torch.nn.functional.max_pool3d(out, kernel_size=pool_size)
    out = torch.nn.functional.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
    out = out + bias
    output[:] = torch.sum(out, dim=sum_dim)

import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
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
    out = torch.nn.functional.gelu(out)
    out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
    output[:] = out.squeeze(-1).squeeze(-1)

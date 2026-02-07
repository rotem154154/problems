import torch

def solution(input, kernel, output, scale_factor, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = out * scale_factor
    output[:] = torch.min(out, dim=1, keepdim=True)[0]

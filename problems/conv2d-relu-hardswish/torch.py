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
    out = torch.relu(out)
    output[:] = out * torch.clamp((out + 3.0) / 6.0, 0.0, 1.0)

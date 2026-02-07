import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
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
    out = torch.min(out, dim=1, keepdim=True)[0]
    out = torch.tanh(out)
    output[:] = torch.tanh(out)

import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, min_dim):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.min(out, dim=min_dim)[0]
    output[:] = torch.softmax(out, dim=1)

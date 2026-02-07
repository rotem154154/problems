import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv_transpose2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.mean(out, dim=(2, 3), keepdim=True)
    out = out + bias
    out = torch.logsumexp(out, dim=1, keepdim=True)
    out = torch.sum(out, dim=(2, 3))
    output[:] = out * 10.0

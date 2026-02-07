import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            length, kernel_size, stride, dilation):
    output[:] = torch.nn.functional.conv1d(
        input,
        kernel,
        bias=None,
        stride=stride,
        dilation=dilation,
        groups=1,
    )

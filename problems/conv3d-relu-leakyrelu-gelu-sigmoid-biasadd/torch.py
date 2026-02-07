import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.relu(out)
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
    out = torch.nn.functional.gelu(out)
    out = torch.sigmoid(out)
    output[:] = out + bias

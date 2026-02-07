import torch

def solution(input, kernel, conv_bias, sum_tensor, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
    out = out + sum_tensor
    out = torch.clamp(out, min=-1.0, max=1.0)
    output[:] = torch.nn.functional.gelu(out)
